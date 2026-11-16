import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Default font sizes tuned for A4 / journal figures
FONT_SIZES = {
    "title": 16,
    "label": 14,
    "ticks": 12,
    "legend": 12,
    "row_label": 14,
}


def _to_keur(axis):
    """Show an axis holding euro values in thousand euros (k€), without e-notation."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:g}"))


def plot_cumulative_npv(npvs, scenario, n_years):
    discounted_cfs = np.append(
        npvs[:, 0, scenario].reshape(-1, 1),
        npvs[:, 1:, scenario] - npvs[:, :-1, scenario],
        axis=1,
    )
    percentiles_npv = np.percentile(npvs[:, :, scenario], [2.5, 50, 97.5], axis=0)
    percentiles_cf = np.percentile(discounted_cfs, [2.5, 50, 97.5], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(0, n_years + 1),
        percentiles_npv[1],
        label="Median",
        color="blue",
    )
    plt.plot(
        np.arange(0, n_years + 1),
        percentiles_npv[::2, :].T,
        "--",
        label="Confidence Interval 95%",
        color="blue",
    )
    plt.bar(
        np.arange(0, n_years + 1),
        percentiles_cf[1],
        width=0.6,
        label="Median Cash Flow",
        color="orange",
    )
    ax = plt.gca()
    ax.set_xlabel("Year", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("NPV (k€)", fontsize=FONT_SIZES["label"])
    _to_keur(ax.yaxis)
    ax.axhline(0, color="black", linewidth=1.5, zorder=1)
    ax.grid(axis="y", alpha=0.5)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    plt.show()


def plot_npv_boxplots(site_powers, results_with_different_site_power, scenarios):
    n_site_powers = len(site_powers)
    positions = np.arange(1, len(scenarios) * n_site_powers, n_site_powers)
    displacement = np.linspace(0.2, 1.8, n_site_powers)

    # Generate distinct colors for each site power
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_site_powers))

    plt.figure(figsize=(10, 6))
    for i, results in enumerate(results_with_different_site_power):
        color = colors[i]
        plt.boxplot(
            results[:, -1, :],
            patch_artist=True,
            boxprops=dict(facecolor=(*color[:3], 0.4), color=color),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            positions=positions + displacement[i],
            label=f"{site_powers[i]} kW Site Mean Power",
            showfliers=False,
        )

    ax = plt.gca()
    ax.set_xticks(positions + 1)
    ax.set_xticklabels(
        [f"Scenario {s}" for s in scenarios], fontsize=FONT_SIZES["ticks"]
    )
    ax.set_ylabel("NPV (k€)", fontsize=FONT_SIZES["label"])
    _to_keur(ax.yaxis)
    ax.grid(axis="y", alpha=0.5)
    ax.tick_params(axis="y", labelsize=FONT_SIZES["ticks"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    plt.show()


def plot_npv_hist_grid(site_powers, results_with_different_site_power, scenarios):
    n_site_powers = len(site_powers)
    _, axs = plt.subplots(n_site_powers, len(scenarios), figsize=(15, 10))

    # Handle case where there's only one site power (axs would be 1D)
    if n_site_powers == 1:
        axs = axs.reshape(1, -1)

    for i, scenario in enumerate(scenarios):
        for j, results in enumerate(results_with_different_site_power):
            ax = axs[j, i]
            ax.hist(results[:, -1, i] / 1e3, bins=500, density=True, alpha=0.7)
            ax.grid(axis="y", alpha=0.5)

            # Add column titles (scenarios) only on the top row
            if j == 0:
                ax.set_title(
                    f"Scenario {scenario}",
                    fontweight="bold",
                    pad=20,
                    fontsize=FONT_SIZES["title"],
                )

            # Add ylabel only on the leftmost column
            if i == 0:
                ax.set_ylabel("Density", fontsize=FONT_SIZES["label"])

            # Add row titles (site powers) only on the leftmost column
            if i == 0:
                ax.text(
                    -0.5,
                    0.5,
                    f"{site_powers[j]} kW",
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    fontweight="bold",
                    fontsize=FONT_SIZES["row_label"],
                )

            # Add xlabel only on the bottom row
            if j == n_site_powers - 1:
                ax.set_xlabel("NPV (k€)", fontsize=FONT_SIZES["label"])

            ax.ticklabel_format(style="plain", axis="both")

            # tick sizes
            ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

    plt.tight_layout()
    plt.show()


def plot_npv_hist_overlaid(site_powers, results_with_different_site_power, scenarios):
    """Plot overlaid histograms in a 2D grid.

    Grid layout:
      - rows correspond to numeric scenario groups (e.g. 1,2,3)
      - columns correspond to letter cases (e.g. a,b,c)

    `scenarios` is expected to contain labels like ['1a','1b','1c','2a',...].
    For each grid cell we look up the matching scenario and plot overlaid
    histograms for all provided `results_with_different_site_power`.
    """
    # Extract numeric row keys and letter column keys from scenario strings
    parsed = [re.match(r"(\d+)(\D+)", s) for s in scenarios]
    numbers = []
    letters = []
    for m in parsed:
        if m:
            numbers.append(m.group(1))
            letters.append(m.group(2))
    # Keep original order but unique
    numbers = list(dict.fromkeys(numbers))
    letters = list(dict.fromkeys(letters))

    n_rows = len(numbers)
    n_cols = len(letters)

    # Share x-axis within each column so column cells use the same x-scale and ticks
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False, sharex="col"
    )

    # Colors per site power
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(site_powers)))

    # Legend handles collected from the first valid cell
    legend_handles = []
    legend_labels = []

    for r, num in enumerate(numbers):
        for c, let in enumerate(letters):
            ax = axs[r, c]
            scenario_label = f"{num}{let}"

            # Find the index of scenario_label in provided scenarios
            try:
                idx = scenarios.index(scenario_label)
            except ValueError:
                # No data for this cell
                ax.axis("off")
                continue

            # Plot each site power's histogram into this cell
            for j, results in enumerate(results_with_different_site_power):
                color = colors[j]
                h = ax.hist(
                    results[:, -1, idx] / 1e3,
                    bins=100,
                    density=True,
                    alpha=0.6,
                    color=color,
                )

                # Collect legend patch from first valid cell (top-left)
                if r == 0 and c == 0:
                    # h[2] is a list of patches; take the first as representative
                    if len(h) >= 3 and len(h[2]) > 0:
                        legend_handles.append(h[2][0])
                        legend_labels.append(f"{site_powers[j]} kW")

            # Column titles: letters (cases) on top row
            if r == 0:
                ax.set_title(
                    f"Case {let}",
                    fontweight="bold",
                    pad=10,
                    fontsize=FONT_SIZES["title"],
                )

            # Row titles: scenario numbers on leftmost column
            if c == 0:
                ax.text(
                    -0.25,
                    0.5,
                    f"Scenario {num}",
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    fontweight="bold",
                    fontsize=FONT_SIZES["row_label"],
                )

            # y label only on leftmost column
            if c == 0:
                ax.set_ylabel("Density", fontsize=FONT_SIZES["label"])

            # xlabel only on bottom row
            if r == n_rows - 1:
                ax.set_xlabel("NPV (k€)", fontsize=FONT_SIZES["label"])

            # make x-tick labels visible for all rows (they're shared per column)
            ax.tick_params(axis="x", labelbottom=True)
            # set tick label sizes for both axes
            ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

            # plain tick labels, no scientific notation
            ax.ticklabel_format(style="plain", axis="both")

            ax.grid(axis="y", alpha=0.5)

    # Add single legend under the entire figure
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=max(1, len(site_powers)),
            fontsize=FONT_SIZES["legend"],
            frameon=True,
            title="Site Mean Power (applies to all cells)",
            title_fontsize=FONT_SIZES["label"],
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)  # leave room for the legend and larger fonts
    plt.show()


def plot_cumulative_npv_grid(results, scenarios):
    """Plot cumulative NPV and discounted cash flow in a 2D grid.

    Grid layout:
      - rows correspond to numeric scenario groups (e.g. 1,2,3)
      - columns correspond to letter cases (e.g. a,b,c)

    `scenarios` is expected to contain labels like ['1a','1b','1c','2a',...],
    and `results` is the NPV array of a single site mean power (shape:
    n_samples x n_years+1 x n_scenarios). Each cell matches plot_cumulative_npv:
    median and 95% interval of the cumulative NPV and of the discounted cash
    flow. The x-axis is shared per column and the y-axis per row.
    """
    parsed = [re.match(r"(\d+)(\D+)", s) for s in scenarios]
    numbers = []
    letters = []
    for m in parsed:
        if m:
            numbers.append(m.group(1))
            letters.append(m.group(2))
    numbers = list(dict.fromkeys(numbers))
    letters = list(dict.fromkeys(letters))

    n_rows = len(numbers)
    n_cols = len(letters)
    years = np.arange(results.shape[1])

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3 * n_rows),
        squeeze=False,
        sharex="col",
        sharey="row",
    )

    for r, num in enumerate(numbers):
        for c, let in enumerate(letters):
            ax = axs[r, c]
            scenario_label = f"{num}{let}"

            if scenario_label not in scenarios:
                ax.axis("off")
                continue
            idx = scenarios.index(scenario_label)

            npvs = results[:, :, idx]
            discounted_cfs = np.append(
                npvs[:, :1], npvs[:, 1:] - npvs[:, :-1], axis=1
            )
            pct_npv = np.percentile(npvs, [2.5, 50, 97.5], axis=0)
            pct_cf = np.percentile(discounted_cfs, [2.5, 50, 97.5], axis=0)

            ax.plot(years, pct_npv[1], color="blue")
            ax.plot(years, pct_npv[::2].T, "--", color="blue")
            ax.bar(years, pct_cf[1], width=0.6, color="orange")

            if r == 0:
                ax.set_title(
                    f"Case {let}",
                    fontweight="bold",
                    pad=10,
                    fontsize=FONT_SIZES["title"],
                )

            if c == 0:
                ax.text(
                    -0.3,
                    0.5,
                    f"Scenario {num}",
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    fontweight="bold",
                    fontsize=FONT_SIZES["row_label"],
                )
                ax.set_ylabel("NPV (k€)", fontsize=FONT_SIZES["label"])

            if r == n_rows - 1:
                ax.set_xlabel("Year", fontsize=FONT_SIZES["label"])

            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])
            _to_keur(ax.yaxis)
            ax.axhline(0, color="black", linewidth=1.5, zorder=1)
            ax.grid(axis="y", alpha=0.5)

    legend_handles = [
        plt.Line2D([], [], color="blue", label="Median"),
        plt.Line2D([], [], color="blue", linestyle="--", label="Confidence Interval 95%"),
        plt.Rectangle((0, 0), 1, 1, color="orange", label="Median Cash Flow"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        fontsize=FONT_SIZES["legend"],
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    plt.show()


def plot_tornado(sens_params, sens_array, scenario, title=None):
    """Plot a single tornado chart for one scenario.

    `sens_array` has shape (n_params, 2, n_scenarios) with index 0 the
    +20% sensitivity and index 1 the -20% sensitivity, as produced by
    NPVSimulator.run_sensitivity_analysis. All parameters are shown,
    sorted by the magnitude of their NPV swing.
    """
    sens_plus = sens_array[:, 0, scenario]
    sens_minus = sens_array[:, 1, scenario]
    delta = np.abs(sens_plus - sens_minus)
    order = np.argsort(delta)
    param_labels = np.asarray(sens_params)[order]
    y_pos = np.arange(len(order))

    ax = plt.gca()
    ax.grid(axis="y", alpha=0.5)
    ax.barh(y_pos, sens_plus[order], label="Parameter increase 20%")
    ax.barh(y_pos, sens_minus[order], label="Parameter decrease 20%")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_labels, fontsize=FONT_SIZES["ticks"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel("NPV change (%)", fontsize=FONT_SIZES["label"])
    _to_keur(ax.xaxis)
    ax.tick_params(axis="x", labelsize=FONT_SIZES["ticks"])
    plt.show()


def plot_tornado_grid(sens_params, sens_array, scenarios, row_params):
    """Plot tornado charts for all scenarios in a single grid of subplots.

    Grid layout:
      - rows correspond to numeric scenario groups (e.g. 1,2,3)
      - columns correspond to letter cases (e.g. a,b,c)

    `scenarios` is expected to contain labels like ['1a','1b','1c','2a',...],
    matching the scenario axis of `sens_array` (shape: n_params x 2 x
    n_scenarios, with index 0 the +20% sensitivity and index 1 the -20%
    sensitivity, as produced by NPVSimulator.run_sensitivity_analysis).

    `row_params` chooses the parameters (y-axis) shown in each row. Either:
      - a list of parameter-name lists, one per row in order, or
      - a dict mapping the row's scenario number (e.g. "1") to its list.
    Parameters are drawn top to bottom in the order given, and must be names
    from `sens_params`.
    """
    # Extract numeric row keys and letter column keys from scenario strings
    parsed = [re.match(r"(\d+)(\D+)", s) for s in scenarios]
    numbers = []
    letters = []
    for m in parsed:
        if m:
            numbers.append(m.group(1))
            letters.append(m.group(2))
    numbers = list(dict.fromkeys(numbers))
    letters = list(dict.fromkeys(letters))

    n_rows = len(numbers)
    n_cols = len(letters)

    sens_params = np.asarray(sens_params)
    param_index = {name: i for i, name in enumerate(sens_params)}

    if isinstance(row_params, dict):
        row_params = [row_params[num] for num in numbers]
    elif len(row_params) != n_rows:
        raise ValueError(
            f"row_params has {len(row_params)} entries but there are {n_rows} rows"
        )
    for names in row_params:
        unknown = [n for n in names if n not in param_index]
        if unknown:
            raise ValueError(f"Unknown sensitivity parameters: {unknown}")
    max_row_params = max(len(names) for names in row_params)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, max(2.5, 0.4 * max_row_params) * n_rows),
        squeeze=False,
        sharey="row",
        sharex="row",
    )

    legend_handles = []
    legend_labels = []

    for r, num in enumerate(numbers):
        # Reverse so the first listed parameter sits at the top of the barh
        order_idx = np.array([param_index[n] for n in row_params[r]][::-1], dtype=int)
        y_pos = np.arange(len(order_idx))
        param_labels = sens_params[order_idx]

        for c, let in enumerate(letters):
            ax = axs[r, c]
            scenario_label = f"{num}{let}"

            if scenario_label not in scenarios:
                ax.axis("off")
                continue

            idx = scenarios.index(scenario_label)
            ax.grid(axis="y", alpha=0.5)
            h_plus = ax.barh(
                y_pos, sens_array[order_idx, 0, idx], label="Parameter increase 20%"
            )
            h_minus = ax.barh(
                y_pos, sens_array[order_idx, 1, idx], label="Parameter decrease 20%"
            )

            if r == 0 and c == 0:
                legend_handles.extend([h_plus, h_minus])
                legend_labels.extend(
                    ["Parameter increase 20%", "Parameter decrease 20%"]
                )

            # Column titles: letters (cases) on top row
            if r == 0:
                ax.set_title(
                    f"Case {let}",
                    fontweight="bold",
                    pad=10,
                    fontsize=FONT_SIZES["title"],
                )

            # y ticks/labels only on leftmost column (rest share the axis)
            if c == 0:
                ax.set_yticks(y_pos)
                ax.set_yticklabels(param_labels, fontsize=FONT_SIZES["ticks"])
                ax.text(
                    -0.6,
                    0.5,
                    f"Scenario {num}",
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    fontweight="bold",
                    fontsize=FONT_SIZES["row_label"],
                )
            else:
                ax.tick_params(axis="y", labelleft=False)

            # xlabel only on bottom row
            if r == n_rows - 1:
                ax.set_xlabel("NPV change (%)", fontsize=FONT_SIZES["label"])

            _to_keur(ax.xaxis)
            ax.tick_params(axis="x", labelsize=FONT_SIZES["ticks"])

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            fontsize=FONT_SIZES["legend"],
            frameon=True,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, left=0.12)
    plt.show()
