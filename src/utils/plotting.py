import re

import matplotlib.pyplot as plt
import numpy as np


# Default font sizes tuned for A4 / journal figures
FONT_SIZES = {
    "title": 16,
    "label": 14,
    "ticks": 12,
    "legend": 12,
    "row_label": 14,
}


def plot_cumulative_npv(results, scenario, n_years):
    percentiles_npv = np.percentile(results[:, :, scenario], [2.5, 50, 97.5], axis=0)
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
    ax = plt.gca()
    ax.set_xlabel("Year", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("NPV (€)", fontsize=FONT_SIZES["label"])
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
    ax.set_ylabel("NPV (€)", fontsize=FONT_SIZES["label"])
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
            ax.hist(results[:, -1, i], bins=500, density=True, alpha=0.7)
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
                ax.set_xlabel("NPV (€)", fontsize=FONT_SIZES["label"])

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
                    results[:, -1, idx],
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
                ax.set_xlabel("NPV (€)", fontsize=FONT_SIZES["label"])

            # make x-tick labels visible for all rows (they're shared per column)
            ax.tick_params(axis="x", labelbottom=True)
            # set tick label sizes for both axes
            ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

            # use scientific notation for both axes (e.g. 120 -> 1.2e2)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

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
    ax.set_xlabel("NPV change (€)", fontsize=FONT_SIZES["label"])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["ticks"])
    plt.show()


def plot_tornado_grid(sens_params, sens_array, scenarios, n_params=6, agg="max"):
    """Plot tornado charts for all scenarios in a single grid of subplots.

    Grid layout:
      - rows correspond to numeric scenario groups (e.g. 1,2,3)
      - columns correspond to letter cases (e.g. a,b,c)

    `scenarios` is expected to contain labels like ['1a','1b','1c','2a',...],
    matching the scenario axis of `sens_array` (shape: n_params x 2 x
    n_scenarios, with index 0 the +20% sensitivity and index 1 the -20%
    sensitivity, as produced by NPVSimulator.run_sensitivity_analysis).

    Each row shares a y-axis (parameter list), so a subplot's ranking can't
    vary case to case within a row. To keep that shared axis readable, only
    the `n_params` parameters most sensitive within the row are shown; the
    per-parameter sensitivity used to pick and order them is aggregated
    across the row's cases with `agg` ("max", "mean", or "sum" of the
    cases' |sens_plus - sens_minus|).
    """
    agg_funcs = {"max": np.max, "mean": np.mean, "sum": np.sum}
    agg_func = agg_funcs[agg]

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
    delta = np.abs(sens_array[:, 0, :] - sens_array[:, 1, :])  # (n_params, n_scenarios)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, max(2.5, 0.4 * n_params) * n_rows),
        squeeze=False,
        sharey="row",
    )

    legend_handles = []
    legend_labels = []

    for r, num in enumerate(numbers):
        # Scenario indices present for this row, used to rank parameters
        row_indices = [
            scenarios.index(f"{num}{let}")
            for let in letters
            if f"{num}{let}" in scenarios
        ]
        row_sensitivity = agg_func(delta[:, row_indices], axis=1)
        # Most sensitive n_params, ordered ascending so the largest bar sits
        # at the top of the barh (matches the original tornado convention)
        top_params = np.argsort(row_sensitivity)[::-1][:n_params]
        order_idx = top_params[np.argsort(row_sensitivity[top_params])]
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
                legend_labels.extend(["Parameter increase 20%", "Parameter decrease 20%"])

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
                ax.set_xlabel("NPV change (€)", fontsize=FONT_SIZES["label"])

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
