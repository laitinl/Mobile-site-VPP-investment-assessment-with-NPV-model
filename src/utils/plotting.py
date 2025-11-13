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
    import re

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
