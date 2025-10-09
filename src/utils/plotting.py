import matplotlib.pyplot as plt
import numpy as np


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
    plt.xlabel("Year")
    plt.ylabel("NPV (€)")
    plt.grid(axis="y", alpha=0.5)
    plt.legend()
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

    plt.xticks(
        positions + 1,
        [f"Scenario {s}" for s in scenarios],
    )
    plt.ylabel("NPV (€)")
    plt.grid(axis="y", alpha=0.5)
    plt.legend()
    plt.show()


def plot_npv_hist_grid(site_powers, results_with_different_site_power, scenarios):
    n_site_powers = len(site_powers)
    fig, axs = plt.subplots(n_site_powers, len(scenarios), figsize=(15, 10))

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
                ax.set_title(f"Scenario {scenario}", fontweight="bold", pad=20)

            # Add ylabel only on the leftmost column
            if i == 0:
                ax.set_ylabel("Density")

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
                    fontsize=12,
                )

            # Add xlabel only on the bottom row
            if j == n_site_powers - 1:
                ax.set_xlabel("NPV (€)")

    plt.tight_layout()
    plt.show()


def plot_npv_hist_overlaid(site_powers, results_with_different_site_power, scenarios):
    n_scenarios = len(scenarios)
    fig, axs = plt.subplots(1, n_scenarios, figsize=(15, 5))

    # Handle case where there's only one scenario (axs would be 0D)
    if n_scenarios == 1:
        axs = [axs]

    # Generate distinct colors for each site power
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(site_powers)))

    # Store handles and labels for the legend (only need from first subplot)
    legend_handles = []
    legend_labels = []

    for i, scenario in enumerate(scenarios):
        ax = axs[i]

        # Plot histogram for each site power in the same subplot
        for j, results in enumerate(results_with_different_site_power):
            color = colors[j]
            hist_patch = ax.hist(
                results[:, -1, i],
                bins=100,
                density=True,
                alpha=0.6,
                color=color,
                linewidth=0.5,
            )

            # Collect legend info only from the first subplot
            if i == 0:
                legend_handles.append(hist_patch[2][0])  # Get the patch object
                legend_labels.append(f"{site_powers[j]} kW Site Mean Power")

        ax.set_title(f"Scenario {scenario}", fontweight="bold", pad=15)
        ax.set_xlabel("NPV (€)")
        ax.grid(axis="y", alpha=0.5)

        # Add ylabel only on the leftmost subplot
        if i == 0:
            ax.set_ylabel("Density")

    # Add a single legend for the entire figure at the bottom
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(site_powers),
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend at the bottom
    plt.show()
