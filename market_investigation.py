# %%
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.pipelines.reserve_prices import PROCESSED_DIR
from src.utils.plotting import FONT_SIZES

HOURS_IN_WINDOW = 24 * 365  # 12 month window
DAYS_IN_WINDOW = 365
PLOT_START = "2024-01-01"

LABELS = {
    "ffr_up": "FFR up",
    "fcr_d_up": "FCR-D up",
    "fcr_d_down": "FCR-D down",
    "afrr_up": "aFRR up",
    "afrr_down": "aFRR down",
    "best_price_1h": "Best market plan, 1h battery",
    "best_price_3h": "Best market plan, 3h battery",
}
# Categorical palette slots 1-5 (light mode), assigned by market and never reordered
COLORS = {
    "ffr_up": "#2a78d6",
    "fcr_d_up": "#eb6834",
    "fcr_d_down": "#1baf7a",
    "afrr_up": "#eda100",
    "afrr_down": "#e87ba4",
    # Neutral inks: combinations of the markets, not single markets
    "best_price_1h": "#8a8985",
    "best_price_3h": "#0b0b0b",
}
DA_LABELS = {"savings_1h": "1h battery", "savings_3h": "3h battery"}
# Categorical palette slots 6-7 (light mode), distinct from the reserve markets
DA_COLORS = {"savings_1h": "#008300", "savings_3h": "#4a3aa7"}
# Components of the optimal operation in stacking order, bottom to top
COMPONENT_LABELS = {
    "da": "DA arbitrage",
    "ffr": "FFR",
    "fcr_d": "FCR-D",
    "afrr": "aFRR",
}
COMPONENT_COLORS = {
    "da": "#2a78d6",
    "ffr": "#eda100",
    "fcr_d": "#008300",
    "afrr": "#e34948",
}
BATTERIES = ["1h", "3h"]
SURFACE = "#fcfcfb"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# %% Read processed prices (€/MW per hour)
prices = pd.read_csv(
    PROCESSED_DIR / "reserve_prices_hourly.csv",
    index_col="start_time_utc",
    parse_dates=True,
)

# %% Read the best reserve market plans (best hourly revenue, €/MW per hour)
best_markets = pd.read_csv(
    PROCESSED_DIR / "best_reserve_market_hourly.csv",
    index_col="start_time_utc",
    parse_dates=True,
)

# %% 12 month windowed sums (€/MW/year)
# Windows are computed over the full history so that the first plotted points already
# cover a whole year. A sum is only reported once the whole window lies within the
# market's data.
best_prices = best_markets[["best_price_1h", "best_price_3h"]]
sums = (
    prices.join(best_prices).rolling(HOURS_IN_WINDOW, min_periods=HOURS_IN_WINDOW).sum()
)
sums = sums[PLOT_START:]
# Last window of each year, i.e. the 12 months ending on the last day of the data
print(sums.groupby(sums.index.year).last().round(0).to_string())


# %% Plotting helper
def plot_rolling_sums(sums, labels, colors, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for col in sums.columns:
        ax.plot(
            sums.index,
            sums[col],
            label=labels[col],
            color=colors[col],
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(
        axis="both", labelsize=FONT_SIZES["ticks"], colors=TEXT_SECONDARY, length=0
    )
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    plt.show()


# %% Plot reserve market sums
plot_rolling_sums(sums, LABELS, COLORS, "12 month sum of prices (€/MW/year)")

# %% Read daily day-ahead arbitrage savings (€/MW per day)
da_savings = pd.read_csv(
    PROCESSED_DIR / "da_arbitrage_savings_daily.csv",
    index_col="date",
    parse_dates=True,
)

# %% 12 month windowed sums of day-ahead arbitrage savings (€/MW/year)
da_sums = da_savings.rolling(DAYS_IN_WINDOW, min_periods=DAYS_IN_WINDOW).sum()
da_sums = da_sums[PLOT_START:]
print(da_sums.groupby(da_sums.index.year).last().round(0).to_string())

# %% Plot day-ahead arbitrage sums
plot_rolling_sums(da_sums, DA_LABELS, DA_COLORS, "12 month sum of savings (€/MW/year)")

# %% Read the daily optimal operation (€/MW per day)
operation = pd.read_csv(
    PROCESSED_DIR / "daily_optimal_operation.csv",
    index_col="date",
    parse_dates=True,
)

# %% 12 month windowed sums of the optimal operation (€/MW/year)
# One column per component (market or DA arbitrage) and battery case, e.g. "afrr_3h"
yields = operation.drop(columns=[f"mode_{b}" for b in BATTERIES])
operation_sums = yields.rolling(DAYS_IN_WINDOW, min_periods=DAYS_IN_WINDOW).sum()
operation_sums = operation_sums[PLOT_START:]
optimal_sums = pd.DataFrame(
    {
        f"optimal_{b}": operation_sums.filter(regex=f"_{b}$").sum(axis=1)
        for b in BATTERIES
    }
)
print(optimal_sums.groupby(optimal_sums.index.year).last().round(0).to_string())

# %% Plotting helper for the stacked operation components
def plot_stacked_operation(frame, ylabel, bars=False):
    """One panel per battery case, stacking the columns "<component>_<battery>"."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, facecolor=SURFACE)
    for ax, battery in zip(axes, BATTERIES):
        # The 1h battery has no aFRR
        components = [c for c in COMPONENT_LABELS if f"{c}_{battery}" in frame]
        columns = frame[[f"{c}_{battery}" for c in components]]
        labels = [COMPONENT_LABELS[c] for c in components]
        colors = [COMPONENT_COLORS[c] for c in components]
        ax.set_facecolor(SURFACE)
        if bars:
            bottom = np.zeros(len(columns))
            for col, label, color in zip(columns, labels, colors):
                ax.bar(
                    columns.index,
                    columns[col],
                    bottom=bottom,
                    width=25,  # Days
                    label=label,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=1,
                )
                bottom += columns[col].to_numpy()
        else:
            ax.stackplot(
                columns.index,
                columns.T,
                labels=labels,
                colors=colors,
                edgecolor=SURFACE,
                linewidth=1,
            )
        ax.set_title(f"{battery} battery", fontsize=FONT_SIZES["title"])
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(
            axis="both", labelsize=FONT_SIZES["ticks"], colors=TEXT_SECONDARY, length=0
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # Set after both panels are drawn, so that the shared axis covers the taller stack
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    # The 3h battery uses all components
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        labelcolor=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()


# %% Plot the optimal operation sums, stacked by market and DA arbitrage
plot_stacked_operation(operation_sums, "12 month sum of yield (€/MW/year)")

# %% Monthly yields and savings of the optimal operation (€/MW/month)
# The last month is dropped if the data does not cover all of its days
monthly = yields[PLOT_START:].resample("MS").sum()
days_covered = yields[PLOT_START:].index.to_series().resample("MS").count()
monthly = monthly[days_covered == monthly.index.days_in_month]
print(monthly.tail(3).round(0).to_string())

# %% Plot the monthly yields and savings
plot_stacked_operation(monthly, "Monthly yield (€/MW/month)", bars=True)

# %%
