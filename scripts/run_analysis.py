"""Run the reproducible Trip.com DCF, sensitivity, and Monte Carlo analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.valuation import (  # noqa: E402
    dcf_valuation,
    monte_carlo_prices,
    reported_summary_price,
    sensitivity_table,
    three_stage_ddm,
    weighted_relative_valuation,
)


WACC = 0.0792
TERMINAL_GROWTH = 0.03
NET_DEBT = -20_382.0
SHARES_OUTSTANDING = 689.0
REFERENCE_PRICE = 517.0


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "processed" / "base_case_fcff.csv")
    fcff = data["fcff_mcny"].to_numpy()
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    standard = dcf_valuation(
        fcff,
        wacc=WACC,
        terminal_growth=TERMINAL_GROWTH,
        net_debt=NET_DEBT,
        shares_outstanding=SHARES_OUTSTANDING,
    )
    source_timing = dcf_valuation(
        fcff,
        wacc=WACC,
        terminal_growth=TERMINAL_GROWTH,
        net_debt=NET_DEBT,
        shares_outstanding=SHARES_OUTSTANDING,
        terminal_discount_periods=8,
    )
    published_summary = reported_summary_price(
        pv_forecast_fcff=113_312.8,
        pv_terminal_value=297_756.4,
        net_debt=NET_DEBT,
        shares_outstanding=SHARES_OUTSTANDING,
    )

    wacc_values = np.arange(0.0632, 0.0953, 0.004)
    growth_values = np.arange(0.01, 0.0501, 0.005)
    sensitivity = sensitivity_table(
        fcff,
        wacc_values=wacc_values,
        growth_values=growth_values,
        net_debt=NET_DEBT,
        shares_outstanding=SHARES_OUTSTANDING,
    )
    sensitivity.to_csv(ROOT / "data" / "processed" / "sensitivity_prices.csv")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        sensitivity,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        cbar_kws={"label": "Implied price (CNY/share)"},
    )
    plt.xlabel("Terminal growth")
    plt.ylabel("WACC")
    plt.xticks(
        ticks=np.arange(len(sensitivity.columns)) + 0.5,
        labels=[f"{value:.1%}" for value in sensitivity.columns],
    )
    plt.yticks(
        ticks=np.arange(len(sensitivity.index)) + 0.5,
        labels=[f"{value:.1%}" for value in sensitivity.index],
        rotation=0,
    )
    plt.title("Trip.com DCF Sensitivity: WACC vs. Terminal Growth")
    plt.tight_layout()
    plt.savefig(figures / "dcf_sensitivity.png", dpi=180)
    plt.close()

    prices = monte_carlo_prices(
        fcff,
        simulations=10_000,
        base_wacc=WACC,
        wacc_std=0.005,
        base_terminal_growth=TERMINAL_GROWTH,
        terminal_growth_std=0.0035,
        fcff_scale_std=0.04,
        net_debt=NET_DEBT,
        shares_outstanding=SHARES_OUTSTANDING,
        seed=42,
    )
    summary = pd.Series(prices).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    summary.to_csv(ROOT / "data" / "processed" / "monte_carlo_summary.csv")

    plt.figure(figsize=(9, 5.5))
    sns.histplot(prices, bins=60, color="#1f77b4", alpha=0.8)
    plt.axvline(REFERENCE_PRICE, color="#c0392b", linestyle="--", label="Reference price: CNY 517")
    plt.axvline(np.median(prices), color="#1b5e20", linestyle="-", label=f"Median: CNY {np.median(prices):.0f}")
    plt.xlabel("Implied price (CNY/share)")
    plt.ylabel("Simulation count")
    plt.title("Trip.com DCF Monte Carlo Distribution (10,000 simulations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "monte_carlo_distribution.png", dpi=180)
    plt.close()

    ddm_inputs = pd.read_csv(ROOT / "data" / "processed" / "ddm_inputs.csv")
    ddm_scenarios = pd.read_csv(ROOT / "data" / "processed" / "ddm_scenarios.csv")
    modeled_ddm_prices = []
    for row in ddm_scenarios.itertuples(index=False):
        result = three_stage_ddm(
            ddm_inputs["dps_cny"],
            initial_cost_of_equity=0.0962,
            ending_eps=float(ddm_inputs["eps_cny"].iloc[-1]),
            transition_years=9,
            starting_eps_growth=0.09,
            starting_payout_ratio=float(ddm_inputs["payout_ratio"].iloc[-1]),
            stable_eps_growth=row.stable_eps_growth,
            stable_payout_ratio=row.stable_payout_ratio,
            stable_cost_of_equity=row.stable_cost_of_equity,
        )
        modeled_ddm_prices.append(result.implied_price)
    ddm_scenarios["modeled_price_cny"] = modeled_ddm_prices
    ddm_scenarios.to_csv(
        ROOT / "data" / "processed" / "ddm_scenario_results.csv", index=False
    )

    dcf_scenarios = pd.read_csv(ROOT / "data" / "processed" / "dcf_scenarios.csv")
    scenario_long = pd.concat(
        [
            dcf_scenarios[["scenario", "published_price_cny"]].assign(method="DCF"),
            ddm_scenarios[["scenario", "published_price_cny"]].assign(method="DDM"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(9, 5.5))
    sns.barplot(
        data=scenario_long,
        x="scenario",
        y="published_price_cny",
        hue="method",
        order=["Bear", "Base", "Bull"],
        palette=["#2f5597", "#70ad47"],
    )
    plt.axhline(REFERENCE_PRICE, color="#c0392b", linestyle="--", label="Reference price")
    plt.ylabel("Implied price (CNY/share)")
    plt.xlabel("")
    plt.title("Published Scenario Valuation Cross-Check")
    plt.tight_layout()
    plt.savefig(figures / "scenario_comparison.png", dpi=180)
    plt.close()

    relative = pd.read_csv(ROOT / "data" / "processed" / "relative_valuation.csv")
    relative_price = weighted_relative_valuation(
        relative["implied_price_cny"], relative["weight"]
    )

    football = pd.DataFrame(
        [
            {
                "method": "DCF scenarios",
                "low": dcf_scenarios["published_price_cny"].min(),
                "mid": float(dcf_scenarios.loc[dcf_scenarios.scenario == "Base", "published_price_cny"].iloc[0]),
                "high": dcf_scenarios["published_price_cny"].max(),
            },
            {
                "method": "DDM scenarios",
                "low": ddm_scenarios["published_price_cny"].min(),
                "mid": float(ddm_scenarios.loc[ddm_scenarios.scenario == "Base", "published_price_cny"].iloc[0]),
                "high": ddm_scenarios["published_price_cny"].max(),
            },
            {
                "method": "Relative valuation",
                "low": relative["implied_price_cny"].min(),
                "mid": relative_price,
                "high": relative["implied_price_cny"].max(),
            },
            {
                "method": "Monte Carlo (5th-95th)",
                "low": np.percentile(prices, 5),
                "mid": np.median(prices),
                "high": np.percentile(prices, 95),
            },
        ]
    )
    football.to_csv(ROOT / "data" / "processed" / "football_field.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(football))
    ax.hlines(y, football["low"], football["high"], color="#4472c4", linewidth=8, alpha=0.75)
    ax.scatter(football["mid"], y, marker="D", s=85, color="#1f1f1f", zorder=3, label="Base / weighted value")
    ax.axvline(REFERENCE_PRICE, color="#c0392b", linestyle="--", label="Reference price: CNY 517")
    ax.set_yticks(y, football["method"])
    ax.set_xlabel("Implied price (CNY/share)")
    ax.set_title("Trip.com Valuation Football Field")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures / "valuation_football_field.png", dpi=180)
    plt.close()

    print("Trip.com valuation audit")
    print(f"Standard 9-period DCF: CNY {standard.implied_price:,.2f}")
    print(f"Source-model 8-period terminal timing: CNY {source_timing.implied_price:,.2f}")
    print(f"Published DCF summary reproduction: CNY {published_summary:,.2f}")
    print(f"Monte Carlo median: CNY {np.median(prices):,.2f}")
    print(f"Monte Carlo 5th-95th percentile: CNY {np.percentile(prices, 5):,.2f} - {np.percentile(prices, 95):,.2f}")
    print(f"Probability above CNY {REFERENCE_PRICE:,.0f}: {(prices > REFERENCE_PRICE).mean():.1%}")
    print(f"Modeled base DDM: CNY {ddm_scenarios.loc[ddm_scenarios.scenario == 'Base', 'modeled_price_cny'].iloc[0]:,.2f}")
    print("Published base DDM: CNY 583.30")
    print(f"Weighted relative valuation: CNY {relative_price:,.2f}")


if __name__ == "__main__":
    main()
