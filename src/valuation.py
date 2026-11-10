"""Core valuation utilities for the Trip.com quantitative valuation project.

All monetary values are expressed in millions of CNY unless otherwise noted.
The functions are intentionally small and explicit so that every assumption is
auditable from the notebook or command-line analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DCFResult:
    """Calculated components of a free-cash-flow-to-the-firm valuation."""

    pv_forecast_fcff: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    implied_price: float
    terminal_value_share: float


@dataclass(frozen=True)
class DDMResult:
    """Calculated components of a three-stage dividend discount model."""

    pv_explicit_dividends: float
    pv_transition_dividends: float
    terminal_value: float
    pv_terminal_value: float
    implied_price: float


def _validate_inputs(
    fcff: np.ndarray,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    terminal_discount_periods: float,
) -> None:
    if fcff.ndim != 1 or fcff.size == 0:
        raise ValueError("fcff must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(fcff)):
        raise ValueError("fcff contains non-finite values")
    if wacc <= terminal_growth:
        raise ValueError("wacc must be greater than terminal growth")
    if wacc <= -1:
        raise ValueError("wacc must be greater than -100%")
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive")
    if terminal_discount_periods <= 0:
        raise ValueError("terminal_discount_periods must be positive")


def dcf_valuation(
    fcff: Iterable[float],
    *,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    terminal_discount_periods: float | None = None,
) -> DCFResult:
    """Value forecast FCFF and a Gordon-growth terminal value.

    ``net_debt`` follows the conventional definition debt minus cash. A
    negative value therefore represents net cash and increases equity value.
    By default, the terminal value is discounted by the number of forecast
    periods. ``terminal_discount_periods`` is exposed to audit alternative
    timing conventions used in a source model.
    """

    fcff_array = np.asarray(tuple(fcff), dtype=float)
    terminal_periods = (
        float(fcff_array.size)
        if terminal_discount_periods is None
        else float(terminal_discount_periods)
    )
    _validate_inputs(
        fcff_array, wacc, terminal_growth, shares_outstanding, terminal_periods
    )

    periods = np.arange(1, fcff_array.size + 1, dtype=float)
    pv_forecast_fcff = float(np.sum(fcff_array / (1.0 + wacc) ** periods))
    terminal_cash_flow = float(fcff_array[-1] * (1.0 + terminal_growth))
    terminal_value = terminal_cash_flow / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / (1.0 + wacc) ** terminal_periods
    enterprise_value = pv_forecast_fcff + pv_terminal_value
    equity_value = enterprise_value - net_debt
    implied_price = equity_value / shares_outstanding

    return DCFResult(
        pv_forecast_fcff=pv_forecast_fcff,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        implied_price=implied_price,
        terminal_value_share=pv_terminal_value / enterprise_value,
    )


def reported_summary_price(
    *,
    pv_forecast_fcff: float,
    pv_terminal_value: float,
    net_debt: float,
    shares_outstanding: float,
) -> float:
    """Reproduce a reported target price from its published DCF summary."""

    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive")
    enterprise_value = pv_forecast_fcff + pv_terminal_value
    return (enterprise_value - net_debt) / shares_outstanding


def three_stage_ddm(
    explicit_dividends: Iterable[float],
    *,
    initial_cost_of_equity: float,
    ending_eps: float,
    transition_years: int,
    starting_eps_growth: float,
    starting_payout_ratio: float,
    stable_eps_growth: float,
    stable_payout_ratio: float,
    stable_cost_of_equity: float,
) -> DDMResult:
    """Value dividends across explicit, transition, and stable stages.

    The transition linearly converges from the final explicit-period growth,
    payout, and cost of equity to stable assumptions. Dividends are expressed
    per share, so the resulting value is also a per-share price.
    """

    dividends = np.asarray(tuple(explicit_dividends), dtype=float)
    if dividends.ndim != 1 or dividends.size == 0:
        raise ValueError("explicit_dividends must be a non-empty sequence")
    if not np.all(np.isfinite(dividends)) or np.any(dividends < 0):
        raise ValueError("explicit_dividends must be finite and non-negative")
    if transition_years <= 0:
        raise ValueError("transition_years must be positive")
    if ending_eps <= 0:
        raise ValueError("ending_eps must be positive")
    if stable_cost_of_equity <= stable_eps_growth:
        raise ValueError("stable cost of equity must exceed stable EPS growth")
    if not 0 <= starting_payout_ratio <= 1 or not 0 <= stable_payout_ratio <= 1:
        raise ValueError("payout ratios must be between zero and one")

    explicit_periods = np.arange(1, dividends.size + 1, dtype=float)
    pv_explicit = float(
        np.sum(dividends / (1.0 + initial_cost_of_equity) ** explicit_periods)
    )

    growth_path = np.linspace(
        starting_eps_growth, stable_eps_growth, transition_years + 1
    )[1:]
    payout_path = np.linspace(
        starting_payout_ratio, stable_payout_ratio, transition_years + 1
    )[1:]
    cost_path = np.linspace(
        initial_cost_of_equity, stable_cost_of_equity, transition_years + 1
    )[1:]

    eps = float(ending_eps)
    discount_factor = (1.0 + initial_cost_of_equity) ** dividends.size
    pv_transition = 0.0
    for growth, payout, cost in zip(growth_path, payout_path, cost_path):
        eps *= 1.0 + growth
        transition_dividend = eps * payout
        discount_factor *= 1.0 + cost
        pv_transition += transition_dividend / discount_factor

    terminal_dividend = eps * (1.0 + stable_eps_growth) * stable_payout_ratio
    terminal_value = terminal_dividend / (stable_cost_of_equity - stable_eps_growth)
    pv_terminal = terminal_value / discount_factor
    implied_price = pv_explicit + pv_transition + pv_terminal

    return DDMResult(
        pv_explicit_dividends=pv_explicit,
        pv_transition_dividends=pv_transition,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal,
        implied_price=implied_price,
    )


def weighted_relative_valuation(
    implied_prices: Iterable[float], weights: Iterable[float]
) -> float:
    """Combine valuation-method prices using normalized non-negative weights."""

    prices = np.asarray(tuple(implied_prices), dtype=float)
    weight_array = np.asarray(tuple(weights), dtype=float)
    if prices.size == 0 or prices.shape != weight_array.shape:
        raise ValueError("implied_prices and weights must be non-empty and equal length")
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(weight_array)):
        raise ValueError("prices and weights must be finite")
    if np.any(weight_array < 0) or weight_array.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return float(np.average(prices, weights=weight_array))


def sensitivity_table(
    fcff: Iterable[float],
    *,
    wacc_values: Iterable[float],
    growth_values: Iterable[float],
    net_debt: float,
    shares_outstanding: float,
    terminal_discount_periods: float | None = None,
) -> pd.DataFrame:
    """Return implied prices across WACC and terminal-growth assumptions."""

    fcff_values = tuple(fcff)
    rows: dict[float, dict[float, float]] = {}
    for wacc in wacc_values:
        rows[float(wacc)] = {}
        for growth in growth_values:
            if wacc <= growth:
                rows[float(wacc)][float(growth)] = np.nan
                continue
            result = dcf_valuation(
                fcff_values,
                wacc=float(wacc),
                terminal_growth=float(growth),
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
                terminal_discount_periods=terminal_discount_periods,
            )
            rows[float(wacc)][float(growth)] = result.implied_price
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().sort_index(axis=1)


def monte_carlo_prices(
    fcff: Iterable[float],
    *,
    simulations: int,
    base_wacc: float,
    wacc_std: float,
    base_terminal_growth: float,
    terminal_growth_std: float,
    fcff_scale_std: float,
    net_debt: float,
    shares_outstanding: float,
    seed: int = 42,
) -> np.ndarray:
    """Simulate DCF prices under explicit, reproducible uncertainty.

    The simplified simulation varies WACC, terminal growth, and the level of
    the FCFF path. Invalid draws where WACC is too close to terminal growth are
    clipped to preserve a minimum 1 percentage point spread.
    """

    if simulations <= 0:
        raise ValueError("simulations must be positive")
    rng = np.random.default_rng(seed)
    fcff_array = np.asarray(tuple(fcff), dtype=float)
    wacc_draws = rng.normal(base_wacc, wacc_std, simulations)
    growth_draws = rng.normal(base_terminal_growth, terminal_growth_std, simulations)
    growth_draws = np.clip(growth_draws, -0.02, 0.06)
    wacc_draws = np.clip(wacc_draws, growth_draws + 0.01, 0.16)
    scale_draws = np.clip(rng.normal(1.0, fcff_scale_std, simulations), 0.70, 1.30)

    prices = np.empty(simulations, dtype=float)
    for index in range(simulations):
        result = dcf_valuation(
            fcff_array * scale_draws[index],
            wacc=float(wacc_draws[index]),
            terminal_growth=float(growth_draws[index]),
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
        )
        prices[index] = result.implied_price
    return prices
