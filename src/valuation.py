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
