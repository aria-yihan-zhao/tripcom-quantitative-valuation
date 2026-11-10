import unittest

import numpy as np

from src.valuation import (
    dcf_valuation,
    monte_carlo_prices,
    reported_summary_price,
    three_stage_ddm,
    weighted_relative_valuation,
)


FCFF = [
    11252.8,
    14398.8,
    16119.9,
    18106.1,
    18635.4,
    20169.7,
    21907.0,
    23607.1,
    26159.8,
]


class ValuationTests(unittest.TestCase):
    def test_reported_summary_reproduces_target(self) -> None:
        price = reported_summary_price(
            pv_forecast_fcff=113_312.8,
            pv_terminal_value=297_756.4,
            net_debt=-20_382.0,
            shares_outstanding=689.0,
        )
        self.assertAlmostEqual(price, 626.20, places=1)

    def test_standard_dcf_is_in_expected_range(self) -> None:
        result = dcf_valuation(
            FCFF,
            wacc=0.0792,
            terminal_growth=0.03,
            net_debt=-20_382.0,
            shares_outstanding=689.0,
        )
        self.assertAlmostEqual(result.implied_price, 594.33, places=1)
        self.assertGreater(result.terminal_value_share, 0.70)

    def test_wacc_must_exceed_terminal_growth(self) -> None:
        with self.assertRaises(ValueError):
            dcf_valuation(
                FCFF,
                wacc=0.03,
                terminal_growth=0.03,
                net_debt=0,
                shares_outstanding=1,
            )

    def test_monte_carlo_is_seeded_and_finite(self) -> None:
        kwargs = dict(
            simulations=200,
            base_wacc=0.0792,
            wacc_std=0.005,
            base_terminal_growth=0.03,
            terminal_growth_std=0.0035,
            fcff_scale_std=0.04,
            net_debt=-20_382.0,
            shares_outstanding=689.0,
            seed=7,
        )
        first = monte_carlo_prices(FCFF, **kwargs)
        second = monte_carlo_prices(FCFF, **kwargs)
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.isfinite(first)))

    def test_three_stage_ddm_is_close_to_published_base_case(self) -> None:
        result = three_stage_ddm(
            [10.59, 15.56, 18.12, 20.84, 23.67, 26.74, 29.81, 33.35, 37.18],
            initial_cost_of_equity=0.0962,
            ending_eps=69.90,
            transition_years=9,
            starting_eps_growth=0.09,
            starting_payout_ratio=0.5318,
            stable_eps_growth=0.045,
            stable_payout_ratio=0.58,
            stable_cost_of_equity=0.091,
        )
        self.assertAlmostEqual(result.implied_price, 585.81, places=1)
        self.assertLess(abs(result.implied_price - 583.3) / 583.3, 0.01)

    def test_weighted_relative_valuation_reproduces_report(self) -> None:
        value = weighted_relative_valuation(
            [615.9, 751.6, 1055.1], [0.75, 0.20, 0.05]
        )
        self.assertAlmostEqual(value, 665.0, places=1)


if __name__ == "__main__":
    unittest.main()
