import unittest

import numpy as np

from src.valuation import dcf_valuation, monte_carlo_prices, reported_summary_price


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


if __name__ == "__main__":
    unittest.main()
