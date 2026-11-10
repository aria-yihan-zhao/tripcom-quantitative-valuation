# Trip.com Quantitative Valuation

An auditable Python valuation of Trip.com Group that reproduces the core DCF
conclusion from a CFA Institute Research Challenge team report, tests the model's
timing assumptions, and quantifies valuation uncertainty.

## Headline results

- Reproduced the report's published DCF summary target of **CNY 626.2/share**
  from CNY 113.3bn of forecast-period PV, CNY 297.8bn of terminal-value PV,
  CNY 20.4bn of net cash, and 689m diluted shares.
- Rebuilt the 2025-2033 FCFF forecast and obtained **CNY 594.3/share** using a
  standard nine-period terminal-value discount.
- Traced the difference to terminal-value timing: discounting the terminal value
  for eight periods, as implied by the source-model summary, produces about
  **CNY 626.0/share**.
- Added reproducible WACC/growth sensitivity analysis and a seeded 10,000-draw
  Monte Carlo simulation.

This timing check is deliberate: the project treats valuation as an auditable
modeling exercise rather than a single target-price output.

## Methodology

The base case uses free cash flow to the firm (FCFF), a 7.92% WACC, 3.00%
terminal growth, CNY 20.382bn of net cash, and 689m diluted shares. The code:

1. discounts the annual 2025-2033 FCFF forecast;
2. calculates a Gordon-growth terminal value;
3. bridges enterprise value to equity value;
4. tests WACC and terminal-growth combinations; and
5. simulates WACC, terminal growth, and FCFF-level uncertainty with a fixed seed.

The Monte Carlo assumptions are simplified and explicitly documented. They do
not claim to recreate undisclosed workbook formulas from the original team model.

## Repository structure

```text
data/processed/             Clean model inputs and generated summaries
notebooks/                  Reproducible exploratory notebooks (in progress)
reports/figures/            Generated sensitivity and simulation charts
scripts/run_analysis.py     End-to-end analysis entry point
src/valuation.py            DCF, sensitivity, and Monte Carlo functions
tests/test_valuation.py     Unit tests for core valuation logic
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_analysis.py
python -m unittest discover -s tests -v
```

Generated outputs include:

- `reports/figures/dcf_sensitivity.png`
- `reports/figures/monte_carlo_distribution.png`
- `data/processed/sensitivity_prices.csv`
- `data/processed/monte_carlo_summary.csv`

## Data and scope

Forecast inputs are transcribed from the team's published model summary and are
stored in millions of CNY. Raw licensed reports and proprietary spreadsheets are
not included. This repository is an educational portfolio project, not investment
advice.
