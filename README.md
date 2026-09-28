# xprice_check

`xprice_check.py` compares daily adjusted OHLC market data from multiple providers and reports where they disagree.

It is intended as a data-quality companion to backtesting programs: detect disagreements first, then decide which source or consensus series to use. It does **not** silently repair or overwrite provider data.

## Providers

Built in:

- **Yahoo Finance**, through `yfinance`.
- **Tiingo EOD**, through Tiingo's REST API. Tiingo requires a free API token; pass `--tiingo-token` or set `TIINGO_API_TOKEN`.
- Any number of local CSV files via `--provider-file NAME=PATH`.

Yahoo is requested with `auto_adjust=True`, so Open/High/Low/Close are on the adjusted basis. `--yahoo-repair` optionally enables yfinance's price-repair logic.

Tiingo uses `adjOpen`, `adjHigh`, `adjLow`, `adjClose`, and `adjVolume` from its EOD endpoint.

## Installation

```bash
python -m pip install numpy pandas yfinance
```

Tiingo needs no additional Python package.

## Basic comparison

```bash
python xprice_check.py SPY --providers yahoo tiingo --start 2005-01-01
```

With a Tiingo token in the environment:

```bash
set TIINGO_API_TOKEN=your_token_here
python xprice_check.py SPY --providers yahoo tiingo --start 2005-01-01
```

On PowerShell:

```powershell
$env:TIINGO_API_TOKEN="your_token_here"
python xprice_check.py SPY --providers yahoo tiingo --start 2005-01-01
```


## Multiple symbols

Several symbols can be checked in one run using the same bracketed syntax as the trading programs:

```bash
python xprice_check.py [SPY TIP] --providers yahoo tiingo --start 2005-01-01
```

Plain positional symbols also work:

```bash
python xprice_check.py SPY TIP IEF --providers yahoo tiingo --start 2005-01-01
```

For larger universes, use a symbol file:

```bash
python xprice_check.py --symbols-file symbols.txt --providers yahoo tiingo --start 2005-01-01
```

The symbol file accepts one or more symbols per line; blank lines and `#` comments are ignored. Duplicate symbols are removed while preserving first-occurrence order.

Each symbol receives the same detailed provider comparison. When more than one symbol is requested, the program also prints a cross-symbol summary with provider count, all-provider common-date count, pairwise missing-date count, mismatch count, and maximum relative discrepancy.

When `--write-discrepancies` is used with multiple symbols, all discrepancies are written to one CSV with a `Symbol` column. Multi-symbol consensus output likewise includes `Symbol` and `Date` columns.

Local provider files can be wide files containing a `Date` column plus ticker columns, or a path template can contain `{symbol}` / `{SYMBOL}`:

```bash
python xprice_check.py [SPY TIP] --providers \
    --provider-file vendorA=data/{symbol}.csv \
    --provider-file vendorB=other/{SYMBOL}.csv
```


## Save provider data with `--out-dir`

Use `--out-dir` to save the normalized data returned by every provider for every symbol in the run. No additional write flag is required.

```bash
python xprice_check.py [SPY TIP] --providers yahoo tiingo --start 2005-01-01 --out-dir checked_data
```

For each symbol, the directory contains one normalized CSV per provider plus one wide outer-joined CSV containing all providers side by side:

```text
checked_data/
    SPY_yahoo.csv
    SPY_tiingo.csv
    SPY_all_providers.csv
    SPY_discrepancies.csv

    TIP_yahoo.csv
    TIP_tiingo.csv
    TIP_all_providers.csv
    TIP_discrepancies.csv
```

The wide file uses provider-prefixed columns, for example:

```csv
Date,yahoo_Open,yahoo_High,yahoo_Low,yahoo_Close,yahoo_Volume,tiingo_Open,tiingo_High,tiingo_Low,tiingo_Close,tiingo_Volume
```

It uses an **outer join on dates**, so an observation missing from one provider remains visible as a blank value instead of disappearing from the comparison. This makes `SYMBOL_all_providers.csv` convenient for direct inspection in Excel, pandas, or another analysis program.

If discrepancies exist, `SYMBOL_discrepancies.csv` is written automatically. If `--consensus median` or `--consensus mean` is requested, `SYMBOL_consensus.csv` is also written automatically. Symbols containing filename-unfriendly characters are sanitized in output filenames; for example `^GSPC` becomes `_GSPC`.

## Compare provider data with a local file

```bash
python xprice_check.py SPY --providers yahoo --provider-file frozen=spy_prices.csv
```

`--provider-file` is repeatable:

```bash
python xprice_check.py SPY --providers yahoo \
    --provider-file vendorA=a.csv \
    --provider-file vendorB=b.csv
```

## Tolerances

A value is reported only when it exceeds **both** the absolute and relative tolerances:

```bash
python xprice_check.py SPY --providers yahoo tiingo \
    --abs-tol 0.01 --rel-tol 0.0001
```

Defaults are 1 cent and 0.01%.

## Data-quality checks

For each provider the program also checks:

- duplicate dates (the last copy is retained during normalization),
- nonpositive prices,
- impossible OHLC ordering,
- absolute one-day adjusted-close returns above `--return-threshold` (default 25%),
- dates present in one provider but missing from another.

When a field disagrees across a provider pair, the program also prints diagnostics over the full pairwise common history. For each provider it shows separate **Level** and **Change** rows with:

- lag-1 autocorrelation (`ACF1`),
- median,
- mean,
- sample standard deviation,
- sample skewness,
- excess kurtosis,
- minimum, and
- maximum.

The Change row uses log differences when every nonmissing value is positive; otherwise it uses ordinary first differences. A one-day bad print that reverses on the following day often creates unusually large extrema, skewness/kurtosis, and a more negative change ACF(1). These are diagnostics rather than automatic proof that one provider is correct.

By default, each displayed discrepancy is also shown with one common trading day before and after it for both providers. Change the context window with:

```bash
--context-days 2
```

or suppress context with:

```bash
--context-days 0
```

## Yahoo repair

```bash
python xprice_check.py SPY --providers yahoo tiingo --yahoo-repair
```

This enables yfinance's built-in `repair=True` logic. The comparison still reports the resulting values rather than assuming a repair is correct.

## Consensus data

A median consensus is useful when three or more independent sources are available:

```bash
python xprice_check.py SPY \
    --providers yahoo tiingo \
    --provider-file third=vendor.csv \
    --consensus median \
    --write-consensus spy_consensus.csv
```

The program can also use `--consensus mean`. `--consensus-min-providers` controls how many nonmissing sources are required for a value.

Consensus generation is explicit; the program never modifies source data automatically.

## Save all discrepancies

```bash
python xprice_check.py SPY --providers yahoo tiingo \
    --write-discrepancies spy_discrepancies.csv
```

## Notes on redistribution

Provider licenses can restrict redistribution of historical market data. This program downloads and compares data for local analysis; review each provider's license before publishing provider data or consensus files.

## Pairwise provider ratios

When `--out-dir` is used, add `--write-ratios` to append pairwise provider ratios to each `SYMBOL_all_providers.csv` file. Raw provider columns remain first, followed by columns such as:

```text
yahoo_Close__over__tiingo_Close
yahoo_Open__over__tiingo_Open
yahoo_Volume__over__tiingo_Volume
```

Example:

```bash
python xprice_check.py [SPY TIP] --providers yahoo tiingo --start 2005-01-01 --out-dir checked_data --write-ratios
```

Ratios are written only when both provider values are present. A zero denominator produces a blank/NaN rather than infinity. With three or more providers, all provider pairs are included.


## Adjusted close-to-close return comparison

Because adjusted close-to-close returns are often the quantities ultimately used in backtests, `xprice_check` can compare those returns directly across providers:

```cmd
python xprice_check.py [SPY TIP] ^
  --providers yahoo tiingo ^
  --start 2005-01-01 ^
  --compare-returns
```

The program computes simple adjusted close-to-close returns from each provider's normalized adjusted `Close` series and reports, for each provider pair:

- the number of common return dates;
- return correlation;
- mean and median absolute return difference;
- maximum absolute return difference;
- RMSE;
- the number of return differences above `--return-diff-tol`;
- ACF(1), median, mean, sample standard deviation, skewness, excess kurtosis, minimum, and maximum for each provider's return series; and
- the largest individual return disagreements.

The default return-difference tolerance is one basis point (`0.0001`). It can be changed with, for example:

```cmd
--return-diff-tol 0.0005
```

which uses a 5-basis-point absolute difference threshold.

When `--compare-returns` is combined with `--out-dir`, each `SYMBOL_all_providers.csv` also contains columns such as:

```text
yahoo_CloseReturn
tiingo_CloseReturn
yahoo_CloseReturn__minus__tiingo_CloseReturn
```

Return differences are used rather than return ratios because ratios become unstable when the denominator return is close to zero.
