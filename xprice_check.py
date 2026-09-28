#!/usr/bin/env python3
"""Compare daily OHLC price data from multiple providers.

Built-in network providers:
  yahoo   - via yfinance
  tiingo  - via Tiingo EOD REST API (requires TIINGO_API_TOKEN or --tiingo-token)

Local CSV providers may be supplied with --provider-file NAME=PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FIELDS = ("Open", "High", "Low", "Close")


@dataclass
class ProviderData:
    name: str
    frame: pd.DataFrame


def parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:  # pragma: no cover - argparse wrapper
        raise argparse.ArgumentTypeError(f"invalid date {value!r}") from exc


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed adjusted OHLC[V] frame with canonical columns."""
    if frame.empty:
        return pd.DataFrame(columns=[*FIELDS, "Volume"])
    out = frame.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        out = out.set_index("Date")
    else:
        idx = pd.to_datetime(out.index, utc=True, errors="coerce")
        out.index = idx.tz_localize(None).normalize()
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]

    aliases = {
        "Adj Open": "Open", "adjOpen": "Open", "adjusted_open": "Open",
        "Adj High": "High", "adjHigh": "High", "adjusted_high": "High",
        "Adj Low": "Low", "adjLow": "Low", "adjusted_low": "Low",
        "Adj Close": "Close", "adjClose": "Close", "adjusted_close": "Close",
        "Adj Volume": "Volume", "adjVolume": "Volume", "adjusted_volume": "Volume",
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    }
    renamed = {}
    for col in out.columns:
        if col in aliases:
            renamed[col] = aliases[col]
    if renamed:
        out = out.rename(columns=renamed)

    # Some files contain raw OHLC + Adj Close. Convert all OHLC using the
    # same adjustment factor so comparisons remain internally consistent.
    if "Adj Close" in frame.columns and "Close" in frame.columns:
        raw = frame.copy()
        if "Date" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["Date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
            raw = raw.set_index("Date")
        else:
            idx = pd.to_datetime(raw.index, utc=True, errors="coerce")
            raw.index = idx.tz_localize(None).normalize()
        raw = raw.sort_index()
        factor = pd.to_numeric(raw["Adj Close"], errors="coerce") / pd.to_numeric(raw["Close"], errors="coerce")
        for field in ("Open", "High", "Low", "Close"):
            if field in raw.columns:
                out[field] = pd.to_numeric(raw[field], errors="coerce") * factor

    keep = [c for c in [*FIELDS, "Volume"] if c in out.columns]
    out = out[keep].apply(pd.to_numeric, errors="coerce")
    return out


def fetch_yahoo(symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None, repair: bool) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Yahoo provider requires yfinance: python -m pip install yfinance") from exc
    kwargs = dict(
        tickers=symbol,
        start=start.strftime("%Y-%m-%d") if start is not None else None,
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end is not None else None,
        interval="1d",
        auto_adjust=True,
        repair=repair,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    frame = yf.download(**kwargs)
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")
    return normalize_frame(frame)


def fetch_tiingo(symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None, token: str | None) -> pd.DataFrame:
    token = token or os.environ.get("TIINGO_API_TOKEN")
    if not token:
        raise RuntimeError("Tiingo provider requires --tiingo-token or TIINGO_API_TOKEN")
    params = {}
    if start is not None:
        params["startDate"] = start.strftime("%Y-%m-%d")
    if end is not None:
        params["endDate"] = end.strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{urllib.parse.quote(symbol)}/prices"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Tiingo request failed for {symbol}: {exc}") from exc
    if not payload:
        raise RuntimeError(f"Tiingo returned no data for {symbol}")
    raw = pd.DataFrame(payload)
    cols = {
        "date": "Date", "adjOpen": "Open", "adjHigh": "High",
        "adjLow": "Low", "adjClose": "Close", "adjVolume": "Volume",
    }
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Tiingo response missing fields: {', '.join(missing)}")
    return normalize_frame(raw[list(cols)].rename(columns=cols))


def read_local_csv(path: str, symbol: str | None = None) -> pd.DataFrame:
    if symbol is not None:
        path = path.replace("{symbol}", symbol).replace("{SYMBOL}", symbol.upper())
    frame = pd.read_csv(path)
    # Support xma-style wide close files only when a single symbol column is requested.
    if symbol and symbol in frame.columns and not any(c in frame.columns for c in FIELDS):
        return normalize_frame(frame[["Date", symbol]].rename(columns={symbol: "Close"}))
    return normalize_frame(frame)


def parse_provider_file(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--provider-file must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--provider-file must be NAME=PATH")
    return name, path


def load_provider(name: str, symbol: str, args) -> ProviderData:
    key = name.lower()
    if key == "yahoo":
        frame = fetch_yahoo(symbol, args.start, args.end, args.yahoo_repair)
    elif key == "tiingo":
        frame = fetch_tiingo(symbol, args.start, args.end, args.tiingo_token)
    else:
        raise ValueError(f"unknown network provider {name!r}")
    return ProviderData(name=name, frame=frame)


def validate_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(FIELDS).issubset(frame.columns):
        return pd.DataFrame(columns=["reason"])
    f = frame[list(FIELDS)]
    reasons = pd.Series("", index=f.index, dtype=object)
    invalid = (f["High"] < f[["Open", "Close", "Low"]].max(axis=1)) | (f["Low"] > f[["Open", "Close", "High"]].min(axis=1))
    nonpositive = (f <= 0).any(axis=1)
    reasons.loc[invalid] = "invalid OHLC ordering"
    reasons.loc[nonpositive] = np.where(reasons.loc[nonpositive].eq(""), "nonpositive price", reasons.loc[nonpositive] + "; nonpositive price")
    return reasons[reasons.ne("")].rename("reason").to_frame()


def suspicious_returns(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if "Close" not in frame:
        return pd.DataFrame(columns=["Close", "Return"])
    ret = frame["Close"].pct_change()
    mask = ret.abs() > threshold
    return pd.DataFrame({"Close": frame.loc[mask, "Close"], "Return": ret.loc[mask]})


def pairwise_compare(a: ProviderData, b: ProviderData, abs_tol: float, rel_tol: float) -> tuple[pd.DataFrame, dict]:
    common = a.frame.index.intersection(b.frame.index).sort_values()
    missing_a = b.frame.index.difference(a.frame.index)
    missing_b = a.frame.index.difference(b.frame.index)
    rows = []
    compared_fields = [f for f in [*FIELDS, "Volume"] if f in a.frame.columns and f in b.frame.columns]
    for field in compared_fields:
        av = a.frame.loc[common, field]
        bv = b.frame.loc[common, field]
        valid = av.notna() & bv.notna()
        av = av[valid]
        bv = bv[valid]
        diff = av - bv
        abs_diff = diff.abs()
        denom = pd.concat([av.abs(), bv.abs()], axis=1).max(axis=1).replace(0, np.nan)
        rel_diff = abs_diff / denom
        bad = (abs_diff > abs_tol) & (rel_diff > rel_tol)
        for date in av.index[bad]:
            rows.append({
                "Date": date,
                "Field": field,
                a.name: av.loc[date],
                b.name: bv.loc[date],
                "AbsDiff": abs_diff.loc[date],
                "RelDiff": rel_diff.loc[date],
            })
    discrepancies = pd.DataFrame(rows)
    if not discrepancies.empty:
        discrepancies = discrepancies.sort_values(["RelDiff", "AbsDiff"], ascending=False)
    stats = {
        "common_dates": len(common),
        "missing_from_first": len(missing_a),
        "missing_from_second": len(missing_b),
        "missing_first_dates": missing_a,
        "missing_second_dates": missing_b,
        "compared_fields": compared_fields,
        "discrepancies": len(discrepancies),
    }
    return discrepancies, stats



def adjusted_close_returns(provider: ProviderData) -> pd.Series:
    """Return simple adjusted close-to-close returns for one provider."""
    if "Close" not in provider.frame.columns:
        return pd.Series(dtype=float, name=provider.name)
    close = pd.to_numeric(provider.frame["Close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    ret.name = provider.name
    return ret


def compare_adjusted_returns(a: ProviderData, b: ProviderData, diff_tol: float) -> tuple[pd.DataFrame, dict]:
    """Compare adjusted close-to-close simple returns on common return dates."""
    ar = adjusted_close_returns(a)
    br = adjusted_close_returns(b)
    common = ar.dropna().index.intersection(br.dropna().index).sort_values()
    ar = ar.reindex(common)
    br = br.reindex(common)
    valid = ar.notna() & br.notna()
    ar = ar[valid]
    br = br[valid]
    diff = ar - br
    abs_diff = diff.abs()
    mismatches = abs_diff > diff_tol
    rows = pd.DataFrame({
        "Date": ar.index[mismatches],
        f"{a.name}Return": ar[mismatches].to_numpy(),
        f"{b.name}Return": br[mismatches].to_numpy(),
        "ReturnDiff": diff[mismatches].to_numpy(),
        "AbsReturnDiff": abs_diff[mismatches].to_numpy(),
    })
    corr = float(ar.corr(br)) if len(ar) >= 2 else float("nan")
    stats = {
        "common_return_dates": len(ar),
        "mismatches": int(mismatches.sum()),
        "correlation": corr,
        "mean_abs_diff": float(abs_diff.mean()) if len(abs_diff) else float("nan"),
        "median_abs_diff": float(abs_diff.median()) if len(abs_diff) else float("nan"),
        "max_abs_diff": float(abs_diff.max()) if len(abs_diff) else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else float("nan"),
    }
    return rows.sort_values("AbsReturnDiff", ascending=False), stats


def return_diagnostics(provider: ProviderData, common_dates: pd.DatetimeIndex | None = None) -> dict[str, float]:
    """Descriptive statistics for adjusted close-to-close returns."""
    ret = adjusted_close_returns(provider)
    if common_dates is not None:
        ret = ret.reindex(common_dates)
    stats = distribution_stats(ret)
    return {"Provider": provider.name, "ACF1": acf1(ret), **stats}


def format_return_diagnostics(providers: list[ProviderData], common_dates: pd.DatetimeIndex) -> str:
    table = pd.DataFrame([return_diagnostics(p, common_dates) for p in providers])
    if table.empty:
        return ""
    out = table.copy()
    def fmt(x):
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.6g}"
    for col in ("ACF1", "Median", "Mean", "SD", "Skew", "ExKurt", "Min", "Max"):
        out[col] = out[col].map(fmt)
    return out.to_string(index=False)


def consensus_frame(providers: list[ProviderData], method: str, min_providers: int) -> pd.DataFrame:
    all_dates = providers[0].frame.index
    for p in providers[1:]:
        all_dates = all_dates.union(p.frame.index)
    all_dates = all_dates.sort_values()
    result = pd.DataFrame(index=all_dates)
    fields = sorted(set().union(*(p.frame.columns for p in providers)))
    for field in fields:
        columns = [p.frame[field].reindex(all_dates).rename(p.name) for p in providers if field in p.frame]
        if not columns:
            continue
        panel = pd.concat(columns, axis=1)
        enough = panel.notna().sum(axis=1) >= min_providers
        if method == "median":
            vals = panel.median(axis=1, skipna=True)
        elif method == "mean":
            vals = panel.mean(axis=1, skipna=True)
        else:  # pragma: no cover - argparse restricts
            raise ValueError(method)
        result[field] = vals.where(enough)
    return result.dropna(how="all")



def acf1(values: pd.Series) -> float:
    """Return lag-1 autocorrelation, or NaN when it is not estimable."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 3:
        return float("nan")
    a = x.iloc[:-1].to_numpy(dtype=float)
    b = x.iloc[1:].to_numpy(dtype=float)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def transformed_changes(values: pd.Series) -> tuple[pd.Series, str]:
    """Return log differences for positive data, otherwise ordinary differences."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return pd.Series(dtype=float), "diff"
    if (x > 0).all():
        return np.log(x).diff().dropna(), "log diff"
    return x.diff().dropna(), "diff"


def distribution_stats(values: pd.Series) -> dict[str, float]:
    """Return descriptive sample statistics for a numeric series."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {k: float("nan") for k in ("Median", "Mean", "SD", "Skew", "ExKurt", "Min", "Max")}
    return {
        "Median": float(x.median()),
        "Mean": float(x.mean()),
        "SD": float(x.std(ddof=1)) if len(x) >= 2 else float("nan"),
        "Skew": float(x.skew()) if len(x) >= 3 else float("nan"),
        "ExKurt": float(x.kurt()) if len(x) >= 4 else float("nan"),
        "Min": float(x.min()),
        "Max": float(x.max()),
    }


def field_diagnostics(a: ProviderData, b: ProviderData, field: str) -> pd.DataFrame:
    """Summarize levels and changes for one field over pairwise common dates."""
    common = a.frame.index.intersection(b.frame.index).sort_values()
    rows = []
    for provider in (a, b):
        values = provider.frame[field].reindex(common)
        changes, method = transformed_changes(values)
        for series_name, series, method_name in (
            ("Level", values, "level"),
            ("Change", changes, method),
        ):
            stats = distribution_stats(series)
            rows.append({
                "Provider": provider.name,
                "Series": series_name,
                "Method": method_name,
                "ACF1": acf1(series),
                **stats,
            })
    return pd.DataFrame(rows)


def format_diagnostics_table(table: pd.DataFrame) -> str:
    """Format provider diagnostics compactly for console output."""
    if table.empty:
        return ""
    out = table.copy()
    def fmt(x):
        if pd.isna(x):
            return "n/a"
        ax = abs(float(x))
        if ax != 0 and (ax >= 1e6 or ax < 1e-4):
            return f"{x:.5e}"
        return f"{x:.6g}"
    for col in ("ACF1", "Median", "Mean", "SD", "Skew", "ExKurt", "Min", "Max"):
        out[col] = out[col].map(fmt)
    return out.to_string(index=False)


def discrepancy_context(a: ProviderData, b: ProviderData, field: str, date: pd.Timestamp, context_days: int) -> pd.DataFrame:
    """Return pairwise common-date values around one discrepant observation."""
    common = a.frame.index.intersection(b.frame.index).sort_values()
    if date not in common:
        return pd.DataFrame()
    loc = common.get_loc(date)
    if not isinstance(loc, (int, np.integer)):
        return pd.DataFrame()
    start = max(0, int(loc) - context_days)
    stop = min(len(common), int(loc) + context_days + 1)
    dates = common[start:stop]
    return pd.DataFrame({
        "Date": dates,
        a.name: a.frame[field].reindex(dates).to_numpy(),
        b.name: b.frame[field].reindex(dates).to_numpy(),
    })

def print_dates(label: str, dates: pd.DatetimeIndex, max_rows: int) -> None:
    if len(dates) == 0:
        return
    shown = dates[:max_rows]
    print(f"  {label}: {len(dates)}")
    print("    " + ", ".join(d.strftime("%Y-%m-%d") for d in shown))
    if len(dates) > len(shown):
        print(f"    ... {len(dates) - len(shown)} more")


def parse_symbols(tokens: Iterable[str], symbols_file: str | None = None) -> list[str]:
    """Parse plain or bracketed ticker tokens and an optional symbol file."""
    values: list[str] = []
    for token in tokens:
        cleaned = token.replace("[", " ").replace("]", " ").replace(",", " ")
        values.extend(part.strip().upper() for part in cleaned.split() if part.strip())
    if symbols_file:
        for raw in Path(symbols_file).read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            cleaned = line.replace("[", " ").replace("]", " ").replace(",", " ")
            values.extend(part.strip().upper() for part in cleaned.split() if part.strip())
    return list(dict.fromkeys(values))


def all_provider_common_dates(providers: list[ProviderData]) -> pd.DatetimeIndex:
    if not providers:
        return pd.DatetimeIndex([])
    common = providers[0].frame.index
    for pvd in providers[1:]:
        common = common.intersection(pvd.frame.index)
    return common.sort_values()


def safe_symbol_filename(symbol: str) -> str:
    """Return a filesystem-friendly symbol string."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in symbol.strip())
    return safe or "symbol"


def all_providers_wide_frame(providers: list[ProviderData], write_ratios: bool = False, compare_returns: bool = False) -> pd.DataFrame:
    """Outer-join normalized provider data, optionally appending pairwise ratios."""
    pieces = []
    for provider in providers:
        renamed = provider.frame.rename(columns={col: f"{provider.name}_{col}" for col in provider.frame.columns})
        pieces.append(renamed)
    if not pieces:
        return pd.DataFrame()
    wide = pd.concat(pieces, axis=1, join="outer").sort_index()
    if compare_returns:
        return_columns = {}
        for provider in providers:
            if "Close" in provider.frame.columns:
                return_columns[f"{provider.name}_CloseReturn"] = adjusted_close_returns(provider).reindex(wide.index)
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                a = providers[i]
                b = providers[j]
                if "Close" in a.frame.columns and "Close" in b.frame.columns:
                    ar = adjusted_close_returns(a).reindex(wide.index)
                    br = adjusted_close_returns(b).reindex(wide.index)
                    return_columns[f"{a.name}_CloseReturn__minus__{b.name}_CloseReturn"] = ar - br
        if return_columns:
            wide = pd.concat([wide, pd.DataFrame(return_columns, index=wide.index)], axis=1)
    if write_ratios:
        ratio_columns = {}
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                a = providers[i]
                b = providers[j]
                shared_fields = [
                    field for field in [*FIELDS, "Volume"]
                    if field in a.frame.columns and field in b.frame.columns
                ]
                for field in shared_fields:
                    numerator = a.frame[field].reindex(wide.index)
                    denominator = b.frame[field].reindex(wide.index)
                    denominator = denominator.where(denominator != 0.0)
                    name = f"{a.name}_{field}__over__{b.name}_{field}"
                    ratio_columns[name] = numerator / denominator
        if ratio_columns:
            wide = pd.concat([wide, pd.DataFrame(ratio_columns, index=wide.index)], axis=1)
    wide.index.name = "Date"
    return wide


def write_symbol_outputs(
    out_dir: str,
    symbol: str,
    providers: list[ProviderData],
    discrepancies: pd.DataFrame,
    consensus: pd.DataFrame | None,
    write_ratios: bool = False,
    compare_returns: bool = False,
) -> list[Path]:
    """Write normalized provider, wide, discrepancy, and optional consensus files."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = safe_symbol_filename(symbol)
    written: list[Path] = []

    for provider in providers:
        provider_name = safe_symbol_filename(provider.name)
        path = directory / f"{stem}_{provider_name}.csv"
        provider.frame.to_csv(path, index_label="Date")
        written.append(path)

    wide_path = directory / f"{stem}_all_providers.csv"
    all_providers_wide_frame(providers, write_ratios=write_ratios, compare_returns=compare_returns).to_csv(wide_path, index_label="Date")
    written.append(wide_path)

    if not discrepancies.empty:
        path = directory / f"{stem}_discrepancies.csv"
        discrepancies.to_csv(path, index=False)
        written.append(path)

    if consensus is not None:
        path = directory / f"{stem}_consensus.csv"
        consensus.to_csv(path, index_label="Date")
        written.append(path)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare daily OHLC price data from multiple providers.")
    p.add_argument("symbols", nargs="*", metavar="SYMBOL", help="ticker symbol(s), e.g. SPY or [SPY TIP]")
    p.add_argument("--symbols-file", metavar="PATH", help="read ticker symbols from a text file; blank lines and # comments are ignored")
    p.add_argument("--providers", nargs="*", default=["yahoo", "tiingo"], metavar="NAME",
                   help="network providers to query (default: yahoo tiingo)")
    p.add_argument("--provider-file", action="append", type=parse_provider_file, default=[], metavar="NAME=PATH",
                   help="add a local CSV as another provider; repeatable")
    p.add_argument("--start", type=parse_date, help="first date, YYYY-MM-DD")
    p.add_argument("--end", type=parse_date, help="last date, YYYY-MM-DD")
    p.add_argument("--abs-tol", type=float, default=0.01, help="absolute discrepancy tolerance (default: 0.01)")
    p.add_argument("--rel-tol", type=float, default=1e-4, help="relative discrepancy tolerance (default: 0.0001)")
    p.add_argument("--return-threshold", type=float, default=0.25,
                   help="flag absolute one-day adjusted-close returns above this value (default: 0.25)")
    p.add_argument("--top", type=int, default=20, help="maximum discrepancy rows shown per pair (default: 20; 0=all)")
    p.add_argument("--context-days", type=int, default=1, metavar="N",
                   help="common trading days before/after each displayed discrepancy (default: 1; 0=none)")
    p.add_argument("--yahoo-repair", action="store_true", help="ask yfinance to repair known Yahoo price errors")
    p.add_argument("--tiingo-token", help="Tiingo API token; otherwise TIINGO_API_TOKEN is used")
    p.add_argument("--consensus", choices=("median", "mean"), help="construct a consensus data set")
    p.add_argument("--consensus-min-providers", type=int, default=2, metavar="N",
                   help="minimum nonmissing providers required for a consensus value (default: 2)")
    p.add_argument("--write-consensus", metavar="PATH", help="write consensus CSV; requires --consensus")
    p.add_argument("--write-discrepancies", metavar="PATH", help="write all pairwise discrepancies to CSV")
    p.add_argument("--out-dir", metavar="DIR", help="write normalized provider data and a wide all-provider CSV for each symbol; also writes per-symbol discrepancies and requested consensus")
    p.add_argument("--write-ratios", action="store_true", help="append pairwise provider ratio columns to each _all_providers.csv written by --out-dir")
    p.add_argument("--compare-returns", action="store_true", help="compare adjusted close-to-close simple returns across providers")
    p.add_argument("--return-diff-tol", type=float, default=1e-4, metavar="RATE", help="absolute adjusted-return difference tolerance for --compare-returns (default: 0.0001 = 1 bp)")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error("--start must not be after --end")
    if args.abs_tol < 0 or args.rel_tol < 0:
        parser.error("tolerances must be nonnegative")
    if args.return_threshold <= 0:
        parser.error("--return-threshold must be positive")
    if args.consensus_min_providers < 1:
        parser.error("--consensus-min-providers must be positive")
    if args.context_days < 0:
        parser.error("--context-days must be nonnegative")
    if args.write_consensus and not args.consensus:
        parser.error("--write-consensus requires --consensus")
    if args.write_ratios and not args.out_dir:
        parser.error("--write-ratios requires --out-dir")
    if args.return_diff_tol < 0:
        parser.error("--return-diff-tol must be nonnegative")

    try:
        symbols = parse_symbols(args.symbols, args.symbols_file)
    except Exception as exc:
        parser.error(f"could not read symbols: {exc}")
    if not symbols:
        parser.error("provide at least one symbol or --symbols-file")

    all_symbol_discrepancies: list[pd.DataFrame] = []
    all_symbol_consensus: list[pd.DataFrame] = []
    summary_rows = []
    had_error = False

    for symbol_index, symbol in enumerate(symbols):
        providers: list[ProviderData] = []
        seen = set()
        try:
            for name in args.providers:
                if name.lower() in seen:
                    continue
                providers.append(load_provider(name, symbol, args))
                seen.add(name.lower())
            for name, path in args.provider_file:
                if name.lower() in seen:
                    raise ValueError(f"duplicate provider name {name!r}")
                frame = read_local_csv(path, symbol)
                if args.start is not None:
                    frame = frame.loc[frame.index >= args.start]
                if args.end is not None:
                    frame = frame.loc[frame.index <= args.end]
                providers.append(ProviderData(name=name, frame=frame))
                seen.add(name.lower())
        except Exception as exc:
            print(f"Error [{symbol}]: {exc}", file=sys.stderr)
            had_error = True
            continue

        if len(providers) < 2:
            print(f"Error [{symbol}]: at least two providers are required for comparison", file=sys.stderr)
            had_error = True
            continue

        if symbol_index:
            print("\n" + "=" * 80)
        print(f"Symbol: {symbol}")
        print("Providers:")
        for pvd in providers:
            if pvd.frame.empty:
                print(f"  {pvd.name}: no rows")
            else:
                print(f"  {pvd.name}: {len(pvd.frame):,} rows, {pvd.frame.index.min():%Y-%m-%d} to {pvd.frame.index.max():%Y-%m-%d}, fields: {', '.join(pvd.frame.columns)}")
                bad_ohlc = validate_ohlc(pvd.frame)
                bad_ret = suspicious_returns(pvd.frame, args.return_threshold)
                if not bad_ohlc.empty:
                    print(f"    OHLC integrity warnings: {len(bad_ohlc)}")
                if not bad_ret.empty:
                    print(f"    returns above {args.return_threshold:.1%}: {len(bad_ret)}")

        symbol_discrepancies = []
        missing_pair_count = 0
        max_rel_diff = float("nan")
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                a, b = providers[i], providers[j]
                discrepancies, stats = pairwise_compare(a, b, args.abs_tol, args.rel_tol)
                missing_pair_count += stats["missing_from_first"] + stats["missing_from_second"]
                if not discrepancies.empty:
                    this_max = float(discrepancies["RelDiff"].max())
                    max_rel_diff = this_max if not math.isfinite(max_rel_diff) else max(max_rel_diff, this_max)
                print(f"\n{a.name} vs {b.name}:")
                print(f"  Common dates: {stats['common_dates']:,}")
                print(f"  Compared fields: {', '.join(stats['compared_fields']) if stats['compared_fields'] else 'none'}")
                print_dates(f"dates missing from {a.name}", stats["missing_first_dates"], 10)
                print_dates(f"dates missing from {b.name}", stats["missing_second_dates"], 10)
                print(f"  Values outside both tolerances: {stats['discrepancies']:,}")
                if not discrepancies.empty:
                    show = discrepancies if args.top == 0 else discrepancies.head(max(args.top, 0))
                    printable = show.copy()
                    printable["Date"] = printable["Date"].dt.strftime("%Y-%m-%d")
                    printable["RelDiff"] = printable["RelDiff"].map(lambda x: f"{x:.6%}" if pd.notna(x) else "nan")
                    print(printable.to_string(index=False))

                    mismatched_fields = list(dict.fromkeys(discrepancies["Field"].tolist()))
                    for field in mismatched_fields:
                        print(f"\n  Diagnostics for mismatched field {field} over common dates:")
                        print(format_diagnostics_table(field_diagnostics(a, b, field)))

                    if args.context_days > 0:
                        for row in show.itertuples(index=False):
                            ctx = discrepancy_context(a, b, row.Field, row.Date, args.context_days)
                            if ctx.empty:
                                continue
                            shown_ctx = ctx.copy()
                            shown_ctx["Date"] = shown_ctx["Date"].dt.strftime("%Y-%m-%d")
                            print(f"\n  Context around {row.Date:%Y-%m-%d} {row.Field} mismatch:")
                            print(shown_ctx.to_string(index=False))

                    tagged = discrepancies.copy()
                    tagged.insert(0, "ProviderA", a.name)
                    tagged.insert(1, "ProviderB", b.name)
                    symbol_discrepancies.append(tagged)

                if args.compare_returns and "Close" in a.frame.columns and "Close" in b.frame.columns:
                    return_rows, return_stats = compare_adjusted_returns(a, b, args.return_diff_tol)
                    print("\n  Adjusted close-to-close return comparison:")
                    print(f"    Common return dates: {return_stats['common_return_dates']:,}")
                    corr = return_stats['correlation']
                    print(f"    Correlation: {'n/a' if not math.isfinite(corr) else f'{corr:.8f}'}")
                    print(f"    Mean absolute difference: {return_stats['mean_abs_diff']:.6%}")
                    print(f"    Median absolute difference: {return_stats['median_abs_diff']:.6%}")
                    print(f"    Maximum absolute difference: {return_stats['max_abs_diff']:.6%}")
                    print(f"    RMSE: {return_stats['rmse']:.6%}")
                    print(f"    Return mismatches above {args.return_diff_tol:.6%}: {return_stats['mismatches']:,}")
                    common_returns = adjusted_close_returns(a).dropna().index.intersection(adjusted_close_returns(b).dropna().index).sort_values()
                    print("\n    Return-series diagnostics over common return dates:")
                    print("    " + format_return_diagnostics([a, b], common_returns).replace("\n", "\n    "))
                    if not return_rows.empty:
                        show_ret = return_rows if args.top == 0 else return_rows.head(max(args.top, 0))
                        printable_ret = show_ret.copy()
                        printable_ret["Date"] = pd.to_datetime(printable_ret["Date"]).dt.strftime("%Y-%m-%d")
                        for col in printable_ret.columns:
                            if col.endswith("Return") or col in ("ReturnDiff", "AbsReturnDiff"):
                                printable_ret[col] = printable_ret[col].map(lambda x: f"{x:.6%}" if pd.notna(x) else "nan")
                        print("\n    Largest adjusted-return mismatches:")
                        print("    " + printable_ret.to_string(index=False).replace("\n", "\n    "))

        combined_symbol = pd.concat(symbol_discrepancies, ignore_index=True) if symbol_discrepancies else pd.DataFrame()
        if not combined_symbol.empty:
            tagged_symbol = combined_symbol.copy()
            tagged_symbol.insert(0, "Symbol", symbol)
            all_symbol_discrepancies.append(tagged_symbol)

        consensus = None
        if args.consensus:
            consensus = consensus_frame(providers, args.consensus, args.consensus_min_providers)
            print(f"\nConsensus ({args.consensus}, minimum {args.consensus_min_providers} providers): {len(consensus):,} rows")
            if len(symbols) == 1 and args.write_consensus:
                consensus.to_csv(args.write_consensus, index_label="Date")
                print(f"Wrote consensus to {args.write_consensus}")
            elif args.write_consensus:
                tagged = consensus.copy()
                tagged.insert(0, "Symbol", symbol)
                tagged.index.name = "Date"
                all_symbol_consensus.append(tagged.reset_index())

        if args.out_dir:
            written = write_symbol_outputs(args.out_dir, symbol, providers, combined_symbol, consensus, args.write_ratios, args.compare_returns)
            print(f"\nWrote {len(written)} file(s) for {symbol} to {args.out_dir}")

        summary_rows.append({
            "Symbol": symbol,
            "Providers": len(providers),
            "CommonDates": len(all_provider_common_dates(providers)),
            "MissingPairDates": missing_pair_count,
            "Discrepancies": len(combined_symbol),
            "MaxRelDiff": max_rel_diff,
        })

    if args.write_discrepancies:
        combined = pd.concat(all_symbol_discrepancies, ignore_index=True) if all_symbol_discrepancies else pd.DataFrame()
        if len(symbols) == 1 and not combined.empty:
            combined = combined.drop(columns=["Symbol"])
        combined.to_csv(args.write_discrepancies, index=False)
        print(f"\nWrote discrepancies to {args.write_discrepancies}")

    if args.write_consensus and len(symbols) > 1:
        combined = pd.concat(all_symbol_consensus, ignore_index=True) if all_symbol_consensus else pd.DataFrame()
        combined.to_csv(args.write_consensus, index=False)
        print(f"\nWrote consensus to {args.write_consensus}")

    if len(summary_rows) > 1:
        summary = pd.DataFrame(summary_rows)
        printable = summary.copy()
        printable["MaxRelDiff"] = printable["MaxRelDiff"].map(
            lambda x: "n/a" if not math.isfinite(x) else f"{x:.6%}"
        )
        print("\nCross-symbol summary:")
        print(printable.to_string(index=False))

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
