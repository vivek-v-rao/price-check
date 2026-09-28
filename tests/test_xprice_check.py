import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xprice_check import ProviderData, consensus_frame, normalize_frame, pairwise_compare, validate_ohlc


def frame(values):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame({"Open": values, "High": np.array(values)+1, "Low": np.array(values)-1, "Close": values}, index=idx)


def test_pairwise_tolerance():
    a = ProviderData("a", frame([100.0, 101.0, 102.0]))
    b = ProviderData("b", frame([100.0, 101.001, 103.0]))
    d, s = pairwise_compare(a, b, abs_tol=0.01, rel_tol=1e-4)
    assert s["common_dates"] == 3
    assert ((d["Field"] == "Close") & (d["Date"] == pd.Timestamp("2024-01-04"))).any()
    assert not ((d["Field"] == "Close") & (d["Date"] == pd.Timestamp("2024-01-03"))).any()


def test_consensus_median():
    a = ProviderData("a", frame([100.0, 101.0, 102.0]))
    b = ProviderData("b", frame([100.0, 101.0, 200.0]))
    c = ProviderData("c", frame([100.0, 101.0, 103.0]))
    out = consensus_frame([a, b, c], "median", 2)
    assert out.loc[pd.Timestamp("2024-01-04"), "Close"] == 103.0


def test_invalid_ohlc_detected():
    f = frame([100.0, 101.0, 102.0])
    f.loc[pd.Timestamp("2024-01-03"), "High"] = 99.0
    bad = validate_ohlc(f)
    assert pd.Timestamp("2024-01-03") in bad.index


def test_normalize_tiingo_columns():
    raw = pd.DataFrame({
        "date": ["2024-01-02T00:00:00.000Z"],
        "adjOpen": [10.0], "adjHigh": [11.0], "adjLow": [9.0], "adjClose": [10.5], "adjVolume": [1000],
    })
    out = normalize_frame(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.iloc[0]["Close"] == 10.5

from xprice_check import acf1, discrepancy_context, distribution_stats, field_diagnostics, transformed_changes


def test_transformed_changes_uses_log_diff_for_positive_values():
    s = pd.Series([100.0, 110.0, 121.0])
    chg, method = transformed_changes(s)
    assert method == "log diff"
    assert np.allclose(chg.to_numpy(), [np.log(1.1), np.log(1.1)])


def test_transformed_changes_falls_back_to_difference_for_nonpositive_values():
    s = pd.Series([1.0, 0.0, 2.0])
    chg, method = transformed_changes(s)
    assert method == "diff"
    assert np.allclose(chg.to_numpy(), [-1.0, 2.0])


def test_distribution_stats_and_acf1():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = distribution_stats(s)
    assert stats["Median"] == 3.0
    assert stats["Mean"] == 3.0
    assert np.isclose(stats["SD"], np.std([1, 2, 3, 4, 5], ddof=1))
    assert stats["Min"] == 1.0
    assert stats["Max"] == 5.0
    assert np.isclose(acf1(s), 1.0)


def test_field_diagnostics_show_reversal_in_change_acf():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    good = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105, 106, 107]}, index=idx)
    bad = good.copy()
    bad.loc[idx[3], "Close"] = 150.0
    a = ProviderData("good", good)
    b = ProviderData("bad", bad)
    diag = field_diagnostics(a, b, "Close")
    good_change = diag[(diag.Provider == "good") & (diag.Series == "Change")].iloc[0]
    bad_change = diag[(diag.Provider == "bad") & (diag.Series == "Change")].iloc[0]
    assert bad_change["ACF1"] < good_change["ACF1"]
    assert bad_change["ExKurt"] > good_change["ExKurt"]


def test_discrepancy_context_uses_common_trading_dates():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    a = ProviderData("a", pd.DataFrame({"Close": [10, 11, 12, 13]}, index=idx))
    b = ProviderData("b", pd.DataFrame({"Close": [10, 11, 99, 13]}, index=idx))
    ctx = discrepancy_context(a, b, "Close", pd.Timestamp("2024-01-05"), 1)
    assert list(ctx["Date"]) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")]

from xprice_check import parse_symbols


def test_parse_multiple_symbols_bracket_syntax():
    assert parse_symbols(["[SPY", "TIP]"]) == ["SPY", "TIP"]


def test_parse_multiple_symbols_plain_and_deduplicate():
    assert parse_symbols(["SPY", "TIP", "SPY"]) == ["SPY", "TIP"]


def test_parse_symbols_file(tmp_path):
    p = tmp_path / "symbols.txt"
    p.write_text("# test universe\nSPY\nTIP, IEF\n\nSPY\n", encoding="utf-8")
    assert parse_symbols([], str(p)) == ["SPY", "TIP", "IEF"]


def test_local_wide_file_supports_multiple_symbols(tmp_path):
    p = tmp_path / "wide.csv"
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "SPY": [100.0, 101.0],
        "TIP": [90.0, 90.5],
    }).to_csv(p, index=False)
    from xprice_check import read_local_csv
    spy = read_local_csv(str(p), "SPY")
    tip = read_local_csv(str(p), "TIP")
    assert np.allclose(spy["Close"], [100.0, 101.0])
    assert np.allclose(tip["Close"], [90.0, 90.5])

from xprice_check import all_providers_wide_frame, main, write_symbol_outputs


def test_all_providers_wide_frame_outer_joins_dates():
    a_idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    b_idx = pd.to_datetime(["2024-01-03", "2024-01-04"])
    a = ProviderData("yahoo", pd.DataFrame({"Close": [100.0, 101.0]}, index=a_idx))
    b = ProviderData("tiingo", pd.DataFrame({"Close": [101.1, 102.0]}, index=b_idx))
    wide = all_providers_wide_frame([a, b])
    assert list(wide.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert list(wide.columns) == ["yahoo_Close", "tiingo_Close"]
    assert pd.isna(wide.loc[pd.Timestamp("2024-01-02"), "tiingo_Close"])
    assert pd.isna(wide.loc[pd.Timestamp("2024-01-04"), "yahoo_Close"])


def test_write_symbol_outputs_writes_provider_and_wide_files(tmp_path):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    providers = [
        ProviderData("yahoo", pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)),
        ProviderData("tiingo", pd.DataFrame({"Close": [100.0, 101.2]}, index=idx)),
    ]
    discrepancies = pd.DataFrame({
        "ProviderA": ["yahoo"], "ProviderB": ["tiingo"],
        "Date": [pd.Timestamp("2024-01-03")], "Field": ["Close"],
        "yahoo": [101.0], "tiingo": [101.2], "AbsDiff": [0.2], "RelDiff": [0.2 / 101.2],
    })
    consensus = pd.DataFrame({"Close": [100.0, 101.1]}, index=idx)
    written = write_symbol_outputs(str(tmp_path), "^GSPC", providers, discrepancies, consensus)
    names = {path.name for path in written}
    assert names == {
        "_GSPC_yahoo.csv", "_GSPC_tiingo.csv", "_GSPC_all_providers.csv",
        "_GSPC_discrepancies.csv", "_GSPC_consensus.csv",
    }
    wide = pd.read_csv(tmp_path / "_GSPC_all_providers.csv")
    assert "yahoo_Close" in wide.columns
    assert "tiingo_Close" in wide.columns


def test_out_dir_cli_writes_files_without_extra_write_flag(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "out"
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "Open": [100, 101], "High": [101, 102], "Low": [99, 100], "Close": [100, 101], "Volume": [1000, 1100],
    }).to_csv(a, index=False)
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "Open": [100, 101], "High": [101, 102], "Low": [99, 100], "Close": [100, 101.2], "Volume": [1000, 1100],
    }).to_csv(b, index=False)
    rc = main([
        "SPY", "--providers",
        "--provider-file", f"a={a}",
        "--provider-file", f"b={b}",
        "--out-dir", str(out),
    ])
    assert rc == 0
    assert (out / "SPY_a.csv").exists()
    assert (out / "SPY_b.csv").exists()
    assert (out / "SPY_all_providers.csv").exists()
    assert (out / "SPY_discrepancies.csv").exists()


def test_all_providers_wide_frame_ratios():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    a = ProviderData("yahoo", pd.DataFrame({"Close": [100.0, 102.0, 104.0], "Volume": [1000.0, 0.0, 1200.0]}, index=idx))
    b = ProviderData("tiingo", pd.DataFrame({"Close": [50.0, 51.0, np.nan], "Volume": [500.0, 0.0, 600.0]}, index=idx))
    wide = all_providers_wide_frame([a, b], write_ratios=True)
    assert np.isclose(wide.loc[idx[0], "yahoo_Close__over__tiingo_Close"], 2.0)
    assert np.isclose(wide.loc[idx[0], "yahoo_Volume__over__tiingo_Volume"], 2.0)
    assert pd.isna(wide.loc[idx[1], "yahoo_Volume__over__tiingo_Volume"])
    assert pd.isna(wide.loc[idx[2], "yahoo_Close__over__tiingo_Close"])


def test_write_ratios_cli_appends_columns(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "out"
    base = {
        "Date": ["2024-01-02", "2024-01-03"],
        "Open": [100, 101], "High": [101, 102], "Low": [99, 100],
        "Close": [100, 101], "Volume": [1000, 1100],
    }
    pd.DataFrame(base).to_csv(a, index=False)
    other = dict(base)
    other["Close"] = [50, 50.5]
    pd.DataFrame(other).to_csv(b, index=False)
    rc = main([
        "SPY", "--providers",
        "--provider-file", f"a={a}",
        "--provider-file", f"b={b}",
        "--out-dir", str(out),
        "--write-ratios",
    ])
    assert rc == 0
    wide = pd.read_csv(out / "SPY_all_providers.csv")
    assert "a_Close__over__b_Close" in wide.columns
    assert np.allclose(wide["a_Close__over__b_Close"], [2.0, 2.0])


def test_write_ratios_requires_out_dir():
    try:
        main(["SPY", "--providers", "--write-ratios"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse error")

from xprice_check import adjusted_close_returns, compare_adjusted_returns


def test_adjusted_close_returns_simple_returns():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    p = ProviderData("a", pd.DataFrame({"Close": [100.0, 101.0, 99.99]}, index=idx))
    r = adjusted_close_returns(p)
    assert pd.isna(r.iloc[0])
    assert np.isclose(r.iloc[1], 0.01)
    assert np.isclose(r.iloc[2], 99.99 / 101.0 - 1.0)


def test_compare_adjusted_returns_detects_mismatch():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    a = ProviderData("a", pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=idx))
    b = ProviderData("b", pd.DataFrame({"Close": [100.0, 101.0, 105.0, 106.0]}, index=idx))
    rows, stats = compare_adjusted_returns(a, b, diff_tol=1e-4)
    assert stats["common_return_dates"] == 3
    assert stats["mismatches"] >= 1
    assert stats["max_abs_diff"] > 0
    assert "aReturn" in rows.columns and "bReturn" in rows.columns


def test_wide_frame_compare_returns_appends_return_columns():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    a = ProviderData("yahoo", pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx))
    b = ProviderData("tiingo", pd.DataFrame({"Close": [100.0, 101.5, 102.0]}, index=idx))
    wide = all_providers_wide_frame([a, b], compare_returns=True)
    assert "yahoo_CloseReturn" in wide.columns
    assert "tiingo_CloseReturn" in wide.columns
    diff_col = "yahoo_CloseReturn__minus__tiingo_CloseReturn"
    assert diff_col in wide.columns
    assert np.isclose(wide.loc[idx[1], diff_col], (101/100-1) - (101.5/100-1))


def test_compare_returns_cli_writes_columns(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "out"
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "Open": [100, 101, 102], "High": [101, 102, 103], "Low": [99, 100, 101],
        "Close": [100, 101, 102], "Volume": [1000, 1100, 1200],
    }).to_csv(a, index=False)
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "Open": [100, 101, 102], "High": [101, 102, 103], "Low": [99, 100, 101],
        "Close": [100, 101.2, 102.1], "Volume": [1000, 1100, 1200],
    }).to_csv(b, index=False)
    rc = main([
        "SPY", "--providers",
        "--provider-file", f"a={a}",
        "--provider-file", f"b={b}",
        "--out-dir", str(out),
        "--compare-returns",
        "--return-diff-tol", "0.00001",
    ])
    assert rc == 0
    wide = pd.read_csv(out / "SPY_all_providers.csv")
    assert "a_CloseReturn" in wide.columns
    assert "b_CloseReturn" in wide.columns
    assert "a_CloseReturn__minus__b_CloseReturn" in wide.columns
