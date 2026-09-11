import pandas as pd

from alpha_lab.data.quality import check_bars, clean_mask
from alpha_lab.data.schema import normalize_bars


def _bars(ts):
    return normalize_bars(pd.DataFrame({
        "ts": pd.to_datetime(ts, utc=True),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 10.0, "quote_volume": 1000.0, "trades": 5,
        "taker_buy_volume": 6.0,
    }))


def test_clean_series_has_no_issues():
    df = _bars(pd.date_range("2024-01-01", periods=100, freq="1min"))

    rep = check_bars(df, "1m")

    assert rep.is_clean
    assert rep.total_rows == 100
    assert rep.gaps == 0


def test_detects_gap():
    ts = list(pd.date_range("2024-01-01", periods=50, freq="1min"))
    ts += list(pd.date_range("2024-01-01 02:00", periods=50, freq="1min"))

    rep = check_bars(_bars(ts), "1m")

    assert not rep.is_clean
    assert rep.gaps == 1


def test_detects_zero_volume():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[3, "volume"] = 0.0

    rep = check_bars(df, "1m")

    assert rep.zero_volume == 1


def test_detects_impossible_ohlc():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[2, "high"] = 50.0   # high ниже low

    rep = check_bars(df, "1m")

    assert rep.anomalous == 1


def test_duplicates_removed_by_normalize():
    df = _bars(pd.date_range("2024-01-01", periods=5, freq="1min"))
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

    rep = check_bars(normalize_bars(df), "1m")

    assert rep.duplicates == 0
    assert rep.total_rows == 5


def test_summary_mentions_problems():
    ts = list(pd.date_range("2024-01-01", periods=10, freq="1min"))
    ts += list(pd.date_range("2024-01-01 05:00", periods=10, freq="1min"))

    text = check_bars(_bars(ts), "1m").summary()

    assert "пропуск" in text.lower()


def test_clean_mask_keeps_clean_bars():
    df = _bars(pd.date_range("2024-01-01", periods=5, freq="1min"))

    mask = clean_mask(df)

    assert mask.all()
    assert mask.dtype == bool


def test_clean_mask_excludes_zero_volume():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[3, "volume"] = 0.0

    mask = clean_mask(df)

    assert not mask.loc[3]
    assert mask.sum() == 9


def test_clean_mask_excludes_high_below_low():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[2, "high"] = 50.0

    assert not clean_mask(df).loc[2]


def test_clean_mask_excludes_high_below_open_and_close():
    # Точный случай из ревью: high ниже open/close, но выше low —
    # репортёр считает такой бар аномальным, значит и фильтр обязан его исключить.
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[4, ["open", "high", "low", "close"]] = [100.0, 50.0, 49.0, 100.0]

    rep = check_bars(df, "1m")

    assert not clean_mask(df).loc[4]
    assert rep.anomalous == 1
    assert not rep.is_clean


def test_clean_mask_excludes_nan_high_and_report_flags_it():
    # Регрессия: сравнения с NaN давали False, и отчёт мог назвать бар чистым,
    # пока clean_mask его молча исключала.
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[6, "high"] = float("nan")

    rep = check_bars(df, "1m")

    assert not clean_mask(df).loc[6]
    assert not rep.is_clean


def test_clean_mask_excludes_non_positive_prices():
    for column in ("open", "high", "low", "close"):
        df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
        df.loc[4, column] = 0.0

        assert not clean_mask(df).loc[4], column


def test_clean_mask_excludes_unparseable_volume():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df["volume"] = df["volume"].astype(object)
    df.loc[5, "volume"] = "n/a"

    rep = check_bars(df, "1m")

    assert not clean_mask(df).loc[5]
    assert rep.zero_volume == 1


def _dirty_fixtures():
    ts = pd.date_range("2024-01-01", periods=10, freq="1min")
    frames = {"clean": _bars(ts)}

    df = _bars(ts)
    df.loc[1, "volume"] = 0.0
    frames["zero_volume"] = df

    df = _bars(ts)
    df.loc[2, "high"] = 50.0
    frames["high_below_low"] = df

    df = _bars(ts)
    df.loc[3, ["open", "high", "low", "close"]] = [100.0, 50.0, 49.0, 100.0]
    frames["high_below_open_close"] = df

    df = _bars(ts)
    df.loc[4, "high"] = float("nan")
    frames["nan_high"] = df

    df = _bars(ts)
    df.loc[5, "low"] = 0.0
    frames["non_positive_low"] = df

    df = _bars(ts)
    df.loc[6, "volume"] = float("nan")
    frames["nan_volume"] = df

    return frames


def test_clean_mask_and_report_agree_on_dirty_frames():
    for name, df in _dirty_fixtures().items():
        rep = check_bars(df, "1m")
        assert rep.is_clean == bool(clean_mask(df).all()), name
        assert rep.bad_rows == int((~clean_mask(df)).sum()), name
