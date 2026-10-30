import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    load_data, clean_data, engineer_features, encode_categoricals,
    drop_leaky_features, LEAKY_COLUMNS, remove_accidental_duplicate_bookings,
    expanding_target_encode, target_encode_column,
)

DATA_PATH = "hotel_bookings.csv"

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)

@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)

@pytest.fixture(scope="module")
def engineered_df(clean_df):
    return engineer_features(clean_df)


def test_load_shape(raw_df):
    assert raw_df.shape[0] > 100_000
    assert raw_df.shape[1] == 32


def test_no_negative_adr(clean_df):
    assert (clean_df["adr"] >= 0).all()


def test_no_null_country(clean_df):
    assert clean_df["country"].isna().sum() == 0


def test_no_null_agent(clean_df):
    assert clean_df["agent"].isna().sum() == 0


def test_no_null_children(clean_df):
    assert clean_df["children"].isna().sum() == 0


def test_exact_duplicate_rows_removed():
    df = pd.DataFrame(
        [
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0},
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0},
            {"arrival_date": "2024-01-02", "is_canceled": 1, "adr": 120.0},
        ]
    )

    deduped = remove_accidental_duplicate_bookings(df)

    assert len(deduped) == 2


def test_clean_data_removes_exact_duplicates():
    df = pd.DataFrame(
        [
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0, "agent": None, "company": None, "country": None, "children": None},
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0, "agent": None, "company": None, "country": None, "children": None},
            {"arrival_date": "2024-01-02", "is_canceled": 1, "adr": 120.0, "agent": None, "company": None, "country": None, "children": None},
        ]
    )

    cleaned = clean_data(df)

    assert len(cleaned) == 2


def test_total_nights(engineered_df):
    expected = engineered_df["stays_in_weekend_nights"] + engineered_df["stays_in_week_nights"]
    pd.testing.assert_series_equal(engineered_df["total_nights"], expected, check_names=False)


def test_total_guests(engineered_df):
    expected = engineered_df["adults"] + engineered_df["children"] + engineered_df["babies"]
    pd.testing.assert_series_equal(engineered_df["total_guests"], expected, check_names=False)


def test_revenue_non_negative(engineered_df):
    assert (engineered_df["revenue"] >= 0).all()


def test_is_no_show(engineered_df):
    expected = (engineered_df["reservation_status"] == "No-Show").astype(int)
    pd.testing.assert_series_equal(engineered_df["is_no_show"], expected, check_names=False)


def test_encode_drops_leaky_columns(engineered_df):
    X, _, _ = encode_categoricals(engineered_df)
    assert "reservation_status" not in X.columns
    assert "reservation_status_date" not in X.columns


def test_encode_no_object_columns(engineered_df):
    X, _, _ = encode_categoricals(engineered_df)
    assert X.select_dtypes(include="object").empty


def test_encode_retains_lead_time(engineered_df):
    X, _, _ = encode_categoricals(engineered_df)
    assert "lead_time" in X.columns


def test_drop_leaky_removes_all_leaky_columns(engineered_df):
    clean = drop_leaky_features(engineered_df)
    for col in LEAKY_COLUMNS:
        assert col not in clean.columns, f"Leaky column '{col}' still present after drop"


def test_drop_leaky_preserves_non_leaky_columns(engineered_df):
    clean = drop_leaky_features(engineered_df)
    assert "lead_time" in clean.columns
    assert "adr" in clean.columns
    assert "total_nights" in clean.columns


def test_drop_leaky_then_encode_has_no_leaky_cols(engineered_df):
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    for col in LEAKY_COLUMNS:
        assert col not in X.columns, f"Leaky column '{col}' leaked into feature matrix"


def test_engineer_features_lead_time_bucket():
    """lead_time_bucket: 0=short(<7 days), 1=medium(7-30), 2=long(>30)."""
    df = pd.DataFrame({
        "stays_in_weekend_nights": [1, 1, 1],
        "stays_in_week_nights":    [2, 2, 2],
        "adults": [2, 2, 2],
        "children": [0, 0, 0],
        "babies": [0, 0, 0],
        "adr": [100.0, 100.0, 100.0],
        "lead_time": [3, 15, 60],
        "is_canceled": [0, 0, 0],
    })
    out = engineer_features(df)
    assert "lead_time_bucket" in out.columns
    assert list(out["lead_time_bucket"]) == [0, 1, 2]


def test_engineer_features_adr_vs_hotel_mean():
    """adr_vs_hotel_mean: ratio clipped to [0, 10], finite for all rows."""
    df = pd.DataFrame({
        "stays_in_weekend_nights": [1, 1],
        "stays_in_week_nights":    [2, 2],
        "adults": [2, 2],
        "children": [0, 0],
        "babies": [0, 0],
        "adr": [50.0, 150.0],
        "lead_time": [10, 10],
        "is_canceled": [0, 0],
    })
    out = engineer_features(df)
    assert "adr_vs_hotel_mean" in out.columns
    assert (out["adr_vs_hotel_mean"] >= 0).all()
    assert (out["adr_vs_hotel_mean"] <= 10).all()


def test_add_temporal_features_booking_dow():
    """booking_dow: integer in 0-6 derived from arrival_date minus lead_time days."""
    df = pd.DataFrame({
        "arrival_date": pd.to_datetime(["2024-06-10", "2024-06-15"]),
        "lead_time": [10, 5],
    })
    from src.preprocessing import add_temporal_features
    out = add_temporal_features(df)
    assert "booking_dow" in out.columns
    assert out["booking_dow"].between(0, 6).all()


def test_lead_time_bucket_boundary():
    """Boundary values: 6 → bucket 0, 7 → bucket 1, 30 → bucket 1, 31 → bucket 2."""
    df = pd.DataFrame({
        "stays_in_weekend_nights": [0, 0, 0, 0],
        "stays_in_week_nights":    [1, 1, 1, 1],
        "adults": [2, 2, 2, 2],
        "children": [0, 0, 0, 0],
        "babies": [0, 0, 0, 0],
        "adr": [100.0] * 4,
        "lead_time": [6, 7, 30, 31],
        "is_canceled": [0, 0, 0, 0],
    })
    out = engineer_features(df)
    assert list(out["lead_time_bucket"]) == [0, 1, 1, 2]


def test_time_split_no_future_data_in_train():
    """Time-based split: max(train arrival_date) must be < min(test arrival_date)."""
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    df = pd.DataFrame({"arrival_date": dates})
    df_sorted = df.sort_values("arrival_date")
    n_test = max(1, int(len(df_sorted) * 0.2))
    train = df_sorted.iloc[:-n_test]
    test = df_sorted.iloc[-n_test:]
    assert train["arrival_date"].max() < test["arrival_date"].min()


# ---------------------------------------------------------------------------
# Expanding-window target encoding (Track 10, Part 4)
# ---------------------------------------------------------------------------

def _te_frame():
    # category "A" appears 3 times with labels 1, 0, 1 (in arrival_date order);
    # category "B" once. global mean = 0.6.
    return pd.DataFrame({
        "arrival_date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
        ]),
        "agent": ["A", "A", "B", "A", "B"],
        "is_canceled": [1, 0, 1, 1, 0],
    })


def test_expanding_first_occurrence_is_global_mean():
    df = _te_frame()
    gmean = float(df["is_canceled"].mean())
    enc = expanding_target_encode(df, "agent", global_mean=gmean, smoothing=20.0)
    # First row of category A and first row of category B have no prior → global mean.
    assert enc.iloc[0] == pytest.approx(gmean)  # A, first
    assert enc.iloc[2] == pytest.approx(gmean)  # B, first


def test_expanding_excludes_own_label():
    """Row 4 (3rd 'A') must encode from rows 1-2's labels only (1, 0), never its own."""
    df = _te_frame()
    gmean, k = 0.6, 20.0
    enc = expanding_target_encode(df, "agent", global_mean=gmean, smoothing=k)
    # 3rd A: prior labels [1, 0] → sum=1, count=2 → (1 + k*gmean)/(2+k)
    expected = (1 + k * gmean) / (2 + k)
    assert enc.iloc[3] == pytest.approx(expected)
    # 2nd A (row index 1): prior labels [1] → (1 + k*gmean)/(1+k)
    expected_2nd = (1 + k * gmean) / (1 + k)
    assert enc.iloc[1] == pytest.approx(expected_2nd)


def test_expanding_prior_does_not_cross_categories():
    """2nd 'B' (row 4) follows an 'A' row; its prior must be B's labels only ([1])."""
    df = _te_frame()
    gmean, k = 0.6, 20.0
    enc = expanding_target_encode(df, "agent", global_mean=gmean, smoothing=k)
    assert enc.iloc[4] == pytest.approx((1 + k * gmean) / (1 + k))
    assert enc.between(0.0, 1.0).all()


def test_expanding_preserves_index_order():
    df = _te_frame()
    enc = expanding_target_encode(df, "agent", global_mean=0.6)
    assert list(enc.index) == list(df.index)
    assert len(enc) == len(df)
    assert enc.notna().all()


def test_expanding_handles_unsorted_input():
    """Encoding must order by arrival_date internally, then restore row order."""
    df = _te_frame().sample(frac=1.0, random_state=1)  # shuffle rows
    enc = expanding_target_encode(df, "agent", global_mean=0.6, smoothing=20.0)
    # The row that is chronologically first for its category still gets global mean.
    first_a_label_row = df.index[df["arrival_date"] == df["arrival_date"].min()][0]
    assert enc.loc[first_a_label_row] == pytest.approx(0.6)


def test_static_map_unchanged_for_inference():
    """The static final map (inference contract) is unaffected by expanding encoding."""
    df = _te_frame()
    rate_map = df.groupby("agent")["is_canceled"].mean().to_dict()
    assert rate_map["A"] == pytest.approx(2 / 3)  # labels 1,0,1
    assert rate_map["B"] == pytest.approx(0.5)    # labels 1,0
