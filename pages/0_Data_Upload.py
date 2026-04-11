"""
Page 0 — Data Upload
Upload your own hotel CSV and map its columns to the canonical schema.
All other pages will use this data if provided, falling back to the built-in demo.
"""
import warnings
warnings.filterwarnings("ignore")

import io
import pandas as pd
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingestion import auto_detect_columns, map_columns, validate_schema, CANONICAL_SCHEMA
from src.data_capabilities import check_page_capabilities

st.set_page_config(page_title="Data Upload", page_icon="📂", layout="wide")
st.title("📂 Data Upload & Schema Mapping")
st.markdown(
    "Upload your hotel bookings CSV and map its columns to the standard schema. "
    "All other dashboard pages will automatically use your data once mapped."
)

# ---------------------------------------------------------------------------
# Demo dataset shortcut
# ---------------------------------------------------------------------------
if st.button("Use Built-in Kaggle Demo Dataset", type="secondary"):
    st.session_state.pop("user_df", None)
    st.session_state.pop("user_daily_df", None)
    st.success("Demo dataset selected. All pages will use the built-in Kaggle dataset.")

st.divider()

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload hotel bookings CSV", type=["csv"])

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(raw_df):,} rows** × **{len(raw_df.columns)} columns**")
        with st.expander("Preview raw data"):
            st.dataframe(raw_df.head(5), use_container_width=True)

        # Auto-detect column mappings
        suggested = auto_detect_columns(raw_df)
        canonical_options = ["(skip)"] + list(CANONICAL_SCHEMA["columns"].keys())

        st.subheader("Column Mapping")
        st.markdown(
            "Review the auto-detected mappings below. Adjust any incorrect suggestions, "
            "then click **Validate & Apply**."
        )

        mapping = {}
        cols_left, cols_right = st.columns(2)
        user_cols = list(raw_df.columns)
        half = len(user_cols) // 2 + len(user_cols) % 2

        for i, user_col in enumerate(user_cols):
            suggestion = suggested.get(user_col, "(skip)")
            target = cols_left if i < half else cols_right
            with target:
                chosen = st.selectbox(
                    f"`{user_col}`",
                    options=canonical_options,
                    index=canonical_options.index(suggestion) if suggestion in canonical_options else 0,
                    key=f"col_map_{i}",
                )
                if chosen != "(skip)":
                    mapping[user_col] = chosen

        if st.button("Validate & Apply Mapping", type="primary"):
            mapped_df = map_columns(raw_df, mapping)
            is_valid, issues = validate_schema(mapped_df)

            if is_valid:
                st.success("Schema validation passed. Your data is ready to use.")
                st.session_state["user_df"] = mapped_df
                st.session_state.pop("user_daily_df", None)  # force re-aggregation

                # Show per-page capability warnings based on which columns are present
                available_cols = set(mapped_df.columns)
                capability_warnings = check_page_capabilities(available_cols)
                all_good = all(len(w) == 0 for w in capability_warnings.values())
                if all_good:
                    st.success("All dashboard pages are fully supported by your dataset.")
                else:
                    st.subheader("Page Compatibility")
                    st.caption(
                        "Some pages need columns not present in your file. "
                        "Those pages will still load but may show inaccurate or empty results."
                    )
                    for page, msgs in capability_warnings.items():
                        for msg in msgs:
                            st.warning(msg)
            else:
                st.error("Schema validation failed. Please fix the issues below:")
                for issue in issues:
                    st.markdown(f"- {issue}")

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# ---------------------------------------------------------------------------
# Status indicator
# ---------------------------------------------------------------------------
st.divider()
if "user_df" in st.session_state:
    df = st.session_state["user_df"]
    st.info(
        f"Active dataset: **custom upload** — {len(df):,} rows. "
        "All pages are using this data."
    )
else:
    st.info("Active dataset: **Kaggle demo** (hotel_bookings.csv). Upload a file above to switch.")
