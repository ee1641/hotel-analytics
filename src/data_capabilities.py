"""
Per-page column requirements for uploaded datasets.

Call check_page_capabilities(available_columns) after upload to get
user-facing warnings for any page that is missing required columns.
"""

# Minimum columns required per dashboard page.
# Missing any of these means the page may produce misleading results.
_PAGE_REQUIREMENTS: dict[str, list[str]] = {
    "overview":     ["arrival_date", "adr"],
    "forecasting":  ["arrival_date", "adr"],
    "staffing":     ["arrival_date"],
    "cancellation": [
        "arrival_date", "adr", "is_canceled", "lead_time",
        "adults", "stays_in_week_nights", "stays_in_weekend_nights",
    ],
    "overbooking":  ["arrival_date", "is_canceled", "adr"],
    "profit":       ["arrival_date", "adr"],
}

_PAGE_LABELS: dict[str, str] = {
    "overview":     "Overview Dashboard",
    "forecasting":  "Revenue & Demand Forecast",
    "staffing":     "Demand & Staffing",
    "cancellation": "Cancellation Predictor",
    "overbooking":  "Overbooking Simulator",
    "profit":       "Profit Simulator",
}


def check_page_capabilities(available_columns: set[str]) -> dict[str, list[str]]:
    """
    Return a dict of {page_key: [warning_message, ...]} for pages missing
    required columns. An empty list means the page is fully supported.

    Parameters
    ----------
    available_columns : set of column names present in the uploaded dataset

    Returns
    -------
    dict where keys are page names and values are lists of warning strings.
    """
    warnings: dict[str, list[str]] = {}
    for page, required in _PAGE_REQUIREMENTS.items():
        missing = [c for c in required if c not in available_columns]
        if missing:
            warnings[page] = [
                f"Missing column(s) required for **{_PAGE_LABELS[page]}**: "
                f"`{'`, `'.join(missing)}`. Results on this page may be inaccurate."
            ]
        else:
            warnings[page] = []
    return warnings
