"""Results viewer for the local pipeline outputs (run `uv run dfc run-all` first).

    uv run streamlit run app/streamlit_app.py

Reads the gold Delta tables directly with the `deltalake` package (no Spark needed), so it
starts in a second. Set DF_DATA_DIR to point at another data folder.
"""

from __future__ import annotations

import os
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from deltalake import DeltaTable

DATA_DIR = Path(os.environ.get("DF_DATA_DIR", "data"))
LAKE = DATA_DIR / "lake"

# Reference data-viz palette (validated categorical slots 1-2) plus a neutral grey for actuals.
MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#eb6834"
ACTUAL_COLOR = "#8a8985"  # mid grey: >= 3:1 contrast on both the light and dark theme
MODEL_LABELS = {
    "lightgbm": "LightGBM model",
    "moving_average_28": "28-day moving average",
    "seasonal_naive": "Same weekday last week",
}

st.set_page_config(page_title="Demand forecasting", layout="wide")


def table_path(layer: str, name: str) -> Path:
    return LAKE / layer / name


@st.cache_data(show_spinner=False)
def load(layer: str, name: str, filters: tuple[tuple[str, str, str], ...] = ()) -> pd.DataFrame:
    dt = DeltaTable(str(table_path(layer, name)))
    return dt.to_pandas(filters=list(filters) or None)


def require_outputs() -> None:
    missing = [
        n for n in ("backtest_metrics", "forecasts", "replenishment_plan")
        if not (table_path("gold", n) / "_delta_log").is_dir()
    ]  # fmt: skip
    if missing:
        st.error(
            f"No pipeline outputs in `{DATA_DIR.resolve()}` (missing: {', '.join(missing)}). "
            "Run `uv run dfc run-all --preset small` from the repository root first, "
            "or set DF_DATA_DIR."
        )
        st.stop()


def rows_by(df: pd.DataFrame, key: str) -> dict[str, dict[Hashable, Any]]:
    """Rows as plain dicts keyed by one column (e.g. metrics per model)."""
    return {str(r[key]): r for r in df.to_dict("records")}


def nice(name: object) -> str:
    """'health_personal_care' -> 'Health personal care'."""
    return str(name).replace("_", " ").capitalize()


def pct(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.1%}"


def summary_tab() -> None:
    metrics = load("gold", "backtest_metrics")
    policy = load("gold", "policy_evaluation")
    overall = rows_by(metrics[metrics["level"] == "overall"], "model")
    last_df = policy[policy["level"] == "last_origin"]
    last = rows_by(last_df, "policy")

    st.subheader("Backtest: would the model have helped?")
    st.caption(
        "Three past cut-off dates; the model only saw data up to each one. 'True demand' "
        "includes what customers wanted on days the shelf was empty, which only a simulation knows."
    )
    c = st.columns(4)
    lgbm, ma = overall["lightgbm"], overall["moving_average_28"]
    c[0].metric(
        "Error vs true demand (WAPE)",
        pct(lgbm["wape_vs_true_demand"]),
        f"{(lgbm['wape_vs_true_demand'] - ma['wape_vs_true_demand']) * 100:+.1f} pts vs moving avg",
        delta_color="inverse",
    )
    c[1].metric("Upper forecast coverage (target 90%)", pct(lgbm["coverage_upper"]))
    cal = last["order_up_to:lightgbm_calibrated"]
    c[2].metric(
        "Replenishment cycles without stockout",
        pct(cal["cycle_service_level"]),
        f"target {cal['target_service_level']:.0%}",
        delta_color="off",
    )
    ma_policy = last["order_up_to:moving_average_28"]
    c[3].metric(
        "Demand served (fill rate)",
        pct(cal["fill_rate"]),
        f"{(cal['fill_rate'] - ma_policy['fill_rate']) * 100:+.1f} pts vs moving avg policy",
    )

    by_aisle = metrics[
        (metrics["level"] == "aisle") & metrics["model"].isin(["lightgbm", "moving_average_28"])
    ]
    by_aisle = by_aisle.assign(
        model_label=by_aisle["model"].map(MODEL_LABELS), aisle=by_aisle["level_value"].map(nice)
    )
    colors = alt.Scale(
        domain=[MODEL_LABELS["lightgbm"], MODEL_LABELS["moving_average_28"]],
        range=[MODEL_COLOR, BASELINE_COLOR],
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Error vs true demand by aisle** (lower is better)")
        chart = (
            alt.Chart(by_aisle)
            .mark_bar(cornerRadiusEnd=4, size=10)
            .encode(
                y=alt.Y("aisle:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200)),
                x=alt.X(
                    "wape_vs_true_demand:Q", title="WAPE vs true demand", axis=alt.Axis(format="%")
                ),
                yOffset="model_label:N",
                color=alt.Color(
                    "model_label:N", scale=colors, title=None, legend=alt.Legend(orient="top")
                ),
                tooltip=[
                    alt.Tooltip("aisle:N", title="Aisle"),
                    alt.Tooltip("model_label:N", title="Forecast"),
                    alt.Tooltip("wape_vs_true_demand:Q", title="WAPE", format=".1%"),
                    alt.Tooltip("bias:Q", title="Bias (clean days)", format="+.1%"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("**Error by days ahead**")
        by_h = metrics[
            (metrics["level"] == "horizon")
            & metrics["model"].isin(["lightgbm", "moving_average_28"])
        ]
        by_h = by_h.assign(
            horizon=by_h["level_value"].astype(int), model_label=by_h["model"].map(MODEL_LABELS)
        )
        line = (
            alt.Chart(by_h)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True))
            .encode(
                x=alt.X("horizon:Q", title="Days ahead", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y(
                    "wape_vs_true_demand:Q", title="WAPE vs true demand", axis=alt.Axis(format="%")
                ),
                color=alt.Color(
                    "model_label:N", scale=colors, title=None, legend=alt.Legend(orient="top")
                ),
                tooltip=[
                    alt.Tooltip("horizon:Q", title="Days ahead"),
                    alt.Tooltip("model_label:N", title="Forecast"),
                    alt.Tooltip("wape_vs_true_demand:Q", title="WAPE", format=".1%"),
                ],
            )
        )
        st.altair_chart(line, use_container_width=True)

    st.markdown("**Replenishment what-if (last cut-off, held out from calibration)**")
    view = last_df[
        [
            "policy",
            "cycles",
            "cycle_service_level",
            "fill_rate",
            "avg_excess_units",
            "avg_short_units",
        ]
    ]
    st.dataframe(
        view.style.format(
            {
                "cycle_service_level": "{:.1%}",
                "fill_rate": "{:.1%}",
                "avg_excess_units": "{:.1f}",
                "avg_short_units": "{:.1f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("All accuracy metrics (table view)"):
        st.dataframe(metrics.drop(columns=["run_id"]), hide_index=True, use_container_width=True)


def explorer_tab() -> None:
    plan = load("gold", "replenishment_plan")
    stores = load("bronze", "stores")
    cols = st.columns(4)
    country = cols[0].selectbox("Country", sorted(stores["country_code"].unique()))
    store_ids = sorted(stores[stores["country_code"] == country]["store_id"])
    store_id = cols[1].selectbox("Store", store_ids)
    in_store = plan[plan["store_id"] == store_id]
    aisle = cols[2].selectbox("Aisle", sorted(in_store["aisle"].unique()), format_func=nice)
    skus = in_store[in_store["aisle"] == aisle].sort_values("sku_name")
    sku_name = cols[3].selectbox("Product", skus["sku_name"].tolist())
    if store_id is None or sku_name is None:
        st.info("This store has nothing on the shelf in the forecast period.")
        return
    row = rows_by(skus, "sku_name")[sku_name]
    sku_id = str(row["sku_id"])

    key = (("store_id", "=", str(store_id)), ("sku_id", "=", sku_id))
    panel = load("silver", "daily_panel", key)[["date", "units_sold", "is_stockout"]]
    panel = panel.sort_values("date").tail(120)
    backtest = load("gold", "backtest_predictions", key)
    backtest = backtest[backtest["model"] == "lightgbm"][
        ["date", "forecast_mean", "forecast_upper"]
    ]
    future = load("gold", "forecasts", key)[["date", "forecast_mean", "forecast_upper"]]
    fc = pd.concat(
        [backtest.assign(kind="Backtest forecast"), future.assign(kind="Forward forecast")]
    )

    band = (
        alt.Chart(fc)
        .mark_area(opacity=0.18, color=MODEL_COLOR)
        .encode(
            x=alt.X("date:T", title=None),
            y="forecast_mean:Q",
            y2="forecast_upper:Q",
            detail="kind:N",
        )
    )
    model_line = (
        alt.Chart(fc)
        .mark_line(strokeWidth=2, color=MODEL_COLOR)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("forecast_mean:Q", title="Units per day"),
            strokeDash=alt.StrokeDash("kind:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("kind:N", title="Series"),
                alt.Tooltip("forecast_mean:Q", title="Forecast", format=".1f"),
                alt.Tooltip("forecast_upper:Q", title="Upper (P90)", format=".1f"),
            ],
        )
    )
    actual = (
        alt.Chart(panel)
        .mark_line(strokeWidth=2, color=ACTUAL_COLOR, interpolate="step-after")
        .encode(
            x=alt.X("date:T", title=None),
            y="units_sold:Q",
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("units_sold:Q", title="Units sold"),
                alt.Tooltip("is_stockout:N", title="Sold out"),
            ],
        )
    )
    stockouts = (
        alt.Chart(panel[panel["is_stockout"]])
        .mark_point(shape="triangle-down", size=70, filled=True, color=ACTUAL_COLOR)
        .encode(
            x=alt.X("date:T", title=None),
            y="units_sold:Q",
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("units_sold:Q", title="Units sold (sold out)"),
            ],
        )
    )
    st.altair_chart(
        alt.layer(band, model_line, actual, stockouts).properties(height=360),
        use_container_width=True,
    )
    st.caption(
        "Grey line: actual units sold; triangles mark days the shelf was empty at closing (sales "
        "understate demand). Blue: model forecast, shaded up to the P90 upper forecast."
    )
    c = st.columns(4)
    c[0].metric("Recommended order", f"{int(row['recommended_order_qty'])} units")
    c[1].metric("Expected demand (protection period)", f"{row['expected_demand']:.1f}")
    c[2].metric("Safety stock", f"{row['safety_stock']:.1f}")
    c[3].metric("On hand + on order", f"{int(row['on_hand'])} + {int(row['on_order'])}")


def plan_tab() -> None:
    plan = load("gold", "replenishment_plan")
    cols = st.columns(3)
    store = cols[0].selectbox(
        "Store", ["All", *sorted(plan["store_id"].unique())], key="plan_store"
    )
    aisle = cols[1].selectbox(
        "Aisle", ["All", *sorted(plan["aisle"].unique())], key="plan_aisle", format_func=nice
    )
    only_orders = cols[2].toggle("Only lines with an order", value=True)
    view = plan
    if store != "All":
        view = view[view["store_id"] == store]
    if aisle != "All":
        view = view[view["aisle"] == aisle]
    if only_orders:
        view = view[view["recommended_order_qty"] > 0]
    show = view[
        ["store_id", "sku_id", "sku_name", "aisle", "protection_days", "expected_demand", "safety_factor",
         "safety_stock", "order_up_to_level", "on_hand", "on_order", "recommended_order_qty"]
    ].sort_values(["store_id", "aisle", "sku_name"])  # fmt: skip
    st.caption(f"{len(show):,} lines, {int(show['recommended_order_qty'].sum()):,} units in total.")
    st.dataframe(show.round(2), hide_index=True, use_container_width=True)
    st.download_button(
        "Download CSV", show.to_csv(index=False), "replenishment_plan.csv", "text/csv"
    )


def main() -> None:
    st.title("Convenience-store demand forecasting")
    require_outputs()
    summary, explorer, plan = st.tabs(["Summary", "Explore a product", "Order plan"])
    with summary:
        summary_tab()
    with explorer:
        explorer_tab()
    with plan:
        plan_tab()


main()
