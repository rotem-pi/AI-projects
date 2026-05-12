"""
Definity – Incidents and Tests Dashboard
Live interactive dashboard over test_runs_base.csv.
The CSV is refreshed separately by refresh_data.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "test_runs_base.csv"

TEAL   = "#00C4CC"
TEAL2  = "#008B99"
TEAL3  = "#3DD6D9"
BLUE   = "#0EA5E9"
ORANGE = "#F97316"
RED    = "#EF4444"
GREEN  = "#10B981"
GRAY   = "#6B7280"
BG     = "#F5F6FA"
CARD   = "#FFFFFF"
BORDER = "#E5E7EB"
TEXT   = "#111827"
MUTED  = "#6B7280"

TYPE_COLORS = {
    "Range":   TEAL,
    "PctDiff": BLUE,
    "Const":   ORANGE,
    "Trend":   GREEN,
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for col in ["created_time", "updated_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["app_pit"] = pd.to_datetime(df["app_pit"], utc=True, errors="coerce")
    if "test_def_id" not in df.columns:
        df["test_def_id"] = df["test_id"]
    df["failed"] = 1 - df["is_passed_binary"]
    df["date"]   = df["created_time"].dt.tz_convert("UTC").dt.normalize()
    _ct = df["created_time"].dt.tz_convert("UTC")
    df["week"] = (_ct - pd.to_timedelta(_ct.dt.dayofweek, unit="D")).dt.normalize()
    return df


print("Loading data…")
DF_RAW = load_data()
print(f"Loaded {len(DF_RAW):,} rows")

DATE_MIN = DF_RAW["created_time"].min().date()
DATE_MAX = DF_RAW["created_time"].max().date()
TEST_TYPES = sorted(DF_RAW["test_type"].dropna().unique().tolist())
TENANTS = sorted(DF_RAW["tenant_id"].dropna().unique().tolist())

# ── Chart descriptions for info tooltips ──────────────────────────────────────
CHART_INFO = {
    "chart-timeseries": (
        "Daily volume of test runs (bars) overlaid with the incident rate — the share of runs "
        "that fired an alert (line)."
    ),
    "chart-by-type": (
        "Total runs and incident rate broken down by test algorithm (Range, PctDiff, Const, Trend). "
    ),
    "chart-weekly-type": (
        "Week-over-week incident rate trend for each test type."
    ),
    "chart-asset-type": (
        "Top 15 asset types ranked by incident count. Bar opacity encodes the incident rate — "
        "darker means a higher proportion of runs triggered an alert."
    ),
    "chart-top-pairs": (
        "Top 15 combinations of metric type × asset type by raw incident count. "
        "Color encodes incident rate (darker = higher)."
    ),
    "chart-tenant-vol": (
        "Stability of each tenant's weekly incident rate (standard deviation across weeks). "
        "High volatility means a tenant's alert behaviour changes a lot week-to-week — "
        "a signal of misconfiguration or consistently noisy tests."
    ),
    "chart-app-spread": (
        "Distribution of per-app incident rates within each tenant. Each dot is one app. "
        "A wide spread means some apps are well-configured while others are highly noisy."
    ),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_pct(v):
    return f"{v*100:.1f}%"

def card(children, style=None):
    base = {
        "background": CARD,
        "borderRadius": "8px",
        "border": f"1px solid {BORDER}",
        "padding": "20px",
        "marginBottom": "16px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def section_title(text):
    return html.H3(text, style={
        "fontSize": "13px",
        "fontWeight": "600",
        "color": MUTED,
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "marginBottom": "12px",
        "marginTop": "0",
    })

def chart_header(title_text, chart_id):
    """Title row with an ⓘ icon that reveals a description tooltip on hover."""
    info_id = f"info-{chart_id}"
    desc = CHART_INFO.get(chart_id, "")
    return html.Div([
        html.H4(title_text, style={
            "fontSize": "14px",
            "fontWeight": "600",
            "color": TEXT,
            "margin": "0",
        }),
        html.Span(
            "ⓘ",
            id=info_id,
            style={
                "fontSize": "14px",
                "color": MUTED,
                "cursor": "help",
                "marginLeft": "7px",
                "userSelect": "none",
                "lineHeight": "1",
            },
        ),
        dbc.Tooltip(
            desc,
            target=info_id,
            placement="top",
            style={
                "maxWidth": "320px",
                "fontSize": "12px",
                "lineHeight": "1.5",
                "textAlign": "left",
            },
        ),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})

# Show toolbar only on hover so icons never overlap chart content
PLOTLY_CONFIG = dict(
    displayModeBar="hover",
    displaylogo=False,
    modeBarButtonsToRemove=["select2d", "lasso2d"],
)

LAYOUT_BASE = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family="Inter, -apple-system, sans-serif", color=TEXT, size=12),
    hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(size=12, color=TEXT)),
)
MARGIN_DEFAULT = dict(t=14, b=40, l=50, r=20)

LEGEND_TOP = dict(
    bgcolor="rgba(0,0,0,0)", borderwidth=0,
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
)

# ── App init ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    title="Incidents and Tests Dashboard",
    assets_folder="assets",
)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/logo.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ── Layout helpers ─────────────────────────────────────────────────────────────
filter_bar = html.Div([
    html.Div([
        html.Div([
            html.Label("DATE RANGE", style={"fontSize":"11px","fontWeight":"600","color":MUTED,"marginBottom":"4px","display":"block"}),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=DATE_MIN,
                max_date_allowed=DATE_MAX,
                start_date=DATE_MIN,
                end_date=DATE_MAX,
                display_format="MMM DD, YYYY",
                style={"fontSize":"13px"},
            ),
        ], style={"marginRight":"20px"}),

        html.Div([
            html.Label("TEST TYPE", style={"fontSize":"11px","fontWeight":"600","color":MUTED,"marginBottom":"4px","display":"block"}),
            dcc.Dropdown(
                id="filter-test-type",
                options=[{"label": t, "value": t} for t in TEST_TYPES],
                multi=True,
                placeholder="All types",
                style={"minWidth":"220px","fontSize":"13px"},
                clearable=True,
            ),
        ], style={"marginRight":"20px"}),

        html.Div([
            html.Label("TENANT", style={"fontSize":"11px","fontWeight":"600","color":MUTED,"marginBottom":"4px","display":"block"}),
            dcc.Dropdown(
                id="filter-tenant",
                options=[{"label": f"Tenant {t}", "value": t} for t in TENANTS],
                multi=True,
                placeholder="All tenants",
                style={"minWidth":"220px","fontSize":"13px"},
                clearable=True,
            ),
        ], style={"marginRight":"20px"}),

        html.Div([
            html.Label(" ", style={"fontSize":"11px","display":"block","marginBottom":"4px"}),
            html.Button("Apply Filters", id="btn-apply", n_clicks=0, style={
                "background": TEAL, "color": "white", "border": "none",
                "borderRadius": "6px", "padding": "8px 18px",
                "fontWeight": "600", "fontSize": "13px", "cursor": "pointer",
                "fontFamily": "inherit",
            }),
        ]),
    ], style={"display":"flex","alignItems":"flex-end","flexWrap":"wrap","gap":"4px"}),
], style={
    "background": CARD,
    "borderBottom": f"1px solid {BORDER}",
    "padding": "16px 24px",
})

charts_section = html.Div([
    html.Div([
        card([
            chart_header("Daily Runs vs Incident Rate", "chart-timeseries"),
            dcc.Graph(id="chart-timeseries", config=PLOTLY_CONFIG),
        ]),
    ]),

    html.Div([
        html.Div([
            card([
                chart_header("Runs & Incident Rate by Test Type", "chart-by-type"),
                dcc.Graph(id="chart-by-type", config=PLOTLY_CONFIG),
            ]),
        ], style={"flex":"1","minWidth":"0"}),
        html.Div([
            card([
                chart_header("Weekly Incident Rate by Test Type", "chart-weekly-type"),
                dcc.Graph(id="chart-weekly-type", config=PLOTLY_CONFIG),
            ]),
        ], style={"flex":"1","minWidth":"0"}),
    ], style={"display":"flex","gap":"16px"}),

    html.Div([
        html.Div([
            card([
                chart_header("Incidents by Asset Type", "chart-asset-type"),
                dcc.Graph(id="chart-asset-type", config=PLOTLY_CONFIG),
            ]),
        ], style={"flex":"1","minWidth":"0"}),
        html.Div([
            card([
                chart_header("Top Metric × Asset Pairs by Incident Count", "chart-top-pairs"),
                dcc.Graph(id="chart-top-pairs", config=PLOTLY_CONFIG),
            ]),
        ], style={"flex":"1","minWidth":"0"}),
    ], style={"display":"flex","gap":"16px"}),

    html.Div([
        html.Div([
            card([
                chart_header("Tenant Volatility (Std of Weekly Incident Rate)", "chart-tenant-vol"),
                dcc.Graph(id="chart-tenant-vol", config=PLOTLY_CONFIG),
            ]),
        ], style={"flex":"1","minWidth":"0"}),
        html.Div([
            card([
                chart_header("Incident Rate Distribution by Tenant", "chart-app-spread"),
                dcc.Graph(id="chart-app-spread", config=PLOTLY_CONFIG),
            ]),
        ], style={"flex":"1","minWidth":"0"}),
    ], style={"display":"flex","gap":"16px"}),
])

scorecard_section = html.Div([
    card([
        section_title("Miscalibration Scorecards"),
        dcc.Tabs(id="sc-tabs", value="range", children=[
            dcc.Tab(label="Range", value="range",
                style={"padding":"8px 16px","fontSize":"13px"},
                selected_style={"padding":"8px 16px","fontSize":"13px","fontWeight":"600","borderTop":f"2px solid {TEAL}","color":TEAL}),
            dcc.Tab(label="PctDiff", value="pctdiff",
                style={"padding":"8px 16px","fontSize":"13px"},
                selected_style={"padding":"8px 16px","fontSize":"13px","fontWeight":"600","borderTop":f"2px solid {BLUE}","color":BLUE}),
            dcc.Tab(label="Const", value="const",
                style={"padding":"8px 16px","fontSize":"13px"},
                selected_style={"padding":"8px 16px","fontSize":"13px","fontWeight":"600","borderTop":f"2px solid {ORANGE}","color":ORANGE}),
            dcc.Tab(label="Trend", value="trend",
                style={"padding":"8px 16px","fontSize":"13px"},
                selected_style={"padding":"8px 16px","fontSize":"13px","fontWeight":"600","borderTop":f"2px solid {GREEN}","color":GREEN}),
        ], style={"marginBottom":"16px"}),
        html.Div(id="scorecard-content"),
    ]),
])

# ── Deep-dive section ──────────────────────────────────────────────────────────
deep_dive_section = card([
    html.Div([
        html.Div([
            html.H3("Per-Test Deep-Dive", style={
                "fontSize":"14px","fontWeight":"700","color":TEXT,"margin":"0 0 2px 0",
            }),
            html.Span(
                "Enter a Test ID to see its full run-value timeline, bounds, and incidents.",
                style={"fontSize":"12px","color":MUTED},
            ),
        ]),
    ], style={"marginBottom":"16px"}),

    # Search row
    html.Div([
        html.Div([
            html.Label("TEST ID", style={"fontSize":"11px","fontWeight":"600","color":MUTED,"marginBottom":"4px","display":"block"}),
            dcc.Input(
                id="dd-test-id",
                type="number",
                placeholder="e.g. 107671",
                debounce=False,
                style={
                    "width":"180px","padding":"7px 12px","fontSize":"13px",
                    "border":f"1px solid {BORDER}","borderRadius":"6px",
                    "fontFamily":"inherit","outline":"none",
                },
            ),
        ], style={"marginRight":"12px"}),
        html.Div([
            html.Label(" ", style={"fontSize":"11px","display":"block","marginBottom":"4px"}),
            html.Button("Look up", id="dd-btn", n_clicks=0, style={
                "background":TEAL,"color":"white","border":"none",
                "borderRadius":"6px","padding":"8px 18px",
                "fontWeight":"600","fontSize":"13px","cursor":"pointer","fontFamily":"inherit",
            }),
        ]),
        html.Div(id="dd-error", style={"marginLeft":"16px","fontSize":"13px","color":RED,"alignSelf":"flex-end","paddingBottom":"2px"}),
    ], style={"display":"flex","alignItems":"flex-end","marginBottom":"20px"}),

    # Metadata row (populated by callback)
    html.Div(id="dd-meta", style={"marginBottom":"16px"}),

    # Main chart + stats
    html.Div(id="dd-chart-wrap"),
])

app.layout = html.Div([
    # ── Top nav ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Img(
                src="/assets/logo.svg",
                style={"height": "32px", "width": "32px", "marginRight": "10px"},
            ),
            html.Span("Definity", style={"fontWeight":"700","fontSize":"16px","color":TEXT,"marginRight":"6px"}),
            html.Span("Incidents and Tests Dashboard", style={"fontWeight":"400","fontSize":"14px","color":MUTED}),
        ], style={"display":"flex","alignItems":"center"}),
        html.Div([
            html.Span(f"Data: {DATE_MIN} – {DATE_MAX}", style={"fontSize":"12px","color":MUTED}),
        ]),
    ], style={
        "display":"flex","justifyContent":"space-between","alignItems":"center",
        "background":CARD,"borderBottom":f"1px solid {BORDER}",
        "padding":"12px 24px","position":"sticky","top":"0","zIndex":"100",
        "boxShadow":"0 1px 4px rgba(0,0,0,.06)",
    }),

    filter_bar,

    html.Div([
        html.Div(id="kpi-row", style={"display":"flex","gap":"16px","marginBottom":"16px"}),
        charts_section,
        scorecard_section,
        deep_dive_section,
    ], style={"padding":"20px 24px","maxWidth":"1600px","margin":"0 auto"}),

    dcc.Store(id="filter-store"),

], style={"background":BG,"minHeight":"100vh","fontFamily":"Inter,-apple-system,sans-serif","color":TEXT})


# ── Callbacks ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("filter-store", "data"),
    Input("btn-apply", "n_clicks"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("filter-test-type", "value"),
    State("filter-tenant", "value"),
    prevent_initial_call=False,
)
def store_filters(n, start, end, types, tenants):
    return {"start": str(start), "end": str(end), "types": types or [], "tenants": tenants or []}


def apply_filters(data: dict) -> pd.DataFrame:
    df = DF_RAW.copy()
    if data:
        start = pd.Timestamp(data["start"], tz="UTC") if data.get("start") else None
        end   = pd.Timestamp(data["end"],   tz="UTC") if data.get("end")   else None
        if start:
            df = df[df["created_time"] >= start]
        if end:
            df = df[df["created_time"] < end + pd.Timedelta(days=1)]
        if data.get("types"):
            df = df[df["test_type"].isin(data["types"])]
        if data.get("tenants"):
            df = df[df["tenant_id"].isin(data["tenants"])]
    return df


# ── KPI cards ──────────────────────────────────────────────────────────────────
def make_kpi(label, value, sub=None, color=TEXT):
    return html.Div([
        html.Div(label, style={"fontSize":"11px","fontWeight":"600","color":MUTED,"textTransform":"uppercase","letterSpacing":"0.05em","marginBottom":"6px"}),
        html.Div(value, style={"fontSize":"28px","fontWeight":"700","color":color,"lineHeight":"1"}),
        html.Div(sub, style={"fontSize":"12px","color":MUTED,"marginTop":"4px"}) if sub else html.Div(),
    ], style={
        "background":CARD,"borderRadius":"8px","border":f"1px solid {BORDER}",
        "padding":"16px 20px","flex":"1","minWidth":"160px",
        "boxShadow":"0 1px 3px rgba(0,0,0,.05)",
    })

@app.callback(Output("kpi-row", "children"), Input("filter-store", "data"))
def update_kpis(data):
    df = apply_filters(data)
    n_runs    = len(df)
    n_inc     = int(df["failed"].sum())
    inc_rate  = n_inc / n_runs if n_runs else 0
    n_tests   = df["test_def_id"].nunique()
    n_tenants = df["tenant_id"].nunique()
    n_assets  = df["asset_type"].nunique()

    return [
        make_kpi("Total Runs",      f"{n_runs:,}",  "test executions"),
        make_kpi("Incidents",       f"{n_inc:,}",   f"{fmt_pct(inc_rate)} incident rate", RED),
        make_kpi("Pass Rate",       fmt_pct(1 - inc_rate), "of all runs", GREEN),
        make_kpi("Unique Tests",    f"{n_tests:,}", "test definitions"),
        make_kpi("Tenants",         str(n_tenants), "active tenants"),
        make_kpi("Asset Types",     str(n_assets),  "unique asset types"),
    ]


# ── Chart: time series ─────────────────────────────────────────────────────────
@app.callback(Output("chart-timeseries", "figure"), Input("filter-store", "data"))
def chart_timeseries(data):
    df = apply_filters(data).dropna(subset=["created_time"])
    ts = (
        df.sort_values("created_time")
        .set_index("created_time")
        .resample("D")
        .agg(runs=("failed","size"), incidents=("failed","sum"))
        .reset_index()
    )
    ts["incident_rate"] = ts["incidents"] / ts["runs"].replace(0, np.nan)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=ts["created_time"], y=ts["runs"],
        name="Runs", marker_color=TEAL, opacity=0.7,
        hovertemplate="%{x|%b %d}<br>Runs: %{y:,}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=ts["created_time"], y=ts["incident_rate"],
        name="Incident Rate", line=dict(color=RED, width=2),
        mode="lines+markers", marker_size=4,
        hovertemplate="%{x|%b %d}<br>Incident Rate: %{y:.1%}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(**LAYOUT_BASE, margin=MARGIN_DEFAULT, height=280, showlegend=True, legend=LEGEND_TOP)
    fig.update_xaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(title_text="Daily Runs", gridcolor="#F0F1F5", linecolor=BORDER, secondary_y=False)
    fig.update_yaxes(title_text="Incident Rate", tickformat=".0%", gridcolor="rgba(0,0,0,0)", secondary_y=True)
    return fig


# ── Chart: by test type ────────────────────────────────────────────────────────
@app.callback(Output("chart-by-type", "figure"), Input("filter-store", "data"))
def chart_by_type(data):
    df = apply_filters(data)
    by_tt = (
        df.groupby("test_type", dropna=False)
        .agg(runs=("failed","size"), incidents=("failed","sum"))
        .reset_index()
        .dropna(subset=["test_type"])
    )
    by_tt["incident_rate"] = by_tt["incidents"] / by_tt["runs"].replace(0, np.nan)
    colors = [TYPE_COLORS.get(t, TEAL) for t in by_tt["test_type"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=by_tt["test_type"], y=by_tt["runs"],
        name="Runs", marker_color=colors, opacity=0.8,
        text=[f"{int(v):,}" for v in by_tt["runs"]], textposition="outside",
        hovertemplate="%{x}<br>Runs: %{y:,}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=by_tt["test_type"], y=by_tt["incident_rate"],
        name="Incident Rate", mode="lines+markers",
        line=dict(color=RED, width=2, dash="dot"), marker_size=8,
        hovertemplate="%{x}<br>Incident Rate: %{y:.1%}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(**LAYOUT_BASE, margin=MARGIN_DEFAULT, height=280, legend=LEGEND_TOP)
    fig.update_xaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(title_text="Runs", gridcolor="#F0F1F5", linecolor=BORDER, secondary_y=False)
    fig.update_yaxes(title_text="Incident Rate", tickformat=".0%", gridcolor="rgba(0,0,0,0)", secondary_y=True)
    return fig


# ── Chart: weekly by test type ─────────────────────────────────────────────────
@app.callback(Output("chart-weekly-type", "figure"), Input("filter-store", "data"))
def chart_weekly_type(data):
    df = apply_filters(data).dropna(subset=["created_time", "test_type"])
    order = df["test_type"].value_counts().index.tolist()

    fig = go.Figure()
    for tt in order:
        sub = df[df["test_type"] == tt]
        wk = (
            sub.groupby("week", dropna=False)
            .agg(runs=("failed","size"), incidents=("failed","sum"))
            .reset_index()
        )
        wk["incident_rate"] = wk["incidents"] / wk["runs"].replace(0, np.nan)
        fig.add_trace(go.Scatter(
            x=wk["week"], y=wk["incident_rate"], name=tt,
            mode="lines+markers", line=dict(color=TYPE_COLORS.get(tt, TEAL), width=2), marker_size=4,
            hovertemplate=f"{tt}<br>%{{x|%b %d}}<br>Rate: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(**LAYOUT_BASE, margin=MARGIN_DEFAULT, height=280, legend=LEGEND_TOP)
    fig.update_xaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(tickformat=".0%", title_text="Incident Rate", gridcolor="#F0F1F5", linecolor=BORDER)
    return fig


# ── Chart: asset type ──────────────────────────────────────────────────────────
@app.callback(Output("chart-asset-type", "figure"), Input("filter-store", "data"))
def chart_asset_type(data):
    df = apply_filters(data)
    at = (
        df.groupby("asset_type", dropna=False)
        .agg(runs=("failed","size"), incidents=("failed","sum"))
        .reset_index()
        .dropna(subset=["asset_type"])
    )
    at["incident_rate"] = at["incidents"] / at["runs"].replace(0, np.nan)
    at = at.sort_values("incidents", ascending=True).tail(15)

    max_r = at["incident_rate"].max() or 1
    colors = [f"rgba(0,196,204,{0.3 + 0.7*(v/max_r):.2f})" for v in at["incident_rate"]]

    fig = go.Figure(go.Bar(
        y=at["asset_type"].astype(str), x=at["incidents"],
        orientation="h", marker_color=colors,
        text=[f"{int(v):,} ({fmt_pct(r)})" for v, r in zip(at["incidents"], at["incident_rate"])],
        textposition="outside",
        hovertemplate="%{y}<br>Incidents: %{x:,}<br>Rate: %{customdata:.1%}<extra></extra>",
        customdata=at["incident_rate"],
    ))
    fig.update_layout(**LAYOUT_BASE, height=320, showlegend=False,
                      margin=dict(t=14, b=40, l=100, r=80))
    fig.update_xaxes(title_text="Incident Count", gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    return fig


# ── Chart: top pairs ───────────────────────────────────────────────────────────
@app.callback(Output("chart-top-pairs", "figure"), Input("filter-store", "data"))
def chart_top_pairs(data):
    df = apply_filters(data)
    pair = (
        df.groupby(["metric_type","asset_type"], dropna=False)
        .agg(runs=("failed","size"), incidents=("failed","sum"))
        .reset_index()
    )
    pair["incident_rate"] = pair["incidents"] / pair["runs"].replace(0, np.nan)
    top = pair.sort_values("incidents", ascending=True).tail(15)
    labels = [f"{str(m)} × {str(a)}" for m, a in zip(top["metric_type"], top["asset_type"])]

    fig = go.Figure(go.Bar(
        y=labels, x=top["incidents"], orientation="h",
        marker=dict(
            color=top["incident_rate"],
            colorscale=[[0, "#E0F7FA"], [0.5, TEAL], [1, "#0077B6"]],
            showscale=True,
            colorbar=dict(title="Incident Rate", tickformat=".0%", thickness=12, len=0.7),
        ),
        hovertemplate="%{y}<br>Incidents: %{x:,}<br>Rate: %{marker.color:.1%}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, height=320, showlegend=False,
                      margin=dict(t=14, b=40, l=130, r=60))
    fig.update_xaxes(title_text="Incident Count", gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    return fig


# ── Chart: tenant volatility ───────────────────────────────────────────────────
@app.callback(Output("chart-tenant-vol", "figure"), Input("filter-store", "data"))
def chart_tenant_vol(data):
    df = apply_filters(data).dropna(subset=["created_time"])
    tw = (
        df.groupby(["tenant_id","week"], dropna=False)
        .agg(runs=("failed","size"), incidents=("failed","sum"))
        .reset_index()
    )
    tw = tw[tw["runs"] >= 10]
    tw["incident_rate"] = tw["incidents"] / tw["runs"].replace(0, np.nan)
    vol = (
        tw.groupby("tenant_id")["incident_rate"]
        .agg(volatility="std", mean_rate="mean")
        .reset_index()
        .dropna(subset=["volatility"])
        .sort_values("volatility", ascending=True)
    )

    fig = go.Figure(go.Bar(
        y=[f"T-{t}" for t in vol["tenant_id"]],
        x=vol["volatility"],
        orientation="h",
        marker_color=TEAL,
        opacity=0.8,
        text=[f"σ={v:.3f}  avg={fmt_pct(m)}" for v, m in zip(vol["volatility"], vol["mean_rate"])],
        textposition="outside",
        hovertemplate="Tenant %{y}<br>Volatility (σ): %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, height=280, showlegend=False,
                      margin=dict(t=14, b=40, l=60, r=120))
    fig.update_xaxes(title_text="Std Dev of Weekly Incident Rate", gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    return fig


# ── Chart: app spread ──────────────────────────────────────────────────────────
@app.callback(Output("chart-app-spread", "figure"), Input("filter-store", "data"))
def chart_app_spread(data):
    df = apply_filters(data)
    app_rates = (
        df.groupby(["tenant_id","app_id"], dropna=False)
        .agg(runs=("failed","size"), incidents=("failed","sum"))
        .reset_index()
    )
    app_rates["incident_rate"] = app_rates["incidents"] / app_rates["runs"].replace(0, np.nan)
    app_rates = app_rates[app_rates["runs"] >= 20].dropna(subset=["incident_rate"])

    vc  = app_rates.groupby("tenant_id")["app_id"].nunique()
    sub = app_rates[app_rates["tenant_id"].isin(vc[vc >= 2].index)]
    top = sub.groupby("tenant_id")["runs"].sum().sort_values(ascending=False).head(8).index
    sub = sub[sub["tenant_id"].isin(top)]

    fig = go.Figure()
    for tid in top:
        vals = sub[sub["tenant_id"] == tid]["incident_rate"].tolist()
        fig.add_trace(go.Box(
            y=vals, name=f"T-{tid}",
            marker_color=TEAL, line_color=TEAL2,
            boxpoints="all", jitter=0.4, pointpos=-1.6,
            marker=dict(opacity=0.4, size=4),
            hovertemplate=f"Tenant {tid}<br>Rate: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(**LAYOUT_BASE, margin=MARGIN_DEFAULT, height=280, showlegend=False)
    fig.update_xaxes(gridcolor="#F0F1F5", linecolor=BORDER)
    fig.update_yaxes(tickformat=".0%", title_text="Per-App Incident Rate", gridcolor="#F0F1F5", linecolor=BORDER)
    return fig


# ── Scorecards ─────────────────────────────────────────────────────────────────
def _col_name(c):
    if c == "test_def_id": return "Test ID"
    if c == "test_id":     return "Test ID"
    return c.replace("_"," ").title()

ACTION_LEGEND = [
    ("CANDIDATE_DISABLE",  RED,    "Extremely high incident rate — consider disabling this test entirely."),
    ("NOISE_GENERATOR",    RED,    "High variance with near-zero signal — test produces mostly noise."),
    ("RECALIBRATE",        ORANGE, "Bounds are systematically off; recalibrate lower/upper limits from recent data."),
    ("WIDEN_TOLERANCE",    TEAL2,  "Bounds are too tight relative to observed spread; widen the tolerance band."),
    ("INCREASE_TOLERANCE", BLUE,   "Percentage tolerance is too small; increase var1 to reduce false incidents."),
    ("STALE_CONFIG",       RED,    "Trend slope has drifted far from the configured rate — update the trend parameters."),
    ("STALE_INTERCEPT",    ORANGE, "Trend intercept has shifted significantly from the baseline — reset the origin."),
    ("INVESTIGATE",        ORANGE, "Unusual pattern detected — manual review recommended before changing config."),
    ("OK",                 TEAL2,  "No significant miscalibration detected."),
]

def _action_legend_panel():
    return html.Details([
        html.Summary("Action legend", style={
            "fontSize":"11px","fontWeight":"600","color":MUTED,
            "cursor":"pointer","userSelect":"none","marginBottom":"8px",
            "listStyle":"none","display":"flex","alignItems":"center","gap":"4px",
        }),
        html.Div([
            html.Div([
                html.Span(code, style={
                    "display":"inline-block","marginRight":"8px",
                    "padding":"2px 8px","borderRadius":"10px","fontSize":"10px","fontWeight":"700",
                    "color": clr, "background": clr + "22", "whiteSpace":"nowrap",
                }),
                html.Span(desc, style={"fontSize":"11px","color":MUTED}),
            ], style={"marginBottom":"5px","display":"flex","alignItems":"center"})
            for code, clr, desc in ACTION_LEGEND
        ]),
    ], style={"marginBottom":"14px"})

TABLE_STYLE = dict(
    style_table={"overflowX":"auto","borderRadius":"6px","border":f"1px solid {BORDER}"},
    style_header={
        "backgroundColor":"#F9FAFB","fontWeight":"600","fontSize":"12px",
        "color":TEXT,"borderBottom":f"1px solid {BORDER}","padding":"10px 12px",
    },
    style_cell={
        "fontSize":"12px","color":TEXT,"padding":"9px 12px",
        "borderBottom":f"1px solid #F3F4F6","backgroundColor":CARD,
        "fontFamily":"Inter,-apple-system,sans-serif",
    },
    style_data_conditional=[
        {"if":{"row_index":"odd"},"backgroundColor":"#FAFAFA"},
        {"if":{"filter_query":'{action} contains "DISABLE"'}, "color":RED,   "fontWeight":"600"},
        {"if":{"filter_query":'{action} contains "RECALIB"'}, "color":ORANGE},
        {"if":{"filter_query":'{action} contains "WIDEN"'},   "color":TEAL2},
        {"if":{"filter_query":'{action} contains "INCREASE"'},"color":BLUE},
        {"if":{"filter_query":'{action} contains "INVESTIGATE"'},"color":ORANGE},
        {"if":{"filter_query":'{action} contains "STALE"'},   "color":RED, "fontWeight":"600"},
    ],
    page_size=20,
    sort_action="native",
)

def build_range_scorecard(df):
    r = df[df["test_type"] == "Range"].copy()
    r["bound_width"] = r["upper_bound"] - r["lower_bound"]
    r = r[r["bound_width"] > 0].dropna(subset=["run_value","lower_bound","upper_bound"])
    r["dist_lower"] = (r["run_value"] - r["lower_bound"]) / r["bound_width"]
    r["dist_upper"] = (r["upper_bound"] - r["run_value"]) / r["bound_width"]
    r["borderline"] = (
        (r["dist_lower"].between(-0.05, 0.05)) | (r["dist_upper"].between(-0.05, 0.05))
    ).astype(int)

    sc = (r.groupby("test_def_id", dropna=False).agg(
        runs=("failed","size"), incidents=("failed","sum"),
        metric_type=("metric_type","first"), asset_type=("asset_type","first"),
        tenant_id=("tenant_id","first"),
        borderline_sum=("borderline","sum"),
        median_lower=("lower_bound","median"), median_upper=("upper_bound","median"),
    ).reset_index())
    sc["incident_rate"]    = sc["incidents"] / sc["runs"].replace(0, np.nan)
    sc["borderline_ratio"] = sc["borderline_sum"] / sc["runs"].replace(0, np.nan)
    sc["urgency_score"]    = sc["incident_rate"] * 0.6 + sc["borderline_ratio"] * 0.4

    def action(row):
        if row["incident_rate"] > 0.8: return "CANDIDATE_DISABLE"
        if row["incident_rate"] > 0.3: return "RECALIBRATE"
        if row["borderline_ratio"] > 0.5: return "WIDEN_SHIFT_BOUNDS"
        return "OK"
    sc["action"] = sc.apply(action, axis=1)
    sc = sc[sc["action"] != "OK"].sort_values("urgency_score", ascending=False).head(50)
    cols = ["test_def_id","metric_type","asset_type","tenant_id","runs","incident_rate","borderline_ratio","action","urgency_score"]
    sc = sc[cols].copy()
    sc["incident_rate"]    = sc["incident_rate"].map(lambda x: f"{x:.1%}")
    sc["borderline_ratio"] = sc["borderline_ratio"].map(lambda x: f"{x:.1%}")
    sc["urgency_score"]    = sc["urgency_score"].map(lambda x: f"{x:.3f}")
    return sc

def build_pctdiff_scorecard(df):
    p = df[df["test_type"] == "PctDiff"].copy()
    p = p.dropna(subset=["run_value","lower_bound","upper_bound","var1","var2"])
    p = p[(p["var1"] > 0) & (p["var2"] > 0)]
    p["bound_width"] = (p["upper_bound"] - p["lower_bound"]).replace(0, np.nan)
    p["dist_lower"]  = (p["run_value"] - p["lower_bound"]) / p["bound_width"]
    p["dist_upper"]  = (p["upper_bound"] - p["run_value"]) / p["bound_width"]
    p["borderline"]  = (
        (p["dist_lower"].between(-0.05,0.05)) | (p["dist_upper"].between(-0.05,0.05))
    ).astype(int)
    baseline = (p["lower_bound"] + p["upper_bound"]) / 2
    p["deviation"] = ((p["run_value"] - baseline) / baseline.replace(0, np.nan)).abs()

    sc = (p.groupby("test_def_id", dropna=False).agg(
        runs=("failed","size"), incidents=("failed","sum"),
        metric_type=("metric_type","first"), asset_type=("asset_type","first"),
        tenant_id=("tenant_id","first"),
        borderline_sum=("borderline","sum"),
        p95_dev=("deviation", lambda x: np.nanpercentile(x, 95)),
        var1=("var1","median"), var2=("var2","median"),
    ).reset_index())
    sc["incident_rate"]    = sc["incidents"] / sc["runs"].replace(0, np.nan)
    sc["borderline_ratio"] = sc["borderline_sum"] / sc["runs"].replace(0, np.nan)
    sc["urgency_score"]    = sc["incident_rate"] * 0.6 + sc["borderline_ratio"] * 0.4

    def action(row):
        if row["incident_rate"] > 0.8: return "CANDIDATE_DISABLE"
        if row["p95_dev"] > row["var1"] * 2: return "WIDEN_TOLERANCE"
        if row["incident_rate"] > 0.3 and row["var2"] < 5: return "INCREASE_WINDOW"
        if row["borderline_ratio"] > 0.5: return "NOISE_GENERATOR"
        return "OK"
    sc["sug_var1"] = sc["p95_dev"].apply(
        lambda x: math.ceil(x * 110) / 100 if pd.notna(x) else np.nan
    )
    sc["action"] = sc.apply(action, axis=1)
    sc = sc[sc["action"] != "OK"].sort_values("urgency_score", ascending=False).head(50)
    cols = ["test_def_id","metric_type","asset_type","tenant_id","runs",
            "incident_rate","borderline_ratio","p95_dev","var1","sug_var1","var2","action","urgency_score"]
    sc = sc[[c for c in cols if c in sc.columns]].copy()
    sc["incident_rate"]    = sc["incident_rate"].map(lambda x: f"{x:.1%}")
    sc["borderline_ratio"] = sc["borderline_ratio"].map(lambda x: f"{x:.1%}")
    sc["p95_dev"]          = sc["p95_dev"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    sc["sug_var1"]         = sc["sug_var1"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    sc["urgency_score"]    = sc["urgency_score"].map(lambda x: f"{x:.3f}")
    return sc

def build_const_scorecard(df):
    c = df[df["test_type"] == "Const"].copy()
    c = c.dropna(subset=["run_value","var1"])
    sc = (c.groupby("test_def_id", dropna=False).agg(
        runs=("failed","size"), incidents=("failed","sum"),
        expected_const=("var1","median"),
        metric_type=("metric_type","first"), asset_type=("asset_type","first"),
        tenant_id=("tenant_id","first"),
    ).reset_index())
    sc["incident_rate"]  = sc["incidents"] / sc["runs"].replace(0, np.nan)
    sc["urgency_score"]  = sc["incident_rate"] * sc["runs"] / (sc["runs"].max() or 1)

    def action(row):
        if row["incident_rate"] > 0.5: return "CANDIDATE_DISABLE"
        if row["incident_rate"] > 0.1 and row["expected_const"] == 0: return "INVESTIGATE_ZERO_CONST"
        if row["incident_rate"] > 0.1: return "REVIEW_THRESHOLD"
        return "OK"
    sc["action"] = sc.apply(action, axis=1)
    sc = sc[sc["action"] != "OK"].sort_values("urgency_score", ascending=False).head(50)
    cols = ["test_def_id","metric_type","asset_type","tenant_id","expected_const","runs","incident_rate","action","urgency_score"]
    sc = sc[[c for c in cols if c in sc.columns]].copy()
    sc["incident_rate"]  = sc["incident_rate"].map(lambda x: f"{x:.1%}")
    sc["urgency_score"]  = sc["urgency_score"].map(lambda x: f"{x:.3f}")
    return sc


def build_trend_scorecard(df):
    t = df[df["test_type"] == "Trend"].copy()
    t["app_pit"] = pd.to_datetime(t["app_pit"], utc=True, errors="coerce")
    t = t.dropna(subset=["app_pit","run_value"])

    records = []
    for tid, g in t.groupby("test_id", dropna=False):
        g = g.sort_values("app_pit").copy()
        if len(g) < 3:
            continue
        t0 = g["app_pit"].min()
        g["t_hr"] = (g["app_pit"] - t0).dt.total_seconds() / 3600
        slope, intercept, r_val, _, _ = scipy_stats.linregress(g["t_hr"], g["run_value"])
        abs_resid = (g["run_value"] - (intercept + slope * g["t_hr"])).abs()
        resid_std = float(abs_resid.std())
        resid_p97 = float(abs_resid.quantile(0.97))

        var1 = float(g["var1"].median())
        var2 = float(g["var2"].median())
        var3 = float(g["var3"].median()) if "var3" in g.columns else np.nan
        slope_ratio   = slope / var1 if var1 != 0 else np.nan
        intercept_gap = abs(intercept - var2) / var3 if (pd.notna(var3) and var3 > 0) else np.nan
        sug_var3      = round(max(resid_p97, var3 if pd.notna(var3) else 0), 2)

        slope_stale     = pd.notna(slope_ratio) and (slope_ratio > 1.2 or slope_ratio < 0.8)
        intercept_stale = pd.notna(intercept_gap) and intercept_gap > 5
        tol_narrow      = pd.notna(var3) and resid_std > var3
        if slope_stale:
            action = "STALE_CONFIG"
        elif intercept_stale:
            action = "STALE_INTERCEPT"
        elif tol_narrow:
            action = "WIDEN_TOLERANCE"
        else:
            action = "OK"

        records.append({
            "test_id":       tid,
            "metric_type":   g["metric_type"].iloc[0],
            "asset_type":    g["asset_type"].iloc[0],
            "tenant_id":     g["tenant_id"].iloc[0],
            "runs":          len(g),
            "incident_rate": (g["is_passed_binary"] == 0).mean(),
            "r_squared":     round(r_val**2, 3),
            "cur_var1":      round(var1, 4),
            "sug_var1":      round(slope, 4),
            "slope_ratio":   round(slope_ratio, 2) if pd.notna(slope_ratio) else None,
            "cur_var2":      round(var2, 1),
            "sug_var2":      round(intercept, 1),
            "intercept_gap": round(intercept_gap, 1) if pd.notna(intercept_gap) else None,
            "cur_var3":      round(var3, 2) if pd.notna(var3) else None,
            "sug_var3":      sug_var3,
            "resid_std":     round(resid_std, 2),
            "action":        action,
        })

    if not records:
        return pd.DataFrame()
    sc = pd.DataFrame(records)
    sc = sc[sc["action"] != "OK"].sort_values("incident_rate", ascending=False)
    sc["incident_rate"] = sc["incident_rate"].map(lambda x: f"{x:.1%}")
    sc["slope_ratio"]   = sc["slope_ratio"].map(lambda x: f"{x:.2f}×" if pd.notna(x) else "")
    sc["intercept_gap"] = sc["intercept_gap"].map(lambda x: f"{x:.1f} σ" if pd.notna(x) else "")
    return sc


@app.callback(
    Output("scorecard-content","children"),
    Input("sc-tabs","value"),
    Input("filter-store","data"),
)
def update_scorecard(tab, data):
    df = apply_filters(data)

    if tab == "range":
        sc, color_acc = build_range_scorecard(df), TEAL
    elif tab == "pctdiff":
        sc, color_acc = build_pctdiff_scorecard(df), BLUE
    elif tab == "trend":
        sc, color_acc = build_trend_scorecard(df), GREEN
    else:
        sc, color_acc = build_const_scorecard(df), ORANGE

    if sc.empty:
        return html.Div("No flagged tests in current filter.",
                        style={"color":MUTED,"padding":"20px","textAlign":"center"})

    action_counts = sc["action"].value_counts().to_dict()

    def badge_color(k):
        if "DISABLE" in k:   return RED,    RED+"22"
        if "RECALIB" in k:   return ORANGE, ORANGE+"22"
        if "WIDEN" in k:     return TEAL2,  TEAL2+"22"
        if "INCREASE" in k:  return BLUE,   BLUE+"22"
        return ORANGE, ORANGE+"22"

    badges = html.Div([
        html.Span(f"{v}× {k}", style={
            "display":"inline-block","marginRight":"8px","marginBottom":"8px",
            "padding":"3px 10px","borderRadius":"12px","fontSize":"11px","fontWeight":"600",
            "color": badge_color(k)[0], "background": badge_color(k)[1],
        }) for k, v in action_counts.items()
    ], style={"marginBottom":"12px"})

    tbl = dash_table.DataTable(
        id="sc-table",
        data=sc.to_dict("records"),
        columns=[{"name": _col_name(c), "id": c} for c in sc.columns],
        **TABLE_STYLE,
    )
    return html.Div([badges, _action_legend_panel(), tbl])


@app.callback(
    Output("dd-test-id", "value"),
    Input("sc-table", "active_cell"),
    State("sc-table", "data"),
    prevent_initial_call=True,
)
def prefill_from_scorecard(active_cell, data):
    if not active_cell or not data:
        return dash.no_update
    if active_cell.get("column_id") != "test_def_id":
        return dash.no_update
    row = data[active_cell["row"]]
    return row.get("test_def_id")


# ── Deep-dive callback ─────────────────────────────────────────────────────────
def _meta_chip(label, value, color=None):
    return html.Div([
        html.Span(label, style={"fontSize":"10px","fontWeight":"600","color":MUTED,
                                "textTransform":"uppercase","letterSpacing":"0.04em",
                                "display":"block","marginBottom":"2px"}),
        html.Span(str(value), style={"fontSize":"13px","fontWeight":"600",
                                     "color": color or TEXT}),
    ], style={
        "background":BG,"borderRadius":"6px","padding":"8px 14px",
        "border":f"1px solid {BORDER}","marginRight":"8px","marginBottom":"8px",
    })

def _stat_box(label, value):
    return html.Div([
        html.Div(label, style={"fontSize":"11px","color":MUTED,"marginBottom":"2px"}),
        html.Div(value, style={"fontSize":"14px","fontWeight":"600","color":TEXT}),
    ], style={"marginRight":"24px"})

def _dots_on_fig(fig, passed, failed, severe):
    """Add pass / incident / severe-breach scatter traces to a figure."""
    fig.add_trace(go.Scatter(
        x=passed["app_pit"], y=passed["run_value"], mode="markers", name="Pass",
        marker=dict(color=GREEN, size=6, opacity=0.75),
        hovertemplate="Pass<br>%{x|%Y-%m-%d %H:%M}<br>value: %{y:,.4g}<extra></extra>",
    ))
    if not failed.empty:
        fig.add_trace(go.Scatter(
            x=failed["app_pit"], y=failed["run_value"], mode="markers", name="Incident",
            marker=dict(color=RED, size=9, symbol="triangle-up", opacity=0.85),
            hovertemplate="Incident<br>%{x|%Y-%m-%d %H:%M}<br>value: %{y:,.4g}<extra></extra>",
        ))
    if not severe.empty:
        fig.add_trace(go.Scatter(
            x=severe["app_pit"], y=severe["run_value"], mode="markers", name="Severe breach",
            marker=dict(color=ORANGE, size=18, symbol="circle-open",
                        line=dict(width=2, color=ORANGE)),
            hovertemplate="Severe breach<br>%{x|%Y-%m-%d %H:%M}<br>value: %{y:,.4g}<extra></extra>",
        ))

def _fig_layout(fig, height=380, title=None):
    fig.update_layout(
        **LAYOUT_BASE, margin=dict(t=30 if title else 20, b=50, l=60, r=20), height=height,
        title=dict(text=title, font=dict(size=13, color=MUTED), x=0) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(title="Data timestamp (app_pit)", gridcolor="#F0F1F5", linecolor=BORDER),
        yaxis=dict(title="run_value", gridcolor="#F0F1F5", linecolor=BORDER),
    )


def _runs_for_recalibration_suggestion(
    df: pd.DataFrame,
    *,
    value_col: str = "run_value",
    time_col: str = "app_pit",
    last_n: int = 300,
) -> pd.DataFrame:
    """Last ``last_n`` runs by ``time_col`` (or all rows if fewer), used to train recalibration suggestions."""
    if df.empty or value_col not in df.columns:
        return df
    d = df.sort_values(time_col)
    return d.tail(min(last_n, len(d))).copy()


def _range_suggestion_bounds(rv: pd.Series) -> tuple[float, float]:
    """
    Suggested [lower, upper] from run_value series, preferring p10/p90 but avoiding a zero-width
    band (p10 == p90). Widens quantile span, then min–max, then median ± epsilon.
    """
    s = rv.dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    quads = [(0.10, 0.90), (0.05, 0.95), (0.01, 0.99), (0.0, 1.0)]
    for lo_q, hi_q in quads:
        lo = float(s.quantile(lo_q))
        hi = float(s.quantile(hi_q))
        if hi > lo:
            return lo, hi
    span = float(s.max() - s.min())
    if span > 0:
        return float(s.min()), float(s.max())
    v = float(s.median())
    eps = max(abs(v) * 1e-6, 1e-9)
    return v - eps, v + eps


SUGGESTION_TRAIN_NOTE = (
    "Training set: the last 300 runs by time (or all runs if fewer). Suggested metrics use that window."
)

def _suggestion_box(action, lines, note=None):
    """Styled recalibration suggestion panel."""
    ac_color = RED if "DISABLE" in action else ORANGE if any(x in action for x in ("RECALIB","STALE","INVESTIGATE")) else TEAL2
    return html.Div([
        html.Div([
            html.Span("Recalibration Suggestion", style={
                "fontWeight":"700","fontSize":"13px","color":TEXT,"marginRight":"10px",
            }),
            html.Span(action, style={
                "fontSize":"11px","fontWeight":"700","color":ac_color,
                "background":ac_color+"18","padding":"3px 10px","borderRadius":"10px",
            }),
        ], style={"marginBottom":"10px","display":"flex","alignItems":"center"}),
        html.Div([
            html.Div([
                html.Span(label, style={"fontSize":"11px","color":MUTED,"width":"160px","display":"inline-block"}),
                html.Span(cur,   style={"fontSize":"13px","fontWeight":"600","color":TEXT,"marginRight":"12px"}),
                html.Span("→",   style={"color":MUTED,"marginRight":"12px"}),
                html.Span(sug,   style={"fontSize":"13px","fontWeight":"700","color":TEAL2}),
            ], style={"marginBottom":"5px"})
            for label, cur, sug in lines
        ]),
        html.Div(note, style={"fontSize":"11px","color":MUTED,"marginTop":"8px","fontStyle":"italic"}) if note else html.Div(),
    ], style={
        "background":BG,"border":f"1px solid {BORDER}","borderLeft":f"3px solid {ac_color}",
        "borderRadius":"6px","padding":"14px 18px","marginTop":"14px","marginBottom":"4px",
    })


@app.callback(
    Output("dd-meta",       "children"),
    Output("dd-chart-wrap", "children"),
    Output("dd-error",      "children"),
    Input("dd-btn",         "n_clicks"),
    Input("dd-test-id",     "n_submit"),
    State("dd-test-id",     "value"),
    prevent_initial_call=True,
)
def update_deep_dive(_clicks, _submit, test_id_val):
    empty = (html.Div(), html.Div(), "")
    if test_id_val is None:
        return empty

    tid = str(int(test_id_val))
    sub = DF_RAW[DF_RAW["test_def_id"].astype(str) == tid].copy()
    if sub.empty:
        sub = DF_RAW[DF_RAW["test_id"].astype(str) == tid].copy()
    if sub.empty:
        return html.Div(), html.Div(), f"No runs found for Test ID {tid}"

    sub = sub.sort_values("app_pit").copy()
    test_type = sub["test_type"].iloc[0]
    meta      = sub.iloc[0]
    n_runs    = len(sub)
    n_inc     = int(sub["failed"].sum())
    inc_rate  = n_inc / n_runs
    passed    = sub[sub["is_passed_binary"] == 1]
    failed    = sub[sub["is_passed_binary"] == 0]
    p10_rv    = float(sub["run_value"].quantile(0.10))
    med_rv    = float(sub["run_value"].median())
    p90_rv    = float(sub["run_value"].quantile(0.90))
    sub_sug   = _runs_for_recalibration_suggestion(sub)

    # ── Metadata row ──────────────────────────────────────────────────────
    type_color = TYPE_COLORS.get(test_type, TEAL)
    meta_row = html.Div([
        _meta_chip("Test ID",     tid),
        _meta_chip("Test Type",   test_type, type_color),
        _meta_chip("Tenant",      f"T-{meta.get('tenant_id','')}"),
        _meta_chip("Env",         meta.get("env_id","")),
        _meta_chip("Metric Type", meta.get("metric_type","")),
        _meta_chip("Asset Type",  meta.get("asset_type","")),
        _meta_chip("Asset Name",  str(meta.get("asset_name",""))[:50]),
    ], style={"display":"flex","flexWrap":"wrap"})

    # ── Main figure + per-type logic ──────────────────────────────────────
    fig      = go.Figure()
    fig_sim  = None          # simulated-bounds figure, built per type
    sug_box  = html.Div()   # suggestion panel
    bounds_info = ""
    severe   = pd.DataFrame()

    # shared connector line
    fig.add_trace(go.Scatter(
        x=sub["app_pit"], y=sub["run_value"],
        mode="lines", line=dict(color=TEAL3, width=0.8), opacity=0.5,
        showlegend=False, hoverinfo="skip",
    ))

    # ── RANGE ─────────────────────────────────────────────────────────────
    if test_type == "Range":
        lower_med = float(sub["lower_bound"].median())
        upper_med = float(sub["upper_bound"].median())
        bw        = max(upper_med - lower_med, 1e-9)
        severe    = failed[
            (failed["run_value"] < lower_med - 3 * bw) |
            (failed["run_value"] > upper_med + 3 * bw)
        ]
        fig.add_hrect(y0=lower_med, y1=upper_med, fillcolor=GREEN, opacity=0.08, line_width=0)
        fig.add_hline(y=upper_med, line=dict(color=RED,  width=1.5, dash="dash"),
                      annotation_text=f"Max ({upper_med:,.3g})",
                      annotation_position="top right", annotation_font_size=10)
        fig.add_hline(y=lower_med, line=dict(color=BLUE, width=1.2, dash="dot"),
                      annotation_text=f"Min ({lower_med:,.3g})",
                      annotation_position="bottom right", annotation_font_size=10)
        bounds_info = f"min={lower_med:,.3g}  max={upper_med:,.3g}"

        # suggestion: p10/p90 on last-300 window, with non-degenerate width
        rv_sug = sub_sug["run_value"].dropna()
        if len(rv_sug):
            sug_lower, sug_upper = _range_suggestion_bounds(rv_sug)
        else:
            sug_lower, sug_upper = _range_suggestion_bounds(sub["run_value"].dropna())
        if not np.isfinite(sug_lower) or not np.isfinite(sug_upper):
            sug_lower, sug_upper = p10_rv, p90_rv
            if sug_upper <= sug_lower:
                sug_lower, sug_upper = _range_suggestion_bounds(sub["run_value"].dropna())
        inc_rate_sim = float(((sub["run_value"] < sug_lower) | (sub["run_value"] > sug_upper)).mean())
        action = ("CANDIDATE_DISABLE" if inc_rate > 0.8
                  else "RECALIBRATE" if inc_rate > 0.3 else "WIDEN_SHIFT_BOUNDS")
        sug_box = _suggestion_box(action, [
            ("Lower bound (var1 min)", f"{lower_med:,.3g}", f"{sug_lower:,.3g}  (suggested)"),
            ("Upper bound (var2 max)", f"{upper_med:,.3g}", f"{sug_upper:,.3g}  (suggested)"),
            ("Projected incident rate", fmt_pct(inc_rate), fmt_pct(inc_rate_sim)),
        ], note=(
            f"Suggested bounds use p10–p90 of run_values in the training window when that gives a positive width; "
            f"otherwise quantiles are widened (through min–max) so lower ≠ upper. {SUGGESTION_TRAIN_NOTE} "
            f"Review against domain SLA before applying."
        ))

        # simulated figure
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sub["app_pit"], y=sub["run_value"],
            mode="lines", line=dict(color=TEAL3, width=0.8), opacity=0.5,
            showlegend=False, hoverinfo="skip",
        ))
        fig_sim.add_hrect(y0=lower_med, y1=upper_med, fillcolor="#9CA3AF", opacity=0.08, line_width=0)
        fig_sim.add_hline(y=upper_med, line=dict(color="#9CA3AF", width=1, dash="dash"),
                          annotation_text=f"Current max ({upper_med:,.3g})",
                          annotation_position="top right", annotation_font_size=10)
        fig_sim.add_hline(y=lower_med, line=dict(color="#9CA3AF", width=1, dash="dot"),
                          annotation_position="bottom right", annotation_font_size=10)
        fig_sim.add_hrect(y0=sug_lower, y1=sug_upper, fillcolor=TEAL, opacity=0.10, line_width=0)
        fig_sim.add_hline(y=sug_upper, line=dict(color=TEAL, width=2, dash="dash"),
                          annotation_text=f"Suggested max ({sug_upper:,.3g})",
                          annotation_position="top left", annotation_font_size=10)
        fig_sim.add_hline(y=sug_lower, line=dict(color=TEAL, width=1.5, dash="dot"),
                          annotation_text=f"Suggested min ({sug_lower:,.3g})",
                          annotation_position="bottom left", annotation_font_size=10)
        sim_passed = sub[sub["run_value"].between(sug_lower, sug_upper)]
        sim_failed = sub[~sub["run_value"].between(sug_lower, sug_upper)]
        _dots_on_fig(fig_sim, sim_passed, sim_failed, pd.DataFrame())

    # ── CONST ─────────────────────────────────────────────────────────────
    elif test_type == "Const":
        expected  = float(sub["var1"].median())
        severe    = failed[abs(failed["run_value"] - expected) > 3 * max(abs(expected), 1)]
        fig.add_hline(y=expected, line=dict(color=GREEN, width=2),
                      annotation_text=f"Expected = {expected:,.4g}",
                      annotation_position="top right", annotation_font_size=10)
        bounds_info = f"expected constant = {expected:,.6g}"

        fail_med = float(failed["run_value"].median()) if not failed.empty else np.nan
        fail_cv  = (float(failed["run_value"].std()) / abs(fail_med)
                    if not failed.empty and fail_med != 0 else np.nan)
        rv_sug_c = sub_sug["run_value"].dropna()
        sug_const = round(
            float(rv_sug_c.quantile(0.95)) if len(rv_sug_c) else float(sub["run_value"].quantile(0.95)),
            4,
        )
        action = ("CANDIDATE_DISABLE" if inc_rate > 0.5
                  else "INVESTIGATE_ZERO_CONST" if expected == 0 else "REVIEW_THRESHOLD")
        lines = [("Expected constant (var1)", f"{expected:,.4g}", f"{sug_const:,.4g}  (p95 of runs)")]
        if pd.notna(fail_med):
            lines.append(("Incident median value", f"{fail_med:,.4g}", "—"))
        sug_box = _suggestion_box(action, lines,
            note=f"For zero-assertion tests (var1=0), p95 gives the threshold that would pass 95% of the training subset. {SUGGESTION_TRAIN_NOTE}")

        # simulated
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sub["app_pit"], y=sub["run_value"],
            mode="lines", line=dict(color=TEAL3, width=0.8), opacity=0.5,
            showlegend=False, hoverinfo="skip",
        ))
        fig_sim.add_hline(y=expected, line=dict(color="#9CA3AF", width=1, dash="dash"),
                          annotation_text=f"Current ({expected:,.4g})",
                          annotation_position="top right", annotation_font_size=10)
        fig_sim.add_hline(y=sug_const, line=dict(color=TEAL, width=2),
                          annotation_text=f"Suggested ({sug_const:,.4g})",
                          annotation_position="top left", annotation_font_size=10)
        sim_passed = sub[sub["run_value"] <= sug_const]
        sim_failed = sub[sub["run_value"] > sug_const]
        _dots_on_fig(fig_sim, sim_passed, sim_failed, pd.DataFrame())

    # ── PCTDIFF ───────────────────────────────────────────────────────────
    elif test_type == "PctDiff":
        sub_v   = sub.dropna(subset=["lower_bound","upper_bound"])
        tol     = float(sub["var1"].median())
        win     = float(sub["var2"].median())
        baseline = (sub_v["lower_bound"] + sub_v["upper_bound"]) / 2
        deviation = ((sub_v["run_value"] - baseline) / baseline.replace(0, np.nan)).abs() * 100
        sub_v_sug = _runs_for_recalibration_suggestion(sub_v)
        baseline_s = (sub_v_sug["lower_bound"] + sub_v_sug["upper_bound"]) / 2
        deviation_s = ((sub_v_sug["run_value"] - baseline_s) / baseline_s.replace(0, np.nan)).abs() * 100
        p95_dev = float(deviation_s.quantile(0.95)) if not deviation_s.dropna().empty else float(deviation.quantile(0.95))
        sug_tol = math.ceil(p95_dev * 1.1 * 10) / 10

        def _band_traces(fig_, lb, ub, color, name, opacity=0.10):
            fig_.add_trace(go.Scatter(
                x=pd.concat([sub_v["app_pit"], sub_v["app_pit"].iloc[::-1]]),
                y=pd.concat([ub, lb.iloc[::-1]]),
                fill="toself", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:],16)},{opacity})",
                line=dict(width=0), showlegend=True, name=f"{name} zone", hoverinfo="skip",
            ))
            fig_.add_trace(go.Scatter(
                x=sub_v["app_pit"], y=ub, mode="lines",
                line=dict(color=color, width=1.2, dash="dash"), name=f"{name} upper",
                hovertemplate=f"{name} upper: %{{y:,.3g}}<extra></extra>",
            ))
            fig_.add_trace(go.Scatter(
                x=sub_v["app_pit"], y=lb, mode="lines",
                line=dict(color=color, width=1, dash="dot"), name=f"{name} lower",
                hovertemplate=f"{name} lower: %{{y:,.3g}}<extra></extra>",
            ))

        _band_traces(fig, sub_v["lower_bound"], sub_v["upper_bound"], GREEN, "Current")
        bw     = (sub_v["upper_bound"] - sub_v["lower_bound"]).median()
        severe = failed[
            (failed["run_value"] < failed["lower_bound"] - 3 * bw) |
            (failed["run_value"] > failed["upper_bound"] + 3 * bw)
        ] if bw > 0 else pd.DataFrame()
        bounds_info = f"tolerance ±{tol:.0f}%  window size {win:.0f}"

        sug_lb = baseline * (1 - sug_tol / 100)
        sug_ub = baseline * (1 + sug_tol / 100)
        inc_rate_sim = float(((sub_v["run_value"] < sug_lb) | (sub_v["run_value"] > sug_ub)).mean())
        action = ("CANDIDATE_DISABLE" if inc_rate > 0.8
                  else "WIDEN_TOLERANCE" if p95_dev > tol
                  else "INCREASE_WINDOW" if win < 5 else "NOISE_GENERATOR")
        sug_box = _suggestion_box(action, [
            ("Tolerance var1 (±%)",    f"±{tol:.0f}%",  f"±{sug_tol:.1f}%  (p95 dev × 1.1)"),
            ("Window size var2",       f"{win:.0f} runs","—"),
            ("p95 deviation observed", f"{p95_dev:.1f}%","—"),
            ("Projected incident rate",fmt_pct(inc_rate), fmt_pct(inc_rate_sim)),
        ], note=f"Suggested tolerance is 110% of the p95 observed deviation on the training subset so most of that subset would pass. {SUGGESTION_TRAIN_NOTE}")

        # simulated
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sub_v["app_pit"], y=sub_v["run_value"],
            mode="lines", line=dict(color=TEAL3, width=0.8), opacity=0.5,
            showlegend=False, hoverinfo="skip",
        ))
        _band_traces(fig_sim, sub_v["lower_bound"], sub_v["upper_bound"], "#9CA3AF", "Current", 0.06)
        _band_traces(fig_sim, sug_lb, sug_ub, TEAL, "Suggested", 0.10)
        sim_passed = sub_v[sub_v["run_value"].between(sug_lb, sug_ub)]
        sim_failed = sub_v[~sub_v["run_value"].between(sug_lb, sug_ub)]
        _dots_on_fig(fig_sim, sim_passed, sim_failed, pd.DataFrame())

    # ── TREND ─────────────────────────────────────────────────────────────
    elif test_type == "Trend":
        sub_v = sub.dropna(subset=["lower_bound","upper_bound","app_pit","run_value"]).copy()
        sub_v["app_pit"] = pd.to_datetime(sub_v["app_pit"], utc=True, errors="coerce")
        sub_v = sub_v.dropna(subset=["app_pit"])
        t0    = sub_v["app_pit"].min()
        sub_v["t_hr"] = (sub_v["app_pit"] - t0).dt.total_seconds() / 3600

        sub_v_sug = _runs_for_recalibration_suggestion(sub_v)
        if len(sub_v_sug) < 2:
            sub_v_sug = sub_v
        slope, intercept, r_val, _, _ = scipy_stats.linregress(
            sub_v_sug["t_hr"], sub_v_sug["run_value"]
        )
        fitted    = intercept + slope * sub_v["t_hr"]
        abs_resid = (sub_v["run_value"] - fitted).abs()
        fitted_sug = intercept + slope * sub_v_sug["t_hr"]
        abs_resid_sug = (sub_v_sug["run_value"] - fitted_sug).abs()
        resid_p97 = float(abs_resid_sug.quantile(0.97)) if len(abs_resid_sug) else float(abs_resid.quantile(0.97))
        var1 = float(sub["var1"].median())
        var2 = float(sub["var2"].median())
        var3 = float(sub["var3"].median()) if "var3" in sub.columns else np.nan
        sug_v3 = max(resid_p97, var3 if pd.notna(var3) else 0)

        bw     = (sub_v["upper_bound"] - sub_v["lower_bound"]).median()
        severe = failed[
            (failed["run_value"] < failed["lower_bound"] - 3 * bw) |
            (failed["run_value"] > failed["upper_bound"] + 3 * bw)
        ] if bw > 0 else pd.DataFrame()

        # current band
        fig.add_trace(go.Scatter(
            x=pd.concat([sub_v["app_pit"], sub_v["app_pit"].iloc[::-1]]),
            y=pd.concat([sub_v["upper_bound"], sub_v["lower_bound"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(16,185,129,0.08)",
            line=dict(width=0), showlegend=True, name="Current band", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=sub_v["app_pit"], y=sub_v["upper_bound"], mode="lines",
            line=dict(color=GREEN, width=1.2, dash="dash"), name="Current upper"))
        fig.add_trace(go.Scatter(x=sub_v["app_pit"], y=sub_v["lower_bound"], mode="lines",
            line=dict(color=GREEN, width=1, dash="dot"), name="Current lower"))

        slope_ratio   = slope / var1 if var1 != 0 else np.nan
        intercept_gap = abs(intercept - var2) / var3 if (pd.notna(var3) and var3 > 0) else np.nan
        action = ("STALE_CONFIG"    if pd.notna(slope_ratio) and (slope_ratio > 1.2 or slope_ratio < 0.8)
                  else "STALE_INTERCEPT" if pd.notna(intercept_gap) and intercept_gap > 5
                  else "WIDEN_TOLERANCE" if (pd.notna(var3) and float(abs_resid.std()) > var3)
                  else "OK")
        bounds_info = (f"slope={var1:.4f}/hr  intercept={var2:.1f}  "
                       f"tolerance σ={var3:.2f}  r²={r_val**2:.3f}")

        sug_center = intercept + slope * sub_v["t_hr"]
        inc_rate_sim = float(((sub_v["run_value"] < sug_center - sug_v3) |
                               (sub_v["run_value"] > sug_center + sug_v3)).mean())

        sr_str = f"{slope_ratio:.2f}×" if pd.notna(slope_ratio) else "n/a"
        ig_str = f"{intercept_gap:.1f} σ" if pd.notna(intercept_gap) else "n/a"
        sug_box = _suggestion_box(action if action != "OK" else "WIDEN_TOLERANCE", [
            ("Slope var1 (/hr)",         f"{var1:.4f}", f"{slope:.4f}  (OLS fit, ratio={sr_str})"),
            ("Intercept var2",           f"{var2:.1f}", f"{intercept:.1f}  (gap={ig_str})"),
            ("Tolerance var3 (σ)",       f"{var3:.2f}" if pd.notna(var3) else "n/a",
                                         f"{sug_v3:.2f}  (p97 residual)"),
            ("Projected incident rate",  fmt_pct(inc_rate), fmt_pct(inc_rate_sim)),
        ], note=f"OLS fit on {len(sub_v_sug)} runs (training subset), r²={r_val**2:.3f}. Suggested tolerance covers p97 of residuals on that subset. {SUGGESTION_TRAIN_NOTE}")

        # simulated
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sub_v["app_pit"], y=sub_v["run_value"],
            mode="lines", line=dict(color=TEAL3, width=0.8), opacity=0.5,
            showlegend=False, hoverinfo="skip",
        ))
        # current band (gray)
        fig_sim.add_trace(go.Scatter(
            x=pd.concat([sub_v["app_pit"], sub_v["app_pit"].iloc[::-1]]),
            y=pd.concat([sub_v["upper_bound"], sub_v["lower_bound"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(156,163,175,0.10)",
            line=dict(width=0), showlegend=True, name="Current band", hoverinfo="skip",
        ))
        fig_sim.add_trace(go.Scatter(x=sub_v["app_pit"], y=sub_v["upper_bound"], mode="lines",
            line=dict(color="#9CA3AF", width=1, dash="dash"), name="Current upper"))
        fig_sim.add_trace(go.Scatter(x=sub_v["app_pit"], y=sub_v["lower_bound"], mode="lines",
            line=dict(color="#9CA3AF", width=1, dash="dot"), name="Current lower"))
        # suggested band (teal)
        sug_ub = sug_center + sug_v3
        sug_lb = sug_center - sug_v3
        fig_sim.add_trace(go.Scatter(
            x=pd.concat([sub_v["app_pit"], sub_v["app_pit"].iloc[::-1]]),
            y=pd.concat([sug_ub, sug_lb.iloc[::-1]]),
            fill="toself", fillcolor="rgba(0,196,204,0.12)",
            line=dict(width=0), showlegend=True, name="Suggested band", hoverinfo="skip",
        ))
        fig_sim.add_trace(go.Scatter(x=sub_v["app_pit"], y=sug_ub, mode="lines",
            line=dict(color=TEAL, width=1.5, dash="dash"), name="Suggested upper"))
        fig_sim.add_trace(go.Scatter(x=sub_v["app_pit"], y=sug_lb, mode="lines",
            line=dict(color=TEAL, width=1.2, dash="dot"), name="Suggested lower"))
        # OLS fitted line
        fig_sim.add_trace(go.Scatter(x=sub_v["app_pit"], y=sug_center, mode="lines",
            line=dict(color=TEAL2, width=1), name="OLS fitted line"))
        sim_failed_mask = (sub_v["run_value"] < sug_lb) | (sub_v["run_value"] > sug_ub)
        _dots_on_fig(fig_sim, sub_v[~sim_failed_mask], sub_v[sim_failed_mask], pd.DataFrame())

    else:  # fallback
        lower_med = float(sub["lower_bound"].median())
        upper_med = float(sub["upper_bound"].median())
        fig.add_hrect(y0=lower_med, y1=upper_med, fillcolor=GREEN, opacity=0.08, line_width=0)
        bounds_info = f"bounds [{lower_med:,.2g} , {upper_med:,.2g}]"

    # Add dots to main figure
    _dots_on_fig(fig, passed, failed, severe)
    _fig_layout(fig, height=380)
    _fig_layout(fig_sim, height=340, title="Simulated View — With Suggested Bounds") if fig_sim else None

    # ── Stats bar ─────────────────────────────────────────────────────────
    stats_row = html.Div([
        _stat_box("Total Runs",     f"{n_runs:,}"),
        _stat_box("Incidents",      f"{n_inc:,}"),
        _stat_box("Incident Rate",  fmt_pct(inc_rate)),
        _stat_box("Severe Breaches",f"{len(severe):,}"),
        _stat_box("p10 run_value",  f"{p10_rv:,.4g}"),
        _stat_box("Median run_value",f"{med_rv:,.4g}"),
        _stat_box("p90 run_value",  f"{p90_rv:,.4g}"),
        _stat_box("Bounds",         bounds_info),
    ], style={
        "display":"flex","flexWrap":"wrap","alignItems":"center",
        "background":BG,"borderRadius":"6px","padding":"12px 16px",
        "marginTop":"12px","border":f"1px solid {BORDER}",
    })

    chart_wrap = html.Div([
        dcc.Graph(figure=fig, config=PLOTLY_CONFIG),
        stats_row,
        sug_box,
        dcc.Graph(figure=fig_sim, config=PLOTLY_CONFIG) if fig_sim else html.Div(),
    ])

    return meta_row, chart_wrap, ""


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=8050, host="127.0.0.1")
