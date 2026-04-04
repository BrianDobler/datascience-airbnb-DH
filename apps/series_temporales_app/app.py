import streamlit as st
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from datetime import timedelta
from pathlib import Path

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb · Price Forecast",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── PATHS ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent.parent
MODEL_PATH  = BASE / "models" / "lstm_model.pth"
CONFIG_PATH = BASE / "models" / "lstm_config.pkl"
DATA_PATH   = BASE / "data" / "processed" / "daily_price_processed.csv"

# ─── LSTM CLASS ─────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─── LOAD ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    cfg    = joblib.load(CONFIG_PATH)
    model  = LSTMModel(input_size=1, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return model, cfg["scaler"], cfg["seq_length"], df

model, scaler, SEQ_LEN, df = load_all()

# ─── FORECAST ───────────────────────────────────────────────────────────────────
def forecast_7(model, scaler, df, seq_len):
    scaled  = scaler.transform(df["price"].values.reshape(-1, 1))
    cur_seq = scaled[-seq_len:].copy()
    preds   = []
    for _ in range(7):
        with torch.no_grad():
            p = model(torch.tensor(cur_seq, dtype=torch.float32).unsqueeze(0)).item()
        preds.append(p)
        cur_seq = np.append(cur_seq[1:], [[p]], axis=0)
    prices     = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    last_date  = df["date"].max()
    fut_dates  = [last_date + timedelta(days=i+1) for i in range(7)]
    return fut_dates, prices

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@400;500;600;700&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #1c1e22 !important;
    color: #d4d6db !important;
}
.main .block-container {
    padding: 0 2rem 4rem 2rem;
    max-width: 100%;
}

/* ── HEADER ── */
.app-header {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #2c2e33;
    margin-bottom: 0;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.header-left {}
.app-eyebrow {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #f97316;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.6rem;
}
.app-title {
    font-family: 'Outfit', sans-serif;
    font-size: 4.5rem;
    font-weight: 700;
    color: #f0f1f3;
    line-height: 0.9;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-title em {
    font-style: normal;
    color: #f97316;
}
.app-desc {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 0.8rem;
    max-width: 480px;
    line-height: 1.5;
}
.header-right {
    text-align: right;
    padding-bottom: 0.3rem;
}
.header-author {
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #3d3f45;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}
.header-model-tag {
    display: inline-block;
    margin-top: 0.4rem;
    background: #25272c;
    border: 1px solid #2c2e33;
    border-radius: 4px;
    padding: 0.25rem 0.7rem;
    font-size: 0.7rem;
    color: #f97316;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── STAT STRIP ── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    border-bottom: 1px solid #2c2e33;
    margin-bottom: 0;
}
.stat-item {
    padding: 1.2rem 1.5rem;
    border-right: 1px solid #2c2e33;
    position: relative;
}
.stat-item:last-child { border-right: none; }
.stat-item.accent::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #f97316;
}
.stat-key {
    font-size: 0.65rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}
.stat-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0f1f3;
    line-height: 1.1;
    margin-top: 0.2rem;
}
.stat-sub {
    font-size: 0.7rem;
    color: #4b5563;
    margin-top: 0.1rem;
}
.up   { color: #34d399; }
.down { color: #f87171; }

/* ── CHART ZONE ── */
.chart-zone {
    background: #191b1f;
    border-bottom: 1px solid #2c2e33;
    padding: 1.5rem 2rem 0.5rem 2rem;
    margin: 0 -2rem;
}
.chart-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.chart-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}
.chart-legend {
    display: flex;
    gap: 1.2rem;
    align-items: center;
}
.legend-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.3rem;
}

/* ── FORECAST SECTION ── */
.forecast-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #2c2e33;
    margin-bottom: 1.5rem;
}
.forecast-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f1f3;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.forecast-subtitle {
    font-size: 0.75rem;
    color: #4b5563;
    margin-top: 0.2rem;
}

/* ── DAY CARDS ── */
.day-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 0.75rem;
    margin-bottom: 2rem;
}
.day-card {
    background: #21232a;
    border: 1px solid #2c2e33;
    border-radius: 8px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.15s, transform 0.1s;
}
.day-card:hover {
    border-color: #f97316;
    transform: translateY(-2px);
}
.day-card.is-peak {
    background: #271d0f;
    border-color: #f97316;
    border-top: 3px solid #f97316;
}
.day-card.is-low {
    background: #1f2025;
    border-color: #374151;
    border-top: 3px solid #374151;
}
.dc-dow {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.dc-date {
    font-size: 0.75rem;
    color: #374151;
    margin: 0.25rem 0;
}
.dc-price {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f1f3;
    margin-top: 0.5rem;
    line-height: 1;
}
.dc-price.peak-price { color: #f97316; }
.dc-usd {
    font-size: 0.6rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.dc-badge {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.6rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    background: #f97316;
    color: #fff;
}
.dc-badge.low-b {
    background: #2c2e33;
    color: #6b7280;
}

/* ── BUTTON ── */
.stButton > button {
    background: #f97316 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #ea6c0a !important; }

/* ── MISC ── */
hr { border-color: #2c2e33 !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stCaption {
    color: #3d3f45 !important;
    font-size: 0.7rem !important;
}
label {
    color: #4b5563 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── INSIGHT BOX ── */
.insight-box {
    background: #21232a;
    border: 1px solid #2c2e33;
    border-left: 3px solid #f97316;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
}
.insight-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    color: #f97316;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
}
.insight-text {
    font-size: 0.85rem;
    color: #9ca3af;
    line-height: 1.5;
}

/* ── FOOTER ── */
.app-footer {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #2c2e33;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-text {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    color: #2c2e33;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
</style>
""", unsafe_allow_html=True)

# ─── STATS BASE ─────────────────────────────────────────────────────────────────
precio_mean = df["price"].mean()
precio_max  = df["price"].max()
precio_min  = df["price"].min()
precio_ult  = df["price"].iloc[-1]
precio_ant  = df["price"].iloc[-2]
delta       = precio_ult - precio_ant
delta_pct   = (delta / precio_ant) * 100
delta_cls   = "up" if delta >= 0 else "down"
delta_arrow = "↑" if delta >= 0 else "↓"
vol_30      = df["price"].tail(30).std()
rango_txt   = f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}"

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div class="header-left">
        <div class="app-eyebrow">Serie temporal · Buenos Aires · Airbnb</div>
        <div class="app-title">Price<br><em>Forecast</em></div>
        <div class="app-desc">
            Modelo de red neuronal LSTM entrenado sobre precios históricos de Airbnb
            en Buenos Aires. Predice el precio promedio por noche para los próximos 7 días.
        </div>
    </div>
    <div class="header-right">
        <div class="header-author">Brian Dobler</div>
        <div class="header-model-tag">LSTM · PyTorch · seq=30</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── STAT STRIP ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-strip">
    <div class="stat-item accent">
        <div class="stat-key">Último precio</div>
        <div class="stat-val">USD {round(precio_ult):,}</div>
        <div class="stat-sub {delta_cls}">{delta_arrow} {abs(delta_pct):.1f}% vs día anterior</div>
    </div>
    <div class="stat-item">
        <div class="stat-key">Promedio histórico</div>
        <div class="stat-val">USD {round(precio_mean):,}</div>
        <div class="stat-sub">365 días</div>
    </div>
    <div class="stat-item">
        <div class="stat-key">Máximo registrado</div>
        <div class="stat-val">USD {round(precio_max):,}</div>
        <div class="stat-sub">{rango_txt}</div>
    </div>
    <div class="stat-item">
        <div class="stat-key">Mínimo registrado</div>
        <div class="stat-val">USD {round(precio_min):,}</div>
        <div class="stat-sub">{rango_txt}</div>
    </div>
    <div class="stat-item">
        <div class="stat-key">Volatilidad 30d</div>
        <div class="stat-val">USD {round(vol_30):,}</div>
        <div class="stat-sub">Desv. estándar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── CHART HISTÓRICO FULL WIDTH ──────────────────────────────────────────────────
df["ma30"] = df["price"].rolling(30).mean()
df["ma7"]  = df["price"].rolling(7).mean()

fig_h = go.Figure()

fig_h.add_trace(go.Scatter(
    x=df["date"], y=df["price"],
    mode="lines", name="Precio diario",
    line=dict(color="#f97316", width=1.2),
    fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
    hovertemplate="<b>%{x|%d %b %Y}</b><br>USD %{y:,.0f}<extra></extra>",
))
fig_h.add_trace(go.Scatter(
    x=df["date"], y=df["ma7"],
    mode="lines", name="MA 7 días",
    line=dict(color="#fbbf24", width=1.5, dash="dot"),
    hovertemplate="MA7 %{x|%d %b %Y}<br>USD %{y:,.0f}<extra></extra>",
))
fig_h.add_trace(go.Scatter(
    x=df["date"], y=df["ma30"],
    mode="lines", name="MA 30 días",
    line=dict(color="#e5e7eb", width=1.5),
    hovertemplate="MA30 %{x|%d %b %Y}<br>USD %{y:,.0f}<extra></extra>",
))

fig_h.update_layout(
    paper_bgcolor="#191b1f",
    plot_bgcolor="#191b1f",
    font=dict(color="#6b7280", family="Familjen Grotesk"),
    xaxis=dict(
        gridcolor="#22252c", linecolor="#2c2e33",
        tickfont=dict(size=10, color="#4b5563"),
        rangeslider=dict(visible=True, bgcolor="#141518",
                         bordercolor="#2c2e33", thickness=0.05),
    ),
    yaxis=dict(
        gridcolor="#22252c", linecolor="#2c2e33",
        tickfont=dict(size=10, color="#4b5563"),
        tickprefix="$", side="right",
    ),
    legend=dict(
        bgcolor="rgba(25,27,31,0.9)", bordercolor="#2c2e33", borderwidth=1,
        font=dict(size=10, color="#9ca3af"),
        orientation="h", x=0, y=1.04,
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=340,
    hovermode="x unified",
)

st.markdown('<div class="chart-zone">', unsafe_allow_html=True)
st.markdown("""
<div class="chart-toolbar">
    <span class="chart-title">Precio histórico por noche · Buenos Aires</span>
</div>
""", unsafe_allow_html=True)
st.plotly_chart(fig_h, use_container_width=True)
st.caption(f"Período: {df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')} · {len(df)} observaciones · Fuente: Airbnb Calendar")
st.markdown('</div>', unsafe_allow_html=True)

# ─── FORECAST ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="forecast-header">
    <div>
        <div class="forecast-title">Proyección — próximos 7 días</div>
        <div class="forecast-subtitle">Basada en los últimos 30 días de datos · Red neuronal LSTM</div>
    </div>
</div>
""", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("Generar predicción", use_container_width=True)

if run_btn:
    with st.spinner("Corriendo modelo..."):
        fut_dates, fut_preds = forecast_7(model, scaler, df, SEQ_LEN)
    st.session_state["fd"] = fut_dates
    st.session_state["fp"] = list(fut_preds)

if "fp" in st.session_state:
    fut_dates = st.session_state["fd"]
    fut_preds = st.session_state["fp"]
    max_p     = max(fut_preds)
    min_p     = min(fut_preds)
    days_es   = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    st.markdown("")

    # ── Day cards ────────────────────────────────────────────────────────────────
    cards_html = '<div class="day-grid">'
    for d, p in zip(fut_dates, fut_preds):
        is_peak = p == max_p
        is_low  = p == min_p
        cls     = "is-peak" if is_peak else ("is-low" if is_low else "")
        p_cls   = "peak-price" if is_peak else ""
        badge   = '<div class="dc-badge">↑ Pico</div>' if is_peak else \
                  '<div class="dc-badge low-b">↓ Valle</div>' if is_low else ""
        dow     = days_es[d.weekday()]
        cards_html += (
            f'<div class="day-card {cls}">'
            f'<div class="dc-dow">{dow}</div>'
            f'<div class="dc-date">{d.strftime("%d %b")}</div>'
            f'<div class="dc-price {p_cls}">{round(p):,}</div>'
            f'<div class="dc-usd">USD / noche</div>'
            f'{badge}'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Gráfico forecast full width ───────────────────────────────────────────────
    df_tail = df.tail(60)
    upper   = [p * 1.10 for p in fut_preds]
    lower   = [p * 0.90 for p in fut_preds]

    fig_f = go.Figure()

    # Banda incertidumbre
    fig_f.add_trace(go.Scatter(
        x=fut_dates + fut_dates[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(249,115,22,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo ±10%", hoverinfo="skip",
    ))

    # Histórico reciente
    fig_f.add_trace(go.Scatter(
        x=df_tail["date"], y=df_tail["price"],
        mode="lines", name="Histórico (60d)",
        line=dict(color="#4b5563", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>USD %{y:,.0f}<extra></extra>",
    ))

    # Conector
    fig_f.add_trace(go.Scatter(
        x=[df_tail["date"].iloc[-1], fut_dates[0]],
        y=[df_tail["price"].iloc[-1], fut_preds[0]],
        mode="lines",
        line=dict(color="#f97316", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Predicción
    fig_f.add_trace(go.Scatter(
        x=fut_dates, y=fut_preds,
        mode="lines+markers", name="Predicción LSTM",
        line=dict(color="#f97316", width=3),
        marker=dict(size=9, color="#f97316",
                    line=dict(color="#1c1e22", width=2)),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>USD %{y:,.0f}<extra></extra>",
    ))

    # Separador hoy
    today = df_tail["date"].iloc[-1]

    fig_f.add_shape(
    type="line",
    x0=today,
    x1=today,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(color="#2c2e33", width=1.5, dash="dash")
    )

    fig_f.add_annotation(
    x=today,
    y=1,
    xref="x",
    yref="paper",
    text="Hoy",
    showarrow=False,
    font=dict(color="#4b5563", size=10, family="Familjen Grotesk"),
    xanchor="left",
    yanchor="bottom"
    )

    fig_f.update_layout(
        paper_bgcolor="#191b1f",
        plot_bgcolor="#191b1f",
        font=dict(color="#6b7280", family="Familjen Grotesk"),
        xaxis=dict(gridcolor="#22252c", linecolor="#2c2e33",
                   tickfont=dict(size=10, color="#4b5563")),
        yaxis=dict(gridcolor="#22252c", linecolor="#2c2e33",
                   tickfont=dict(size=10, color="#4b5563"),
                   tickprefix="$", side="right"),
        legend=dict(
            bgcolor="rgba(25,27,31,0.9)", bordercolor="#2c2e33", borderwidth=1,
            font=dict(size=10, color="#9ca3af"),
            orientation="h", x=0, y=1.06,
        ),
        margin=dict(l=0, r=0, t=20, b=10),
        height=380,
        hovermode="x unified",
    )

    st.markdown('<div class="chart-zone">', unsafe_allow_html=True)
    st.plotly_chart(fig_f, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Insight box ──────────────────────────────────────────────────────────────
    pred_mean  = np.mean(fut_preds)
    diff_hist  = ((pred_mean - precio_mean) / precio_mean) * 100
    diff_word  = "por encima" if diff_hist >= 0 else "por debajo"
    peak_day   = fut_dates[fut_preds.index(max_p)].strftime("%A %d de %B").capitalize()
    low_day    = fut_dates[fut_preds.index(min_p)].strftime("%A %d de %B").capitalize()

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-label">Análisis de la proyección</div>
        <div class="insight-text">
            El modelo proyecta un precio promedio de <strong style="color:#f0f1f3;">USD {round(pred_mean):,}</strong>
            para los próximos 7 días, un <strong style="color:#f97316;">{abs(diff_hist):.1f}%</strong> {diff_word}
            del promedio histórico. El precio más alto se espera el <strong style="color:#f0f1f3;">{peak_day}</strong>
            (USD {round(max_p):,}) y el más bajo el <strong style="color:#f0f1f3;">{low_day}</strong>
            (USD {round(min_p):,}). Intervalo de confianza estimado en ±10%.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Modelo entrenado sobre datos Airbnb Buenos Aires 2020–2021 · LSTM PyTorch · seq_length=30 · hidden_size=50")

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span class="footer-text">Digital House · Data Science 2026 · Dobler Brian</span>
    <span class="footer-text">LSTM · PyTorch · Serie temporal · Buenos Aires</span>
</div>
""", unsafe_allow_html=True)