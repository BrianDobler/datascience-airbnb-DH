import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb · Precios",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load("../../models/xgb_pipeline_v2.pkl")
    geo_map   = joblib.load("../../models/geo_map.pkl")
    price_map = joblib.load("../../models/price_map.pkl")
    return model, geo_map, price_map

model, geo_map, price_map = load_models()
T = dict(
        bg          = "#080b12",
        bg2         = "#0d1120",
        bg3         = "#0f1420",
        border      = "#151b29",
        border2     = "#1e2535",
        text        = "#c8ccd8",
        text_strong = "#f0f2f8",
        text_muted  = "#4a5168",
        text_mid    = "#8891a8",
        accent      = "#ff385c",
        accent_soft = "#ff385c33",
        accent_med  = "#ff385c88",
        bar_inactive= "#1e2535",
        bar_inactive_border = "#2a3555",
        result_grad = "linear-gradient(135deg,#120a0d 0%,#0d1120 60%,#0a0d18 100%)",
        empty_text  = "#2a3045",
        tag_low_bg  = "#0a1a2a", tag_low_b = "#1e4a6e",  tag_low_c = "#4d9de0",
        tag_mid_bg  = "#0a1a0f", tag_mid_b = "#1e6e3a",  tag_mid_c = "#4de07a",
        tag_high_bg = "#1a0a0d", tag_high_b= "#6e1e2a",  tag_high_c= "#e04d6a",
        up_c = "#4de07a", dn_c = "#e04d6a",
        plot_bg     = "rgba(0,0,0,0)",
        grid_c      = "#0f1420",
        line_c      = "#151b29",
        tick_c      = "#8891a8",
        gauge_bg    = "#0d1120",
        gauge_step1 = "#0f1420",
        gauge_step2 = "#120a0d",
        gauge_tick  = "#2a3045",
        map_style   = "mapbox://styles/mapbox/dark-v10",
        map_col_normal = [80, 130, 200, 180],
        toggle_icon = "☀️",
        toggle_label= "Light mode",
        footer_c    = "#2a3045",
        slider_track= "#1a2030",
        select_bg   = "#0d1120",
        select_border="#1a2030",
        select_text = "#c8ccd8",
        label_c     = "#4a5168",
        hr_c        = "#151b29",
    )
# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    background-color: #080b12 !important;
    color: #c8ccd8 !important;
}

.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1400px;
}

/* ── HEADER ── */
.hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #151b29;
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #f0f2f8;
    line-height: 1;
    letter-spacing: -0.03em;
    margin: 0;
}
.hero-title span { color: #ff385c; }
.hero-sub {
    font-size: 0.85rem;
    color: #4a5168;
    margin-top: 0.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.hero-badge {
    background: #0f1420;
    border: 1px solid #1e2535;
    border-radius: 50px;
    padding: 0.5rem 1.2rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    color: #4a5168;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* ── SECTION LABELS ── */
.sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    color: #ff385c;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #151b29;
}

/* ── CARDS ── */
.card {
    background: #0d1120;
    border: 1px solid #151b29;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 1.5rem; right: 1.5rem;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff385c33, transparent);
}

/* ── METRIC CARDS ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.kpi {
    background: #0d1120;
    border: 1px solid #151b29;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi:hover { border-color: #ff385c44; }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f0f2f8;
    line-height: 1;
}
.kpi-label {
    font-size: 0.72rem;
    color: #4a5168;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}
.kpi-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

/* ── RESULT BOX ── */
.result-box {
    background: linear-gradient(135deg, #120a0d 0%, #0d1120 60%, #0a0d18 100%);
    border: 1px solid #ff385c33;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1rem;
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff385c88, transparent);
}
.result-amount {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: #ff385c;
    line-height: 1;
    letter-spacing: -0.03em;
}
.result-currency {
    font-size: 1.5rem;
    color: #ff385c88;
    vertical-align: super;
    margin-right: 0.2rem;
}
.result-night {
    font-size: 0.8rem;
    color: #4a5168;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}
.result-tag {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.08em;
}
.tag-low    { background: #0a1a2a; border: 1px solid #1e4a6e; color: #4d9de0; }
.tag-mid    { background: #0a1a0f; border: 1px solid #1e6e3a; color: #4de07a; }
.tag-high   { background: #1a0a0d; border: 1px solid #6e1e2a; color: #e04d6a; }

/* ── SLIDERS & INPUTS ── */
.stSlider > div > div > div > div { background: #ff385c !important; }
.stSlider > div > div > div       { background: #1a2030 !important; }
[data-testid="stNumberInput"] input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #0d1120 !important;
    border-color: #1a2030 !important;
    color: #c8ccd8 !important;
    border-radius: 10px !important;
}
.stSelectbox div[data-baseweb="select"] * { color: #c8ccd8 !important; }

label, .stSlider label {
    color: #4a5168 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── BUTTON ── */
.stButton > button {
    background: #ff385c;
    color: #fff;
    border: none;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: background 0.2s, transform 0.1s;
    cursor: pointer;
}
.stButton > button:hover { background: #e0304e; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── DIVIDER ── */
hr { border-color: #151b29 !important; }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── HERO HEADER ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:2rem 0 2rem 0;border-bottom:1px solid {T['border']};margin-bottom:2.5rem;">
    <div>
        <h1 style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;
            color:{T['text_strong']};line-height:1;letter-spacing:-0.03em;margin:0;">
            Precio <span style="color:{T['accent']};">Airbnb</span>
        </h1>
        <p style="font-size:0.85rem;color:{T['text_muted']};margin-top:0.5rem;
            letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0;">
            XGBoost · Geolocalización · Análisis de mercado
        </p>
    </div>
    <span style="background:{T['bg2']};border:1px solid {T['border2']};border-radius:50px;
          padding:0.5rem 1.2rem;font-family:'Syne',sans-serif;font-size:0.75rem;
          color:{T['text_muted']};text-transform:uppercase;letter-spacing:0.12em;">
        Brian Dobler
    </span>
</div>
""", unsafe_allow_html=True)

# ─── LAYOUT: 3 columnas principales ─────────────────────────────────────────────
col_props, col_metrics, col_result = st.columns([1, 1, 1], gap="large")

neighbourhoods = sorted(list(geo_map.index))

# ── COLUMNA 1: Propiedades ───────────────────────────────────────────────────────
with col_props:
    st.markdown('<div class="sec-label">01 — Propiedad</div>', unsafe_allow_html=True)

    neighbourhood = st.selectbox("📍 Barrio", neighbourhoods)
    lat = geo_map.loc[neighbourhood]["latitude"]
    lon = geo_map.loc[neighbourhood]["longitude"]

    room_type = st.selectbox("Tipo de alojamiento", [
        "Entire home/apt", "Private room", "Shared room", "Hotel room"
    ])
    property_type = st.selectbox("Tipo de propiedad", [
        "Apartment", "House", "Loft", "Condo"
    ])

    accommodates = st.slider("Huéspedes", 1, 10, 2)
    bedrooms     = st.slider("Habitaciones", 1, 10, 2)
    bathrooms    = st.slider("Baños", 1.0, 5.0, 1.0, step=0.5)

# ── COLUMNA 2: Métricas de la publicación ───────────────────────────────────────
with col_metrics:
    st.markdown('<div class="sec-label">02 — Publicación</div>', unsafe_allow_html=True)

    number_of_reviews    = st.slider("Reviews totales", 0, 500, 10)
    reviews_per_month    = st.slider("Reviews por mes", 0.0, 20.0, 1.0, step=0.1)
    review_scores_rating = st.slider("Rating (0–100)", 0.0, 100.0, 90.0, step=0.5)
    minimum_nights       = st.slider("Mínimo de noches", 1, 365, 2)
    availability_365     = st.slider("Disponibilidad anual", 0, 365, 180)

    # KPIs rápidos del barrio
    avg_barrio = float(price_map.loc[neighbourhood])
    n_listings = int(geo_map.loc[neighbourhood].get("count", 0)) if "count" in geo_map.columns else "—"

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-accent" style="background: linear-gradient(90deg,#ff385c,#ff7b6b);"></div>
            <div class="kpi-value">USD {round(avg_barrio)}</div>
            <div class="kpi-label">Promedio barrio</div>
        </div>
        <div class="kpi">
            <div class="kpi-accent" style="background: linear-gradient(90deg,#4d9de0,#6bc5f8);"></div>
            <div class="kpi-value">{accommodates}</div>
            <div class="kpi-label">Huéspedes</div>
        </div>
        <div class="kpi">
            <div class="kpi-accent" style="background: linear-gradient(90deg,#4de07a,#a0f0b0);"></div>
            <div class="kpi-value">{review_scores_rating:.0f}</div>
            <div class="kpi-label">Rating</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── COLUMNA 3: Resultado ─────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="sec-label">03 — Estimación</div>', unsafe_allow_html=True)
    predict_btn = st.button("💰 Estimar precio por noche", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame({
            "accommodates":           [accommodates],
            "bathrooms":              [bathrooms],
            "bedrooms":               [bedrooms],
            "room_type":              [room_type],
            "property_type":          [property_type],
            "neighbourhood_cleansed": [neighbourhood],
            "latitude":               [lat],
            "longitude":              [lon],
            "number_of_reviews":      [number_of_reviews],
            "reviews_per_month":      [reviews_per_month],
            "review_scores_rating":   [review_scores_rating],
            "minimum_nights":         [minimum_nights],
            "availability_365":       [availability_365],
        })

        pred = model.predict(input_df)[0]
        st.session_state["pred"]           = pred
        st.session_state["avg_barrio"]     = avg_barrio
        st.session_state["neighbourhood"]  = neighbourhood

    if "pred" in st.session_state:
        pred       = st.session_state["pred"]
        avg_barrio = st.session_state["avg_barrio"]

        diff_pct = ((pred - avg_barrio) / avg_barrio) * 100
        if pred < 1500:
            tag_class, tag_text = "tag-low",  "🔵 Zona económica / baja demanda"
        elif pred < 5000:
            tag_class, tag_text = "tag-mid",  "🟢 Precio competitivo"
        else:
            tag_class, tag_text = "tag-high", "🔴 Segmento premium"

        arrow = "▲" if diff_pct >= 0 else "▼"
        color = "#4de07a" if diff_pct >= 0 else "#e04d6a"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-night">Precio estimado por noche</div>
            <div class="result-amount">
                <span class="result-currency">USD</span>{round(pred):,}
            </div>
            <div style="margin-top:0.8rem;font-size:0.8rem;color:{color};">
                {arrow} {abs(diff_pct):.1f}% vs promedio del barrio
            </div>
            <div><span class="result-tag {tag_class}">{tag_text}</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box" style="padding:3rem 2rem;">
            <div style="font-size:2.5rem;opacity:0.2;">💡</div>
            <div style="color:#2a3045;font-family:'Syne',sans-serif;font-size:0.9rem;margin-top:0.5rem;text-transform:uppercase;letter-spacing:0.1em;">
                Completá los datos<br>y estimá el precio
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── MAPA + GRÁFICO ─────────────────────────────────────────────────────────────
st.markdown("---")
map_col, chart_col = st.columns([3, 2], gap="large")

with map_col:
    st.markdown('<div class="sec-label">04 — Geolocalización del barrio</div>', unsafe_allow_html=True)

    map_full = geo_map.reset_index()
    # normalizar nombres de columnas
    if "latitude" not in map_full.columns:
        map_full = map_full.rename(columns={"lat": "latitude", "lon": "longitude"})
    map_full = map_full.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Agregar precio promedio al mapa si es posible
    try:
        pm = price_map.reset_index()
        pm.columns = ["neighbourhood_cleansed", "avg_price"]
        map_full = map_full.merge(pm, on="neighbourhood_cleansed", how="left")
        map_full["avg_price"] = map_full["avg_price"].fillna(map_full["avg_price"].median())
        price_min = map_full["avg_price"].min()
        price_max = map_full["avg_price"].max()
        map_full["elevation"] = (
            (map_full["avg_price"] - price_min) / (price_max - price_min + 1)
        ) * 800 + 100
        map_full["avg_price"] = map_full["avg_price"].round(0).astype(int)
    except Exception:
        map_full["elevation"] = 200
        map_full["avg_price"] = 0
 
    # Marcar barrio seleccionado
    map_full["is_selected"] = map_full["neighbourhood_cleansed"] == neighbourhood

    layer_all = pdk.Layer(
        "ColumnLayer",
        data=map_full[~map_full["is_selected"]],
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=150,
        get_fill_color=[80, 130, 200, 180],
        pickable=True,
        auto_highlight=True,
    )

    layer_selected = pdk.Layer(
        "ColumnLayer",
        data=map_full[map_full["is_selected"]],
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1.5,
        radius=250,
        get_fill_color=[255, 56, 92, 230],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=13,
        pitch=55,
        bearing=-15,
    )

    tooltip={
    "html": "<b>{neighbourhood_cleansed}</b><br>Precio promedio: <b>USD {avg_price}</b>",
   
    }


    deck = pdk.Deck(
        layers=[layer_all, layer_selected],
        initial_view_state=view_state,
        map_style="road",
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)
    st.caption(f"📍 Barrio seleccionado: **{neighbourhood}** — columnas rojas indican el barrio, altura proporcional al precio promedio")

with chart_col:
    st.markdown('<div class="sec-label">05 — Comparativa de precios</div>', unsafe_allow_html=True)

    # Top barrios por precio promedio
    try:
        pm_all = price_map.reset_index()
        pm_all.columns = ["Barrio", "Precio"]
        pm_all = pm_all.sort_values("Precio", ascending=False).head(12)

        colors_bar = ["#ff385c" if b == neighbourhood else "#1e2535" for b in pm_all["Barrio"]]
        border_colors = ["#ff385c" if b == neighbourhood else "#2a3555" for b in pm_all["Barrio"]]

        fig = go.Figure(go.Bar(
            x=pm_all["Precio"],
            y=pm_all["Barrio"],
            orientation="h",
            marker=dict(
                color=colors_bar,
                line=dict(color=border_colors, width=1),
            ),
            text=[f"USD {int(p):,}" for p in pm_all["Precio"]],
            textposition="outside",
            textfont=dict(color="#4a5168", size=10),
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#4a5168", family="Instrument Sans"),
            xaxis=dict(
                gridcolor="#0f1420",
                linecolor="#151b29",
                tickfont=dict(size=10),
                title=dict(text="Precio promedio (USD)", font=dict(size=10, color="#4a5168")),
            ),
            yaxis=dict(
                gridcolor="#0f1420",
                linecolor="#151b29",
                tickfont=dict(size=10, color="#8891a8"),
                autorange="reversed",
            ),
            margin=dict(l=10, r=60, t=10, b=10),
            height=420,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Cargá el precio_map para ver la comparativa de barrios.")

    # Si hay predicción, mostrar vs promedio
    if "pred" in st.session_state:
        pred       = st.session_state["pred"]
        avg_barrio = st.session_state["avg_barrio"]

        fig2 = go.Figure()
        fig2.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=round(pred),
            delta={"reference": avg_barrio, "valueformat": ".0f",
                   "increasing": {"color": "#4de07a"},
                   "decreasing": {"color": "#e04d6a"}},
            gauge={
                "axis": {"range": [0, max(pred, avg_barrio) * 1.4],
                         "tickcolor": "#2a3045"},
                "bar":  {"color": "#ff385c", "thickness": 0.25},
                "bgcolor": "#0d1120",
                "bordercolor": "#151b29",
                "steps": [
                    {"range": [0, avg_barrio], "color": "#0f1420"},
                    {"range": [avg_barrio, max(pred, avg_barrio) * 1.4], "color": "#120a0d"},
                ],
                "threshold": {
                    "line": {"color": "#4a5168", "width": 2},
                    "thickness": 0.75,
                    "value": avg_barrio,
                },
            },
            title={"text": "Tu prop. vs. promedio barrio (USD)",
                   "font": {"size": 11, "color": "#4a5168"}},
            number={"font": {"color": "#f0f2f8", "family": "Syne", "size": 36}},
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#4a5168"),
            height=220,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #151b29;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'Syne',sans-serif;font-size:0.8rem;color:#2a3045;
                 text-transform:uppercase;letter-spacing:0.12em;">
        Digital House - Data Science Proyecto 2026 - Dobler Brian 
    </span>
    <span style="font-size:0.8rem;color:#2a3045;">
        XGBoost Pipeline v2 · Geolocalización por barrio
    </span>
</div>
""", unsafe_allow_html=True)