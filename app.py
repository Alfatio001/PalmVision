import streamlit as st
import pandas as pd
import json
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PalmVision",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL THEME & CSS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":        "#0B1120",
    "surface":   "#111827",
    "surface2":  "#1A2540",
    "border":    "#1E3A5F",
    "primary":   "#22C55E",
    "primary_d": "#16A34A",
    "accent":    "#38BDF8",
    "warning":   "#FBBF24",
    "danger":    "#F87171",
    "text":      "#F1F5F9",
    "muted":     "#64748B",
}

st.markdown(f"""
<style>
/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

/* App background */
.stApp {{ background: {PALETTE['bg']}; color: {PALETTE['text']}; font-family: 'Inter', sans-serif; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {PALETTE['surface']};
    border-right: 1px solid {PALETTE['border']};
}}
[data-testid="stSidebar"] * {{ color: {PALETTE['text']}; }}
[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    background: transparent;
    border: 1px solid {PALETTE['border']};
    color: {PALETTE['text']};
    border-radius: 8px;
    padding: 8px 14px;
    transition: all .2s;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {PALETTE['surface2']};
    border-color: {PALETTE['primary']};
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {PALETTE['surface']};
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {PALETTE['border']};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: {PALETTE['muted']};
    font-size: 14px;
    padding: 8px 18px;
    border: none;
    transition: all .2s;
}}
.stTabs [aria-selected="true"] {{
    background: {PALETTE['primary']} !important;
    color: #fff !important;
    font-weight: 600;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 24px;
}}

/* Headings */
h1 {{ color: {PALETTE['text']}; font-weight: 700; letter-spacing: -0.5px; }}
h2, h3, h4 {{ color: {PALETTE['text']}; }}

/* Dataframe */
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

/* Sidebar metric — compact variant */
[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background: #1A2540;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
    font-size: 11px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: .5px;
}}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    font-size: 20px;
    font-weight: 700;
    color: #F1F5F9;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color .2s;
}}
[data-testid="stMetric"]:hover {{ border-color: {PALETTE['primary']}; }}
[data-testid="stMetricLabel"] {{ color: {PALETTE['muted']}; font-size: 13px; }}
[data-testid="stMetricValue"] {{ color: {PALETTE['text']}; font-size: 28px; font-weight: 700; }}

/* File uploader */
[data-testid="stFileUploadDropzone"] {{
    background: {PALETTE['surface2']};
    border: 2px dashed {PALETTE['border']};
    border-radius: 14px;
}}

/* Buttons */
.stButton > button {{
    background: {PALETTE['primary']};
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    transition: all .2s;
    cursor: pointer;
}}
.stButton > button:hover {{
    background: {PALETTE['primary_d']};
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(34,197,94,.35);
}}

/* Alerts */
.stAlert {{ border-radius: 12px; }}

/* Dividers */
hr {{ border-color: {PALETTE['border']}; margin: 20px 0; }}

/* Custom classes */
.card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}}
.card-accent {{
    background: linear-gradient(135deg, {PALETTE['primary_d']}18, {PALETTE['primary']}18);
    border: 1px solid {PALETTE['primary']}40;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}}
.kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
.kpi-box {{
    flex: 1;
    min-width: 140px;
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 18px 20px;
}}
.kpi-box:hover {{ border-color: {PALETTE['primary']}; transition: border-color .2s; }}
.kpi-label {{ font-size: 12px; color: {PALETTE['muted']}; text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px; }}
.kpi-value {{ font-size: 26px; font-weight: 700; color: {PALETTE['text']}; }}
.kpi-unit  {{ font-size: 12px; color: {PALETTE['muted']}; margin-top: 2px; }}
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .4px;
}}
.badge-green  {{ background: {PALETTE['primary']}22; color: {PALETTE['primary']}; border: 1px solid {PALETTE['primary']}44; }}
.badge-yellow {{ background: {PALETTE['warning']}22; color: {PALETTE['warning']}; border: 1px solid {PALETTE['warning']}44; }}
.badge-red    {{ background: {PALETTE['danger']}22;  color: {PALETTE['danger']};  border: 1px solid {PALETTE['danger']}44;  }}
.section-title {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {PALETTE['muted']};
    margin-bottom: 14px;
}}
.hero-tag {{
    display: inline-block;
    background: {PALETTE['primary']}22;
    color: {PALETTE['primary']};
    border: 1px solid {PALETTE['primary']}44;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 14px;
    margin-bottom: 12px;
    letter-spacing: .6px;
}}
.footer {{
    text-align: center;
    color: {PALETTE['muted']};
    font-size: 12px;
    padding: 28px 0 12px;
    border-top: 1px solid {PALETTE['border']};
    margin-top: 32px;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=PALETTE["text"],
    margin=dict(l=16, r=16, t=40, b=16),
)

COLOR_MAP = {
    "Small":  PALETTE["primary"],
    "Medium": PALETTE["warning"],
    "Large":  PALETTE["danger"],
}


def card(content: str, accent: bool = False) -> None:
    cls = "card-accent" if accent else "card"
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)


def section_title(label: str) -> None:
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)


def kpi_html(items: list[dict]) -> str:
    """items = [{'label': ..., 'value': ..., 'unit': ...}]"""
    boxes = ""
    for item in items:
        unit_html = f'<div class="kpi-unit">{item["unit"]}</div>' if item.get("unit") else ""
        boxes += f"""
        <div class="kpi-box">
            <div class="kpi-label">{item['label']}</div>
            <div class="kpi-value">{item['value']}</div>
            {unit_html}
        </div>"""
    return f'<div class="kpi-row">{boxes}</div>'


def confidence_label(conf: float) -> tuple[str, str]:
    """Return (label, badge_class)."""
    if conf >= 0.80:
        return "Very High", "badge-green"
    if conf >= 0.60:
        return "High", "badge-green"
    if conf >= 0.40:
        return "Moderate", "badge-yellow"
    return "Low", "badge-red"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)


# ── Paths ────────────────────────────────────────────────────────────────────

SUMMARY_PATH = "results/summary.json"
CSV_PATH     = "data/tree_analysis_final.csv"
MODEL_PATH   = "runs/palm_yolov8n/weights/best.pt"
OUTPUT_DIR   = "output"

# ── Guard against missing files ───────────────────────────────────────────────

if not os.path.exists(SUMMARY_PATH):
    st.error("⚠️ `results/summary.json` tidak ditemukan. Jalankan pipeline deteksi terlebih dahulu.")
    st.stop()

if not os.path.exists(CSV_PATH):
    st.error("⚠️ `tree_analysis_final.csv` tidak ditemukan.")
    st.stop()

summary = load_summary(SUMMARY_PATH)
df      = load_csv(CSV_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("assets/PalmVision.png", use_container_width=True)
    st.markdown("""
    <div style="text-align:center;padding:4px 0 16px;">
        <div style="font-weight:700;font-size:18px;color:#F1F5F9;">PalmVision</div>
        <div style="font-size:11px;color:#64748B;letter-spacing:.5px;">SMART AGRICULTURE AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Status badge
    model_ok = os.path.exists(MODEL_PATH)
    status_color = PALETTE["primary"] if model_ok else PALETTE["danger"]
    status_text  = "Model Ready" if model_ok else "Model Missing"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;padding:10px 14px;
                background:{status_color}18;border:1px solid {status_color}44;
                border-radius:10px;margin-bottom:16px;">
        <div style="width:8px;height:8px;border-radius:50%;background:{status_color};
                    box-shadow:0 0 6px {status_color};"></div>
        <span style="font-size:13px;font-weight:600;color:{status_color};">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Stats summary
    section_title("Dataset Stats")
    st.metric("🌴 Detected Trees", f"{summary['Total Trees']:,}")
    st.metric("🖼️ Source Images",  str(df['Image'].nunique()))
    st.metric("🎯 Avg Confidence", f"{summary['Average Confidence']:.3f}")
    st.metric("📏 Avg Canopy",     f"{summary['Average Canopy Size']:.1f} px")

    st.divider()
    section_title("Environment")
    st.markdown(f"""
    <div style="font-size:12px;color:{PALETTE['muted']};line-height:2;">
        🧠 &nbsp;YOLOv8 Nano<br>
        🐍 &nbsp;Python 3.11<br>
        ⚡ &nbsp;CUDA · RTX 3050 4 GB<br>
        📊 &nbsp;Streamlit + Plotly
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="font-size:11px;color:{PALETTE['muted']};text-align:center;">
        PalmVision &nbsp;·&nbsp; Alfatio Sultansyah &nbsp;·&nbsp; Telkom University 2026
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_dash, tab_img, tab_analytics, tab_dataset, tab_report, tab_about = st.tabs([
    "🏠  Dashboard",
    "📷  Image Analysis",
    "📈  Analytics",
    "🗂️  Dataset",
    "📄  Report",
    "ℹ️  About",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dash:

    # Hero ────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:32px 0 8px;">
        <div class="hero-tag">🌴 OIL PALM MONITORING</div>
        <h1 style="font-size:36px;margin-bottom:8px;">
            PalmVision
        </h1>
        <p style="color:{PALETTE['muted']};font-size:15px;max-width:560px;line-height:1.6;">
            Automatic detection, counting, and canopy size estimation of oil palm trees
            from aerial imagery — powered by YOLOv8 and computer vision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # KPI row ─────────────────────────────────────────────────────────────────
    section_title("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total  = summary["Total Trees"]
    avg_c  = summary["Average Canopy Size"]
    avg_cf = summary["Average Confidence"]
    n_img  = df["Image"].nunique()

    conf_label, conf_badge = confidence_label(avg_cf)

    with col1:
        st.metric("🌴 Total Trees Detected", f"{total:,}")
    with col2:
        st.metric("📏 Avg Canopy Size", f"{avg_c:.1f} px")
    with col3:
        st.metric("🎯 Avg Confidence", f"{avg_cf:.3f}")
    with col4:
        st.metric("🖼️ Images Processed", n_img)

    st.divider()

    # Canopy distribution ─────────────────────────────────────────────────────
    section_title("Canopy Classification")
    small  = summary["Small Canopy"]
    medium = summary["Medium Canopy"]
    large  = summary["Large Canopy"]

    def pct(n):
        return f"{n/total*100:.1f}%"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="card" style="border-color:{PALETTE['primary']}55;">
            <div class="section-title" style="color:{PALETTE['primary']};">🟢 SMALL CANOPY</div>
            <div style="font-size:32px;font-weight:700;color:{PALETTE['text']};">{small:,}</div>
            <div style="font-size:12px;color:{PALETTE['muted']};margin-top:4px;">
                {pct(small)} of total &nbsp;·&nbsp; &lt; 40 px avg
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card" style="border-color:{PALETTE['warning']}55;">
            <div class="section-title" style="color:{PALETTE['warning']};">🟡 MEDIUM CANOPY</div>
            <div style="font-size:32px;font-weight:700;color:{PALETTE['text']};">{medium:,}</div>
            <div style="font-size:12px;color:{PALETTE['muted']};margin-top:4px;">
                {pct(medium)} of total &nbsp;·&nbsp; 40–80 px avg
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card" style="border-color:{PALETTE['danger']}55;">
            <div class="section-title" style="color:{PALETTE['danger']};">🔴 LARGE CANOPY</div>
            <div style="font-size:32px;font-weight:700;color:{PALETTE['text']};">{large:,}</div>
            <div style="font-size:12px;color:{PALETTE['muted']};margin-top:4px;">
                {pct(large)} of total &nbsp;·&nbsp; &gt; 80 px avg
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick charts row ────────────────────────────────────────────────────────
    section_title("Distribution Overview")
    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig = px.pie(
            df, names="Canopy_Class",
            title="Canopy Class Breakdown",
            hole=0.5,
            color="Canopy_Class",
            color_discrete_map=COLOR_MAP,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(**PLOTLY_THEME, legend_title="Class")
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        count = df["Canopy_Class"].value_counts().reset_index()
        count.columns = ["Class", "Trees"]
        fig = px.bar(
            count, x="Class", y="Trees",
            title="Trees per Class",
            color="Class", color_discrete_map=COLOR_MAP,
            text="Trees",
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_layout(**PLOTLY_THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # AI Insight box ──────────────────────────────────────────────────────────
    section_title("AI Insights")

    dominant_class = max(
        [("Small", small), ("Medium", medium), ("Large", large)],
        key=lambda x: x[1]
    )[0]

    canopy_insight = {
        "Large":  "Most trees have a <strong style=\"color:#F1F5F9;\">large canopy</strong>, indicating a <strong style=\"color:#F1F5F9;\">mature plantation</strong> with well-established growth.",
        "Medium": "Most trees have a <strong style=\"color:#F1F5F9;\">medium canopy</strong>, suggesting a <strong style=\"color:#F1F5F9;\">relatively uniform mid-stage</strong> plantation.",
        "Small":  "Most trees have a <strong style=\"color:#F1F5F9;\">small canopy</strong>, pointing to <strong style=\"color:#F1F5F9;\">younger trees</strong> or sparser vegetation.",
    }[dominant_class]

    st.markdown(f"""
    <div class="card-accent">
        <div style="font-size:18px;font-weight:700;margin-bottom:16px;color:{PALETTE['text']};">
            🧠 Automated Analysis Report
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:14px;line-height:1.8;color:{PALETTE['muted']};">
            <div>
                <strong style="color:{PALETTE['text']};">Dataset Overview</strong><br>
                • {total:,} oil palm trees detected<br>
                • {n_img} aerial images processed<br>
                • Avg canopy: {avg_c:.2f} px
            </div>
            <div>
                <strong style="color:{PALETTE['text']};">Model Performance</strong><br>
                • Confidence: {avg_cf:.3f}
                  <span class="badge {conf_badge}">{conf_label}</span><br>
                • Dominant class: {dominant_class}<br>
                • Small {pct(small)} · Med {pct(medium)} · Large {pct(large)}
            </div>
        </div>
        <hr style="border-color:{PALETTE['border']};margin:16px 0;">
        <div style="font-size:14px;color:{PALETTE['muted']};line-height:1.7;">
            {canopy_insight}
            The model's detection confidence is classified as <strong style="color:{PALETTE['text']};">{conf_label}</strong>,
            indicating consistent performance across the evaluated imagery.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Latest detection output ─────────────────────────────────────────────────
    section_title("Latest Detection Result")
    if os.path.exists(OUTPUT_DIR):
        images = sorted(f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if images:
            latest_path = os.path.join(OUTPUT_DIR, images[-1])
            st.image(latest_path, caption=f"Output: {images[-1]}", use_container_width=True)
        else:
            st.info("No detection images found in the output folder yet.")
    else:
        st.info("Output folder not found. Run Image Analysis to generate results.")

    st.markdown(f'<div class="footer">PalmVision &nbsp;·&nbsp; AI-based Oil Palm Monitoring &nbsp;·&nbsp; Alfatio Sultansyah </div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_img:

    st.markdown(f"""
    <div style="padding:16px 0 24px;">
        <h2 style="margin-bottom:6px;">📷 Image Analysis</h2>
        <p style="color:{PALETTE['muted']};font-size:14px;">
            Upload a drone/aerial image and run YOLOv8 detection to count trees and estimate canopy sizes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload area ─────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        col_orig, col_result = st.columns(2, gap="large")

        with col_orig:
            section_title("Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Filename: {uploaded.name}  |  Size: {image.size[0]}×{image.size[1]} px")

        run_btn = st.button("🚀 Run Detection", type="primary")

        if run_btn:
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model not found at `{MODEL_PATH}`. Please check the path.")
            else:
                model = load_model(MODEL_PATH)
                with st.spinner("Running YOLOv8 inference…"):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    tmp_path = tmp.name
                    tmp.close()                          # close handle BEFORE writing/reading
                    image.save(tmp_path)
                    results = model.predict(source=tmp_path, conf=0.55, verbose=False)
                    result  = results[0]
                    annotated = result.plot()

                # Delete temp file after YOLO has fully released it
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Windows may still lock it briefly; safe to ignore

                with col_result:
                    section_title("Detection Result")
                    st.image(annotated, channels="BGR", use_container_width=True)
                    st.caption(f"Confidence threshold: 0.55  |  Model: YOLOv8 Nano")

                boxes = result.boxes
                n_det = len(boxes)

                st.divider()

                if n_det > 0:
                    confs, widths, heights, canopies, areas = [], [], [], [], []
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w = x2 - x1;  h = y2 - y1
                        widths.append(w);    heights.append(h)
                        areas.append(w * h)
                        canopies.append((w + h) / 2)
                        confs.append(float(box.conf[0]))

                    avg_conf_det = sum(confs) / n_det
                    conf_lbl, conf_bdg = confidence_label(avg_conf_det)

                    section_title("Detection Summary")
                    km1, km2, km3, km4 = st.columns(4)
                    km1.metric("🌴 Trees Detected",  str(n_det))
                    km2.metric("🎯 Avg Confidence",  f"{avg_conf_det:.3f}",  delta=conf_lbl, delta_color="off")
                    km3.metric("📏 Avg Canopy Size", f"{sum(canopies)/n_det:.1f} px")
                    km4.metric("📦 Avg BBox Area",   f"{sum(areas)/n_det:.0f} px²")

                    # Per-box table
                    section_title("Bounding Box Details")
                    det_df = pd.DataFrame({
                        "Tree #": range(1, n_det + 1),
                        "Confidence": [f"{c:.3f}" for c in confs],
                        "Width (px)":  [f"{w:.1f}" for w in widths],
                        "Height (px)": [f"{h:.1f}" for h in heights],
                        "Area (px²)":  [f"{a:.0f}"  for a in areas],
                        "Canopy (px)": [f"{cs:.1f}" for cs in canopies],
                    })
                    st.dataframe(det_df, use_container_width=True, hide_index=True)

                    # Confidence distribution histogram
                    fig = px.histogram(
                        x=confs, nbins=15,
                        title="Confidence Distribution — This Image",
                        labels={"x": "Confidence", "y": "Count"},
                        color_discrete_sequence=[PALETTE["primary"]],
                    )
                    fig.update_layout(**PLOTLY_THEME)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No oil palm trees detected. Try adjusting the confidence threshold or using a clearer image.")
    else:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:48px;">
            <div style="font-size:40px;margin-bottom:12px;">🌿</div>
            <div style="font-size:16px;font-weight:600;color:{PALETTE['text']};">Upload an aerial image to get started</div>
            <div style="font-size:13px;color:{PALETTE['muted']};margin-top:6px;">Supported formats: JPG · JPEG · PNG</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analytics:

    st.markdown(f"""
    <div style="padding:16px 0 24px;">
        <h2 style="margin-bottom:6px;">📈 Analytics</h2>
        <p style="color:{PALETTE['muted']};font-size:14px;">
            Deep-dive visualisations across the full detection dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Pie + Bar ─────────────────────────────────────────────────────────
    section_title("Class Distribution")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig = px.pie(
            df, names="Canopy_Class",
            title="Canopy Class Proportion",
            hole=0.45,
            color="Canopy_Class", color_discrete_map=COLOR_MAP,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        cnt = df["Canopy_Class"].value_counts().reset_index()
        cnt.columns = ["Class", "Count"]
        fig = px.bar(
            cnt, x="Class", y="Count",
            title="Tree Count per Class",
            color="Class", color_discrete_map=COLOR_MAP,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 2: Histograms ────────────────────────────────────────────────────────
    section_title("Size & Confidence Distributions")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig = px.histogram(
            df, x="Canopy_Size", nbins=30,
            title="Canopy Size Distribution",
            color_discrete_sequence=[PALETTE["primary"]],
            labels={"Canopy_Size": "Canopy Size (px)"},
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        fig = px.histogram(
            df, x="Confidence", nbins=25,
            title="Detection Confidence Distribution",
            color_discrete_sequence=[PALETTE["accent"]],
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 3: Scatter ───────────────────────────────────────────────────────────
    section_title("Bounding Box Dimensions")
    fig = px.scatter(
        df, x="Width", y="Height",
        color="Canopy_Class",
        size="Area",
        hover_data=["Confidence", "Canopy_Size"],
        title="Width vs Height (bubble size = bounding box area)",
        color_discrete_map=COLOR_MAP,
        opacity=0.75,
    )
    fig.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 4: Width / Height / Area histograms ─────────────────────────────────
    section_title("Bounding Box Statistics")
    r4c1, r4c2, r4c3 = st.columns(3)

    with r4c1:
        fig = px.histogram(df, x="Width", nbins=30, title="BBox Width",
                           color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with r4c2:
        fig = px.histogram(df, x="Height", nbins=30, title="BBox Height",
                           color_discrete_sequence=[PALETTE["warning"]])
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with r4c3:
        fig = px.histogram(df, x="Area", nbins=35, title="BBox Area",
                           color_discrete_sequence=[PALETTE["accent"]])
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 5: Box plot ─────────────────────────────────────────────────────────
    section_title("Canopy Size per Class — Box Plot")
    fig = px.box(
        df, x="Canopy_Class", y="Canopy_Size",
        color="Canopy_Class", color_discrete_map=COLOR_MAP,
        title="Canopy Size Spread by Class",
        points="outliers",
    )
    fig.update_layout(**PLOTLY_THEME, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dataset:

    st.markdown(f"""
    <div style="padding:16px 0 24px;">
        <h2 style="margin-bottom:6px;">🗂️ Dataset</h2>
        <p style="color:{PALETTE['muted']};font-size:14px;">
            Full detection records — {len(df):,} rows × {len(df.columns)} columns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls ─────────────────────────────────────────────────────────
    section_title("Filters")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        classes = ["All"] + sorted(df["Canopy_Class"].unique().tolist())
        sel_class = st.selectbox("Canopy Class", classes)
    with fc2:
        conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.01)
    with fc3:
        sort_col = st.selectbox("Sort by", df.columns.tolist(), index=df.columns.get_loc("Confidence") if "Confidence" in df.columns else 0)

    filtered = df.copy()
    if sel_class != "All":
        filtered = filtered[filtered["Canopy_Class"] == sel_class]
    filtered = filtered[filtered["Confidence"] >= conf_min]
    filtered = filtered.sort_values(sort_col, ascending=False)

    st.markdown(f'<div style="font-size:12px;color:{PALETTE["muted"]};margin-bottom:8px;">'
                f'Showing {len(filtered):,} of {len(df):,} records</div>', unsafe_allow_html=True)

    st.dataframe(filtered, use_container_width=True, hide_index=True, height=480)

    # Download filtered CSV ───────────────────────────────────────────────────
    st.divider()
    csv_export = filtered.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_export,
        file_name="palmvision_filtered.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_report:

    st.markdown(f"""
    <div style="padding:16px 0 24px;">
        <h2 style="margin-bottom:6px;">📄 Report</h2>
        <p style="color:{PALETTE['muted']};font-size:14px;">
            Summary statistics and downloadable exports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary table ───────────────────────────────────────────────────────────
    section_title("Summary Statistics")
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Trees Detected",
            "Images Processed",
            "Average Confidence",
            "Average Canopy Size (px)",
            "Average Width (px)",
            "Average Height (px)",
            "Average Area (px²)",
            "Small Canopy Trees",
            "Medium Canopy Trees",
            "Large Canopy Trees",
        ],
        "Value": [
            summary["Total Trees"],
            df["Image"].nunique(),
            round(summary["Average Confidence"], 4),
            round(summary["Average Canopy Size"], 2),
            round(summary.get("Average Width", df["Width"].mean()), 2),
            round(summary.get("Average Height", df["Height"].mean()), 2),
            round(summary.get("Average Area", df["Area"].mean()), 2),
            summary["Small Canopy"],
            summary["Medium Canopy"],
            summary["Large Canopy"],
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # Download options ────────────────────────────────────────────────────────
    section_title("Downloads")
    dl1, dl2 = st.columns(2)

    with dl1:
        txt_file = "results/summary.txt"
        if os.path.exists(txt_file):
            with open(txt_file) as f:
                report_txt = f.read()
            st.download_button(
                "⬇️ Download TXT Report",
                report_txt,
                file_name="palmvision_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("`results/summary.txt` not found.")

    with dl2:
        csv_full = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full CSV",
            csv_full,
            file_name="tree_analysis_final.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:

    st.markdown(f"""
    <div style="padding:16px 0 24px;">
        <h2 style="margin-bottom:6px;">ℹ️ About PalmVision</h2>
    </div>
    """, unsafe_allow_html=True)

    col_logo, col_desc = st.columns([1, 2], gap="large")

    with col_logo:
        st.image("assets/PalmVision.png", use_container_width=True)

    with col_desc:
        st.markdown(f"""
        <div style="font-size:15px;line-height:1.8;color:{PALETTE['muted']};">
            <strong style="color:{PALETTE['text']};font-size:20px;">PalmVision </strong><br><br>
            PalmVision is an AI application for automatic oil palm tree detection, counting,
            and canopy size estimation from aerial drone imagery using <strong>YOLOv8</strong>.<br><br>
            It integrates computer vision, statistical analysis, and interactive visualisation
            into a single dashboard designed for plantation monitoring and agronomic insights.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        st.markdown(f"""
        <div class="card">
            <div class="section-title">🤖 AI Stack</div>
            <ul style="color:{PALETTE['muted']};font-size:13px;line-height:2;padding-left:18px;">
                <li>YOLOv8 Nano</li>
                <li>Ultralytics</li>
                <li>OpenCV</li>
                <li>Python 3.11</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with ac2:
        st.markdown(f"""
        <div class="card">
            <div class="section-title">💻 Hardware</div>
            <ul style="color:{PALETTE['muted']};font-size:13px;line-height:2;padding-left:18px;">
                <li>Intel Core i5 11th Gen</li>
                <li>NVIDIA RTX 3050 4 GB</li>
                <li>CUDA Acceleration</li>
                <li>Windows 11</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with ac3:
        st.markdown(f"""
        <div class="card">
            <div class="section-title">📊 Dashboard</div>
            <ul style="color:{PALETTE['muted']};font-size:13px;line-height:2;padding-left:18px;">
                <li>Streamlit</li>
                <li>Plotly</li>
                <li>Pandas</li>
                <li>Pillow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    section_title("Dataset")
    st.markdown(f"""
    <div class="card">
        <div style="font-size:14px;color:{PALETTE['muted']};line-height:1.8;">
            <strong style="color:{PALETTE['text']};">Oil Palm Tree Detection Dataset</strong><br>
            Source: <a href="https://universe.roboflow.com/universitas-lampung-oevdg/oil-palm-tree-gjjx1/browse?queryText=&pageSize=50&startingIndex=50&browseQuery=true" style="color:{PALETTE['accent']};">Roboflow Universe</a><br>
            Class: <span class="badge badge-green">Oil Palm Tree</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_title("Features")
    feat_cols = st.columns(2)
    features = [
        ("✅", "Oil Palm Tree Detection"),
        ("✅", "Automatic Tree Counting"),
        ("✅", "Canopy Size Estimation"),
        ("✅", "Canopy Class Classification"),
        ("✅", "Statistical Analytics"),
        ("✅", "Interactive Dashboard"),
        ("✅", "Automatic Report Generation"),
        ("✅", "CSV Export"),
    ]
    for i, (icon, feat) in enumerate(features):
        with feat_cols[i % 2]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                        background:{PALETTE['surface2']};border-radius:8px;
                        margin-bottom:8px;font-size:13px;color:{PALETTE['text']};">
                <span style="font-size:16px;">{icon}</span> {feat}
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="font-size:14px;color:{PALETTE['muted']};line-height:2;">
        <strong style="color:{PALETTE['text']};">Developer</strong><br>
        Alfatio Sultansyah<br>
        Telecommunication Engineering &nbsp;·&nbsp; Telkom University &nbsp;·&nbsp; 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="footer">PalmVision &nbsp;·&nbsp; AI-based Oil Palm Monitoring &nbsp;·&nbsp; Alfatio Sultansyah </div>', unsafe_allow_html=True)