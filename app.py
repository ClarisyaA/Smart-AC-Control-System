"""
Smart AC Control — Soft Computing UTS Genap 2025/2026
Universitas Padjadjaran — Teknik Informatika
Dosen: Dr. Ir. Intan Nurma Yulita, M.T

Arsitektur Notebook:
  artifacts/dataset_smart_ac.csv   — Dataset 200 baris
  artifacts/metrics.json           — MSE/RMSE/MAE/R2 ketiga model
  artifacts/energy_metrics.json    — Konsumsi energi ketiga model
  artifacts/predictions.json       — Prediksi seluruh dataset
  artifacts/ga_mf.json             — Parameter MF hasil GA (trapezoid)
  artifacts/ga_history.json        — Riwayat konvergensi GA
  artifacts/anfis_mf.json          — Parameter Gaussian MF + consequents ANFIS
  artifacts/anfis_history.json     — Riwayat training loss ANFIS
  artifacts/ablation/
    ablation_results.json          — Hasil 8 konfigurasi ablation GA
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import json
import copy

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart AC Control — UNPAD",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Profesional (Tampilan Bersih, Tidak Ada Emoji Dekoratif) ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #F4F6F9; }
section.main > div { padding-top: 1rem; padding-left: 2rem; padding-right: 2rem; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #DDE1E7;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

h1, h2, h3 { color: #1A2332; letter-spacing: -0.02em; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #FFFFFF;
    border-bottom: 2px solid #DDE1E7;
    gap: 0;
    padding: 0 0.5rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: transparent;
    color: #5A6475;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.7rem 1.1rem;
    margin-bottom: -2px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1A56DB !important;
    border-bottom: 2px solid #1A56DB !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DDE1E7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetric"] label {
    color: #5A6475 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1A2332 !important;
    font-size: 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

hr { border-color: #DDE1E7 !important; margin: 1.2rem 0 !important; }

.page-header {
    background: #FFFFFF;
    border: 1px solid #DDE1E7;
    border-left: 4px solid #1A56DB;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.page-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1A2332;
    margin: 0;
    line-height: 1.3;
}
.page-header-sub {
    font-size: 0.77rem;
    color: #5A6475;
    margin: 0.2rem 0 0 0;
}
.section-label {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #5A6475;
    border-bottom: 1px solid #DDE1E7;
    padding-bottom: 0.45rem;
    margin-bottom: 0.9rem;
    margin-top: 0.4rem;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #DDE1E7;
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
}
.kpi-label {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5A6475;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    color: #1A2332;
}
.kpi-sub { font-size: 0.72rem; color: #8A93A2; margin-top: 0.2rem; }
.progress-bar-wrap {
    background: #EEF0F4;
    border-radius: 99px;
    height: 5px;
    margin: 0.55rem 0 0.2rem;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.info-box {
    background: #F0F4FF;
    border: 1px solid #C7D7FF;
    border-left: 3px solid #1A56DB;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #2D3748;
    line-height: 1.6;
    margin-bottom: 0.8rem;
}
.info-box.warn {
    background: #FFFBF0;
    border-color: #F6C000;
    border-left-color: #D97706;
    color: #3D2B00;
}
.info-box.success {
    background: #F0FFF8;
    border-color: #A0E4C0;
    border-left-color: #059669;
    color: #0C2E1C;
}
.badge {
    display: inline-block;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-ok  { background: #ECFDF5; color: #065F46; }
.badge-err { background: #FEF2F2; color: #991B1B; }
.badge-warn{ background: #FFFBF0; color: #92400E; }
.sidebar-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #1A2332;
    line-height: 1.3;
}
.sidebar-sub { font-size: 0.69rem; color: #8A93A2; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# ─── Pengaturan Style Matplotlib ──────────────────────────────────────────────
def set_mpl():
    plt.rcParams.update({
        "figure.facecolor":  "#FFFFFF",
        "axes.facecolor":    "#FAFBFC",
        "axes.edgecolor":    "#DDE1E7",
        "axes.linewidth":    0.8,
        "axes.labelcolor":   "#3D4757",
        "axes.titlecolor":   "#1A2332",
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.labelsize":    8.5,
        "xtick.color":       "#5A6475",
        "ytick.color":       "#5A6475",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "grid.color":        "#EEF0F4",
        "grid.linestyle":    "-",
        "grid.linewidth":    0.7,
        "text.color":        "#3D4757",
        "legend.facecolor":  "#FFFFFF",
        "legend.edgecolor":  "#DDE1E7",
        "legend.fontsize":   8,
        "figure.dpi":        110,
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

set_mpl()

PAL = {
    "manual":  "#1A56DB",
    "ga":      "#D97706",
    "anfis":   "#059669",
    "actual":  "#374151",
    "danger":  "#DC2626",
    "warn":    "#F59E0B",
    "ok":      "#10B981",
    "muted":   "#9CA3AF",
}

ARTIFACTS = "artifacts"

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER MF MANUAL DAN FUNGSI FIS
# (Disalin persis dari notebook — Section 3.1 & 2.1)
# ═══════════════════════════════════════════════════════════════════════════════
MANUAL_MF_PARAMS = {
    "delta_T": {
        "sangat_dingin": [-10, -10, -7, -4],
        "dingin":        [-6,  -4,  -2,  0],
        "nyaman":        [-2,   0,   0,  2],
        "panas":         [ 0,   2,   4,  6],
        "sangat_panas":  [ 4,   7,  15, 15],
    },
    "kelembaban": {
        "rendah": [30, 30, 45, 55],
        "sedang": [45, 55, 65, 75],
        "tinggi": [65, 75, 90, 90],
    },
    "kecepatan": {
        "off":    0,
        "rendah": 25,
        "sedang": 50,
        "tinggi": 75,
        "max":    100,
    },
}

# 15 Rules — Sugeno Zero-Order (dari notebook Section 3.1)
RULE_TABLE = {
    ("sangat_dingin", "rendah"): "rendah",
    ("sangat_dingin", "sedang"): "rendah",
    ("sangat_dingin", "tinggi"): "sedang",
    ("dingin",        "rendah"): "off",
    ("dingin",        "sedang"): "rendah",
    ("dingin",        "tinggi"): "sedang",
    ("nyaman",        "rendah"): "off",
    ("nyaman",        "sedang"): "off",
    ("nyaman",        "tinggi"): "rendah",
    ("panas",         "rendah"): "sedang",
    ("panas",         "sedang"): "tinggi",
    ("panas",         "tinggi"): "tinggi",
    ("sangat_panas",  "rendah"): "tinggi",
    ("sangat_panas",  "sedang"): "max",
    ("sangat_panas",  "tinggi"): "max",
}


def trapezoid_mf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    eps = 1e-12
    y = np.where((x > a) & (x <= b), (x - a) / (b - a + eps), y)
    y = np.where((x > b) & (x <= c), 1.0, y)
    y = np.where((x > c) & (x < d),  (d - x) / (d - c + eps), y)
    y = np.where(b == a, np.where(x == a, 1.0, y), y)
    return np.clip(y, 0, 1)


def manual_fis_predict_row(T_ruangan, Target_Suhu, Okupansi, Kelembaban, params=None):
    """Inferensi Manual FIS per-baris data."""
    if params is None:
        params = MANUAL_MF_PARAMS
    if Okupansi == 0:
        return 0.0
    dT = T_ruangan - Target_Suhu
    Kl = Kelembaban
    mu_dT = {k: float(trapezoid_mf(np.array([dT]), *v)[0])
              for k, v in params["delta_T"].items()}
    mu_kl = {k: float(trapezoid_mf(np.array([Kl]), *v)[0])
              for k, v in params["kelembaban"].items()}
    total_w = 0.0
    weighted = 0.0
    for (dT_lbl, kl_lbl), spd_lbl in RULE_TABLE.items():
        out_val = params["kecepatan"][spd_lbl]
        firing  = mu_dT[dT_lbl] * mu_kl[kl_lbl]
        weighted += firing * out_val
        total_w  += firing
    if total_w < 1e-12:
        return 0.0
    return float(np.clip(weighted / total_w, 0, 100))


def gaussian_mf_eval(x, center, sigma):
    """Evaluasi Gaussian MF (digunakan ANFIS)."""
    return float(np.exp(-0.5 * ((x - center) / max(sigma, 1e-4)) ** 2))


def anfis_predict_row(T_ruangan, Target_Suhu, Okupansi, Kelembaban,
                      anfis_params, scaler_dT_range, scaler_Kl_range):
    """
    Inferensi ANFIS per-baris menggunakan parameter Gaussian MF
    yang dimuat dari artifacts/anfis_mf.json.
    """
    if Okupansi == 0:
        return 0.0
    dT_raw = T_ruangan - Target_Suhu
    Kl_raw = Kelembaban

    # MinMax scaling (sesuai scaler yang difit pada training)
    dT_sc = (dT_raw - scaler_dT_range[0]) / (scaler_dT_range[1] - scaler_dT_range[0] + 1e-12)
    Kl_sc = (Kl_raw - scaler_Kl_range[0]) / (scaler_Kl_range[1] - scaler_Kl_range[0] + 1e-12)

    deltaT_mf = anfis_params["deltaT_mf"]
    kl_mf     = anfis_params["kl_mf"]
    consequents = anfis_params["consequents"]

    # Hitung firing strength semua rules
    rule_labels = [(d, k, f"{d}_{k}") for d in deltaT_mf for k in kl_mf]
    strengths = np.array([
        gaussian_mf_eval(dT_sc, deltaT_mf[d]["center"], deltaT_mf[d]["sigma"]) *
        gaussian_mf_eval(Kl_sc, kl_mf[k]["center"], kl_mf[k]["sigma"])
        for d, k, _ in rule_labels
    ])
    total = strengths.sum()
    norm_w = np.ones(len(strengths)) / len(strengths) if total < 1e-12 else strengths / total

    out_sc = sum(norm_w[i] * consequents.get(rule_labels[i][2], 0.5)
                 for i in range(len(rule_labels)))

    # Inverse scale output ke range kecepatan 0-100
    y_min, y_max = anfis_params.get("y_range", [0.0, 100.0])
    # Kalikan dengan 100 karena model ANFIS mengeluarkan output skala 0-1
    out_orig = out_sc * 100.0 
    return float(np.clip(out_orig, 0, 100))


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset():
    p = os.path.join(ARTIFACTS, "dataset_smart_ac.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_metrics():
    p = os.path.join(ARTIFACTS, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_energy_metrics():
    p = os.path.join(ARTIFACTS, "energy_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    p = os.path.join(ARTIFACTS, "predictions.json")
    if not os.path.exists(p):
        return None
    return pd.read_json(p)


@st.cache_data
def load_ga_history():
    p = os.path.join(ARTIFACTS, "ga_history.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_anfis_history():
    p = os.path.join(ARTIFACTS, "anfis_history.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_ablation():
    p = os.path.join(ARTIFACTS, "ablation", "ablation_results.json")
    if not os.path.exists(p):
        # Coba path alternatif
        p2 = os.path.join(ARTIFACTS, "ablation_results.json")
        if os.path.exists(p2):
            with open(p2) as f:
                return json.load(f)
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_ga_mf():
    p = os.path.join(ARTIFACTS, "ga_mf.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_anfis_mf():
    p = os.path.join(ARTIFACTS, "anfis_mf.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ─── Load semua data ────────────────────────────────────────────────────────
df           = load_dataset()
metrics      = load_metrics()
energy_m     = load_energy_metrics()
predictions  = load_predictions()
ga_history   = load_ga_history()
anfis_hist   = load_anfis_history()
ablation     = load_ablation()
ga_mf_raw    = load_ga_mf()
anfis_mf_raw = load_anfis_mf()

# Bangun GA MF params (trapezoid)
GA_MF_PARAMS = None
if ga_mf_raw is not None:
    GA_MF_PARAMS = copy.deepcopy(MANUAL_MF_PARAMS)
    for term, v in ga_mf_raw["delta_T"].items():
        GA_MF_PARAMS["delta_T"][term] = v
    for term, v in ga_mf_raw["kelembaban"].items():
        GA_MF_PARAMS["kelembaban"][term] = v

# Bangun rentang scaler untuk ANFIS (dari dataset)
ANFIS_SCALER_DT = [-22.0, 20.0]
ANFIS_SCALER_KL = [30.0, 90.0]
if df is not None:
    dT_all = df["Temperatur_Ruangan"] - df["Target_Suhu"]
    ANFIS_SCALER_DT = [float(dT_all.min()), float(dT_all.max())]
    ANFIS_SCALER_KL = [float(df["Kelembaban"].min()), float(df["Kelembaban"].max())]
if anfis_mf_raw is not None:
    anfis_mf_raw["y_range"] = [0.0, 1.0]  # output ANFIS adalah [0,1] normalized

ARTIFACTS_OK = os.path.isdir(ARTIFACTS)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI PEMBANTU TAMPILAN
# ═══════════════════════════════════════════════════════════════════════════════
def speed_tier(v):
    if v < 20:
        return "Rendah / Off", "#065F46", "#ECFDF5"
    elif v < 55:
        return "Sedang", "#92400E", "#FFFBEB"
    else:
        return "Tinggi", "#991B1B", "#FEF2F2"


def energy_tier(v):
    if v < 1.0:
        return "Hemat", "#065F46", "#ECFDF5"
    elif v < 2.5:
        return "Normal", "#92400E", "#FFFBEB"
    else:
        return "Boros", "#991B1B", "#FEF2F2"


P_RATED_KW = 2.5


def hitung_energi(speed_pct, kelembaban, durasi=1.0):
    """Estimasi konsumsi energi dari kecepatan kompresor (kWh)."""
    hum_factor = 1.0 + 0.1 * (kelembaban / 50.0)
    return max(0.0, (speed_pct / 100.0) * P_RATED_KW * durasi * hum_factor)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom:1rem; border-bottom:1px solid #DDE1E7; margin-bottom:1.2rem;">
        <div class="sidebar-title">Smart AC Control System</div>
        <div class="sidebar-sub">
            Soft Computing — UTS Genap 2025/2026<br>
            Universitas Padjadjaran, Teknik Informatika
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Parameter Ruangan</div>', unsafe_allow_html=True)

    T_room    = st.slider("Suhu Ruangan saat ini (°C)", 15.0, 38.0, 27.0, 0.5,
                          help="Suhu yang terukur di dalam ruangan")
    T_target  = st.slider("Suhu Target / Setpoint (°C)", 18.0, 28.0, 24.0, 0.5,
                          help="Suhu yang ingin dicapai pengguna")
    occ       = st.radio("Status Ruangan", [1, 0],
                         format_func=lambda x: "Ada Penghuni" if x else "Kosong",
                         horizontal=True,
                         help="Jika ruangan kosong, AC akan dimatikan otomatis")
    suhu_luar = st.slider("Suhu Luar Ruangan (°C)", 20.0, 40.0, 32.0, 0.5)
    kelembaban = st.slider("Kelembaban Udara (%RH)", 30.0, 90.0, 60.0, 1.0,
                           help="Semakin lembab, AC bekerja lebih keras")
    waktu_opt  = st.selectbox("Waktu", ["Pagi", "Siang", "Malam"])
    durasi_est = st.slider("Estimasi Durasi Operasi (jam)", 0.5, 4.0, 1.0, 0.25,
                           help="Berapa lama AC akan menyala untuk estimasi konsumsi energi")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Status Data</div>', unsafe_allow_html=True)

    status_items = [
        ("Dataset",        df is not None),
        ("Metrik Performa", metrics is not None),
        ("Metrik Energi",  energy_m is not None),
        ("Prediksi",       predictions is not None),
        ("GA MF",          GA_MF_PARAMS is not None),
        ("ANFIS MF",       anfis_mf_raw is not None),
        ("Riwayat GA",     ga_history is not None),
        ("Riwayat ANFIS",  anfis_hist is not None),
        ("Ablation",       ablation is not None),
    ]
    cols_s = st.columns(2)
    for i, (name, ok) in enumerate(status_items):
        cls = "badge-ok" if ok else "badge-err"
        cols_s[i % 2].markdown(
            f'<span class="badge {cls}">{name}</span><br>', unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#8A93A2;line-height:1.8">
        Dosen: Dr. Ir. Intan Nurma Yulita, M.T<br>
        Prodi S-1 Teknik Informatika FMIPA
    </div>
    """, unsafe_allow_html=True)


if not ARTIFACTS_OK:
    st.warning(
        "Folder `artifacts/` tidak ditemukan. Jalankan dulu seluruh sel notebook, "
        "lalu letakkan folder `artifacts/` di direktori yang sama dengan `app.py` ini."
    )

# ─── Header Halaman ───────────────────────────────────────────────────────────
delta_T_disp = T_room - T_target
st.markdown(f"""
<div class="page-header">
    <div class="page-header-title">Sistem Kendali Suhu Ruangan Cerdas (Smart AC Control)</div>
    <div class="page-header-sub">
        Perbandingan: Manual FIS &nbsp;|&nbsp; GA-Tuned FIS &nbsp;|&nbsp; ANFIS
        &nbsp;&mdash;&nbsp;
        Selisih Suhu (DT) = {delta_T_disp:+.1f} C &nbsp;|&nbsp;
        Kelembaban = {kelembaban:.0f}% &nbsp;|&nbsp;
        Status = {"Ada Penghuni" if occ else "Kosong"}
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS UTAMA
# ═══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_mf, tab_rules, tab_perf, tab_energy, tab_ablation = st.tabs([
    "Ringkasan Utama",
    "Fungsi Keanggotaan",
    "Rule Base",
    "Analisis Performa",
    "Konsumsi Energi",
    "Ablation Study",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — RINGKASAN UTAMA (DASHBOARD)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_dash:
    # Inferensi real-time
    fm = manual_fis_predict_row(T_room, T_target, occ, kelembaban)
    fg = manual_fis_predict_row(T_room, T_target, occ, kelembaban, GA_MF_PARAMS) \
         if GA_MF_PARAMS else None
    fa = anfis_predict_row(T_room, T_target, occ, kelembaban,
                           anfis_mf_raw, ANFIS_SCALER_DT, ANFIS_SCALER_KL) \
         if anfis_mf_raw else None

    valid_vals = [v for v in [fm, fg, fa] if v is not None]
    ensemble   = float(np.mean(valid_vals))

    # Estimasi energi per model
    em_manual = hitung_energi(fm, kelembaban, durasi_est)
    em_ga     = hitung_energi(fg, kelembaban, durasi_est) if fg is not None else None
    em_anfis  = hitung_energi(fa, kelembaban, durasi_est) if fa is not None else None
    em_ens    = hitung_energi(ensemble, kelembaban, durasi_est)

    card_data = [("Manual FIS", fm, em_manual, "Desain ahli manusia")]
    if fg is not None:
        card_data.append(("GA-Tuned FIS", fg, em_ga, "Optimasi evolusioner (GA)"))
    if fa is not None:
        card_data.append(("ANFIS", fa, em_anfis, "Adaptive Neuro-Fuzzy"))
    card_data.append(("Ensemble", ensemble, em_ens, f"Rata-rata {len(valid_vals)} model"))

    card_colors = [PAL["manual"], PAL["ga"], PAL["anfis"], PAL["actual"]]

    # ── Kartu KPI Kecepatan Kompresor ─────────────────────────────────────────
    st.markdown('<div class="section-label">Prediksi Kecepatan Kompresor (Real-Time)</div>',
                unsafe_allow_html=True)
    cols_kpi = st.columns(len(card_data))
    for col, (label, val, e_val, tag), color in zip(cols_kpi, card_data, card_colors):
        tier_lbl, tier_fg, tier_bg = speed_tier(val)
        with col:
            e_str = f"{e_val:.3f} kWh/sesi" if e_val is not None else "—"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">
                    {val:.1f}<span style="font-size:0.85rem;color:#9CA3AF;font-weight:400"> %</span>
                </div>
                <div class="progress-bar-wrap">
                    <div class="progress-bar-fill" style="width:{val}%;background:{color};opacity:0.7"></div>
                </div>
                <span style="background:{tier_bg};color:{tier_fg};
                             padding:0.15rem 0.45rem;border-radius:4px;
                             font-size:0.67rem;font-weight:600">{tier_lbl}</span>
                <div class="kpi-sub" style="margin-top:0.4rem">{tag}</div>
                <div class="kpi-sub">Est. energi: {e_str}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grafik Perbandingan & Rekomendasi ─────────────────────────────────────
    col_chart, col_rec = st.columns([3, 2])

    with col_chart:
        st.markdown('<div class="section-label">Perbandingan Prediksi Kecepatan</div>',
                    unsafe_allow_html=True)
        labels_bar = [d[0] for d in card_data]
        vals_bar   = [d[1] for d in card_data]
        b_colors   = card_colors[:len(card_data)]

        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#FFFFFF")
        bars = ax.barh(labels_bar, vals_bar,
                       color=[c + "22" for c in b_colors],
                       edgecolor=b_colors, linewidth=1.4, height=0.52)
        ax.axvline(33, color="#DDE1E7", lw=1, ls="--")
        ax.axvline(66, color="#DDE1E7", lw=1, ls="--")
        for bar, val, col in zip(bars, vals_bar, b_colors):
            ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9,
                    color=col, fontweight="600")
        ax.set_xlim(0, 115)
        ax.set_xlabel("Kecepatan Kompresor (%)")
        ax.grid(axis="x", alpha=0.4)
        ax.set_facecolor("#FAFBFC")
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_rec:
        st.markdown('<div class="section-label">Rekomendasi Sistem</div>',
                    unsafe_allow_html=True)
        tier_lbl, tier_fg, tier_bg = speed_tier(ensemble)

        if ensemble < 5:
            rec = "Suhu ruangan sudah sesuai target. AC dapat tetap standby atau dimatikan."
        elif ensemble < 25:
            rec = "Pendinginan ringan diperlukan. Kecepatan rendah sudah cukup untuk kenyamanan."
        elif ensemble < 55:
            rec = "Pendinginan sedang dibutuhkan. Pantau kondisi ruangan secara berkala."
        else:
            rec = "Pendinginan agresif diperlukan. Pastikan ventilasi ruangan baik dan pintu tertutup."

        e_tier, e_fg, e_bg = energy_tier(em_ens)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Output Ensemble</div>
            <div style="font-size:1.4rem;font-weight:700;color:{tier_fg};
                        font-family:'JetBrains Mono',monospace;margin:0.3rem 0">
                {ensemble:.1f}%
            </div>
            <div class="progress-bar-wrap">
                <div class="progress-bar-fill" style="width:{ensemble}%;
                     background:linear-gradient(90deg,{PAL['ok']},{PAL['warn']},{PAL['danger']});
                     opacity:0.7"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-size:0.65rem;color:#9CA3AF;margin-bottom:0.7rem">
                <span>0%</span><span>33%</span><span>66%</span><span>100%</span>
            </div>
            <div class="info-box" style="margin-bottom:0.6rem;font-size:0.75rem">{rec}</div>
            <div style="font-size:0.75rem;color:#5A6475;line-height:2">
                Selisih Suhu = <strong style="color:#1A2332">{delta_T_disp:+.1f} C</strong><br>
                Kelembaban = <strong style="color:#1A2332">{kelembaban:.0f}%</strong><br>
                Durasi = <strong style="color:#1A2332">{durasi_est:.1f} jam</strong><br>
                Est. Energi = <strong style="color:#1A2332">{em_ens:.3f} kWh</strong>
                <span style="background:{e_bg};color:{e_fg};padding:0.1rem 0.35rem;
                             border-radius:3px;font-size:0.65rem;font-weight:600;
                             margin-left:0.3rem">{e_tier}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ikhtisar Dataset ─────────────────────────────────────────────────────
    if df is not None:
        with st.expander("Lihat Ikhtisar Dataset (dataset_smart_ac.csv)", expanded=False):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown('<div class="section-label">10 Data Pertama</div>',
                            unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True, height=260)
            with col_b:
                st.markdown('<div class="section-label">Statistik Deskriptif</div>',
                            unsafe_allow_html=True)
                cols_stat = ["Temperatur_Ruangan", "Kelembaban",
                             "Kecepatan_Kompresor", "Konsumsi_Energi"]
                existing  = [c for c in cols_stat if c in df.columns]
                st.dataframe(df[existing].describe().round(3), use_container_width=True)

            fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.5))
            fig2.patch.set_facecolor("#FFFFFF")

            # Distribusi kecepatan
            axes2[0].hist(df["Kecepatan_Kompresor"], bins=20,
                          color=PAL["manual"] + "44", edgecolor=PAL["manual"], lw=0.7)
            axes2[0].axvline(df["Kecepatan_Kompresor"].mean(), color=PAL["danger"],
                             lw=1.5, ls="--",
                             label=f"Rata-rata = {df['Kecepatan_Kompresor'].mean():.1f}%")
            axes2[0].set_title("Distribusi Kecepatan Kompresor")
            axes2[0].set_xlabel("Kecepatan (%)"); axes2[0].set_ylabel("Jumlah Data")
            axes2[0].legend(fontsize=8); axes2[0].grid(True, alpha=0.3)

            # Distribusi energi
            if "Konsumsi_Energi" in df.columns:
                axes2[1].hist(df["Konsumsi_Energi"], bins=20,
                              color=PAL["anfis"] + "44", edgecolor=PAL["anfis"], lw=0.7)
                axes2[1].axvline(df["Konsumsi_Energi"].mean(), color=PAL["danger"],
                                 lw=1.5, ls="--",
                                 label=f"Rata-rata = {df['Konsumsi_Energi'].mean():.3f} kWh")
                axes2[1].set_title("Distribusi Konsumsi Energi")
                axes2[1].set_xlabel("Energi (kWh)"); axes2[1].set_ylabel("Jumlah Data")
                axes2[1].legend(fontsize=8); axes2[1].grid(True, alpha=0.3)

            # Distribusi mode AC
            mode_counts = df["Mode_AC"].value_counts().sort_index()
            mode_labels = {0: "Off", 1: "Pendinginan", 2: "Pemanasan"}
            axes2[2].bar(
                [mode_labels.get(k, str(k)) for k in mode_counts.index],
                mode_counts.values,
                color=[PAL["muted"] + "66", PAL["manual"] + "66", PAL["danger"] + "66"],
                edgecolor=[PAL["muted"], PAL["manual"], PAL["danger"]], lw=0.7
            )
            axes2[2].set_title("Distribusi Mode AC")
            axes2[2].set_xlabel("Mode"); axes2[2].set_ylabel("Jumlah Data")
            axes2[2].grid(axis="y", alpha=0.3)

            for ax in axes2:
                ax.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=1.2)
            st.pyplot(fig2, use_container_width=True)
            plt.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — FUNGSI KEANGGOTAAN (MEMBERSHIP FUNCTIONS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_mf:
    st.markdown('<div class="section-label">Visualisasi Fungsi Keanggotaan (MF)</div>',
                unsafe_allow_html=True)

    options_mf = ["Manual FIS"]
    if GA_MF_PARAMS is not None:
        options_mf.append("GA-Tuned FIS")
    if anfis_mf_raw is not None:
        options_mf.append("ANFIS (Gaussian MF)")
    if len(options_mf) >= 2:
        options_mf.append("Perbandingan Semua Model")

    view = st.radio("Pilih tampilan:", options_mf, horizontal=True, label_visibility="collapsed")

    DT_COLORS = {
        "sangat_dingin": "#1D4ED8",
        "dingin":        "#0EA5E9",
        "nyaman":        "#059669",
        "panas":         "#F59E0B",
        "sangat_panas":  "#DC2626",
    }
    KL_COLORS = {"rendah": "#6366F1", "sedang": "#0EA5E9", "tinggi": "#EF4444"}

    x_dT = np.linspace(-10, 15, 500)
    x_kl = np.linspace(30, 90, 500)
    dT_cur = T_room - T_target

    def plot_mf_single(params_trap, anfis_params_data, title, show_anfis=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor("#FFFFFF")
        fig.suptitle(title, fontsize=11, fontweight="bold", color="#1A2332", y=1.02)

        # MF ΔT
        for term, col in DT_COLORS.items():
            if show_anfis and anfis_params_data:
                mf = anfis_params_data["deltaT_mf"][term]
                y  = np.array([gaussian_mf_eval(x, mf["center"], mf["sigma"])
                                for x in np.linspace(0, 1, 500)])
                ax1.plot(x_dT, y, color=col, lw=2.2, label=term.replace("_", " "))
            else:
                y = trapezoid_mf(x_dT, *params_trap["delta_T"][term])
                ax1.plot(x_dT, y, color=col, lw=2.2, label=term.replace("_", " "))
            ax1.fill_between(x_dT, y, alpha=0.06, color=col)

        ax1.axvline(dT_cur, color="#374151", lw=1.3, ls="--", alpha=0.7,
                    label=f"Saat ini: {dT_cur:+.1f} C")
        ax1.set_title("Selisih Suhu (DT = Suhu Ruangan - Target)")
        ax1.set_xlabel("Selisih Suhu (°C)"); ax1.set_ylabel("Derajat Keanggotaan (mu)")
        ax1.set_xlim(-10, 15); ax1.set_ylim(-0.05, 1.15)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_facecolor("#FAFBFC")

        # MF Kelembaban
        for term, col in KL_COLORS.items():
            if show_anfis and anfis_params_data:
                mf = anfis_params_data["kl_mf"][term]
                # Gaussian diplot pada domain asli
                y  = np.array([gaussian_mf_eval(
                    (x - ANFIS_SCALER_KL[0]) / (ANFIS_SCALER_KL[1] - ANFIS_SCALER_KL[0] + 1e-12),
                    mf["center"], mf["sigma"]) for x in x_kl])
                ax2.plot(x_kl, y, color=col, lw=2.2, label=term)
            else:
                y = trapezoid_mf(x_kl, *params_trap["kelembaban"][term])
                ax2.plot(x_kl, y, color=col, lw=2.2, label=term)
            ax2.fill_between(x_kl, y, alpha=0.06, color=col)

        ax2.axvline(kelembaban, color="#374151", lw=1.3, ls="--", alpha=0.7,
                    label=f"Saat ini: {kelembaban:.0f}%")
        ax2.set_title("Kelembaban Udara (%RH)")
        ax2.set_xlabel("Kelembaban (%)"); ax2.set_ylabel("Derajat Keanggotaan (mu)")
        ax2.set_xlim(30, 90); ax2.set_ylim(-0.05, 1.15)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_facecolor("#FAFBFC")

        plt.tight_layout(pad=1.5)
        return fig

    if view == "Manual FIS":
        fig = plot_mf_single(MANUAL_MF_PARAMS, None, "Fungsi Keanggotaan — Manual FIS (Trapezoid)")
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown("""
        <div class="info-box">
            <strong>Manual FIS</strong> menggunakan fungsi keanggotaan trapezoid yang dirancang
            berdasarkan intuisi ahli. Parameter a, b, c, d setiap trapezoid ditentukan secara manual
            berdasarkan pengetahuan domain kendali suhu ruangan.
        </div>
        """, unsafe_allow_html=True)

    elif view == "GA-Tuned FIS":
        fig = plot_mf_single(GA_MF_PARAMS, None, "Fungsi Keanggotaan — GA-Tuned FIS (Trapezoid)")
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown('<div class="section-label" style="margin-top:1.2rem">Pergeseran Parameter MF (Manual ke GA)</div>',
                    unsafe_allow_html=True)
        rows = []
        for var in ("delta_T", "kelembaban"):
            for term in MANUAL_MF_PARAMS[var]:
                p0 = MANUAL_MF_PARAMS[var][term]
                p1 = GA_MF_PARAMS[var][term]
                rows.append({
                    "Variabel": var,
                    "Term": term,
                    "Parameter Manual": str([round(v, 2) for v in p0]),
                    "Parameter GA": str([round(v, 3) for v in p1]),
                    "Perubahan": "Ya" if any(abs(b-a) > 0.01 for a,b in zip(p0,p1)) else "-",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif view == "ANFIS (Gaussian MF)":
        fig = plot_mf_single(MANUAL_MF_PARAMS, anfis_mf_raw,
                             "Fungsi Keanggotaan — ANFIS (Gaussian, setelah training)",
                             show_anfis=True)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown("""
        <div class="info-box">
            <strong>ANFIS</strong> menggunakan fungsi keanggotaan Gaussian yang lebih mulus
            dibanding trapezoid. Setiap MF direpresentasikan oleh center (c) dan lebar (sigma)
            yang dioptimalkan menggunakan algoritma Adam (gradient descent) selama 300 epoch.
            MF ini memungkinkan komputasi gradien yang dibutuhkan dalam backpropagation.
        </div>
        """, unsafe_allow_html=True)

        if anfis_mf_raw:
            st.markdown('<div class="section-label" style="margin-top:1rem">Parameter Gaussian MF — ANFIS</div>',
                        unsafe_allow_html=True)
            rows_a = []
            for term, v in anfis_mf_raw.get("deltaT_mf", {}).items():
                rows_a.append({"Variabel": "delta_T", "Term": term,
                                "Center (c)": round(v["center"], 4),
                                "Sigma (sigma)": round(v["sigma"], 4)})
            for term, v in anfis_mf_raw.get("kl_mf", {}).items():
                rows_a.append({"Variabel": "kelembaban", "Term": term,
                                "Center (c)": round(v["center"], 4),
                                "Sigma (sigma)": round(v["sigma"], 4)})
            st.dataframe(pd.DataFrame(rows_a), use_container_width=True, hide_index=True)

    else:  # Perbandingan Semua
        st.markdown('<div class="section-label">Perbandingan Ketiga Model MF</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#FFFFFF")
        fig.suptitle("Perbandingan MF: Manual vs GA-Tuned vs ANFIS",
                     fontsize=11, fontweight="bold", color="#1A2332", y=1.02)

        ax1, ax2 = axes

        for term, col in DT_COLORS.items():
            ym = trapezoid_mf(x_dT, *MANUAL_MF_PARAMS["delta_T"][term])
            ax1.plot(x_dT, ym, color=col, lw=1.5, ls=":", alpha=0.8)
            if GA_MF_PARAMS:
                yg = trapezoid_mf(x_dT, *GA_MF_PARAMS["delta_T"][term])
                ax1.plot(x_dT, yg, color=col, lw=1.8, ls="--", alpha=0.85)
            if anfis_mf_raw:
                mf = anfis_mf_raw["deltaT_mf"][term]
                ya = np.array([gaussian_mf_eval(x, mf["center"], mf["sigma"])
                                for x in np.linspace(0, 1, 500)])
                ax1.plot(x_dT, ya, color=col, lw=2.2, ls="-", label=term.replace("_", " "))

        ax1.axvline(dT_cur, color="#374151", lw=1, ls="-.", alpha=0.5)
        ax1.set_title("Selisih Suhu (DT)\n[titik-titik=Manual | putus-putus=GA | garis=ANFIS]")
        ax1.set_xlabel("Selisih Suhu (°C)"); ax1.set_ylabel("Derajat Keanggotaan")
        ax1.legend(fontsize=7.5, loc="upper left"); ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-10, 15); ax1.set_facecolor("#FAFBFC")

        for term, col in KL_COLORS.items():
            ym = trapezoid_mf(x_kl, *MANUAL_MF_PARAMS["kelembaban"][term])
            ax2.plot(x_kl, ym, color=col, lw=1.5, ls=":", alpha=0.8)
            if GA_MF_PARAMS:
                yg = trapezoid_mf(x_kl, *GA_MF_PARAMS["kelembaban"][term])
                ax2.plot(x_kl, yg, color=col, lw=1.8, ls="--", alpha=0.85)
            if anfis_mf_raw:
                mf = anfis_mf_raw["kl_mf"][term]
                ya = np.array([gaussian_mf_eval(
                    (x - ANFIS_SCALER_KL[0]) / (ANFIS_SCALER_KL[1] - ANFIS_SCALER_KL[0] + 1e-12),
                    mf["center"], mf["sigma"]) for x in x_kl])
                ax2.plot(x_kl, ya, color=col, lw=2.2, ls="-", label=term)

        ax2.axvline(kelembaban, color="#374151", lw=1, ls="-.", alpha=0.5)
        ax2.set_title("Kelembaban\n[titik-titik=Manual | putus-putus=GA | garis=ANFIS]")
        ax2.set_xlabel("Kelembaban (%)"); ax2.set_ylabel("Derajat Keanggotaan")
        ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.3)
        ax2.set_xlim(30, 90); ax2.set_facecolor("#FAFBFC")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="info-box">
            Perbandingan ini menunjukkan bagaimana setiap metode memodifikasi bentuk kurva MF.
            Manual FIS menggunakan trapezoid simetris berdasarkan intuisi.
            GA menggeser batas-batas trapezoid untuk meminimalkan RMSE pada dataset.
            ANFIS mengubah bentuk menjadi Gaussian dan menyesuaikan center serta sigma
            melalui optimasi gradient.
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — RULE BASE (15 RULES)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_rules:
    st.markdown('<div class="section-label">Rule Base Sugeno Zero-Order (15 Rules)</div>',
                unsafe_allow_html=True)

    dT_cur = T_room - T_target
    p = MANUAL_MF_PARAMS

    mu_dT = {k: float(trapezoid_mf(np.array([dT_cur]), *v)[0])
              for k, v in p["delta_T"].items()}
    mu_kl = {k: float(trapezoid_mf(np.array([kelembaban]), *v)[0])
              for k, v in p["kelembaban"].items()}

    # Hitung firing strength semua rules
    all_rules = []
    for (dT_lbl, kl_lbl), spd_lbl in RULE_TABLE.items():
        firing = mu_dT[dT_lbl] * mu_kl[kl_lbl]
        all_rules.append({
            "IF Selisih Suhu": dT_lbl.replace("_", " "),
            "DAN Kelembaban":  kl_lbl,
            "MAKA Kecepatan": spd_lbl,
            "Nilai Output (%)": p["kecepatan"][spd_lbl],
            "Kekuatan Aktivasi": round(firing, 5),
            "_firing": firing,
            "_dT": dT_lbl,
            "_kl": kl_lbl,
        })

    max_fire = max(r["_firing"] for r in all_rules)
    dom_rule = max(all_rules, key=lambda r: r["_firing"])

    col_ctx, col_tbl = st.columns([1, 2])

    with col_ctx:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Kondisi Input Saat Ini</div>
            <div style="font-size:0.81rem;color:#3D4757;line-height:2.1;margin-top:0.5rem">
                Suhu Ruangan = <strong>{T_room} C</strong><br>
                Suhu Target &nbsp;= <strong>{T_target} C</strong><br>
                Selisih (DT) = <strong>{dT_cur:+.1f} C</strong><br>
                Kelembaban &nbsp;= <strong>{kelembaban:.0f}%</strong><br>
                Status &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong>{"Ada Penghuni" if occ else "Kosong"}</strong>
            </div>
            <hr style="margin:0.7rem 0">
            <div class="kpi-label">Derajat Keanggotaan</div>
            <div style="font-size:0.77rem;color:#3D4757;line-height:2;margin-top:0.4rem">
        """, unsafe_allow_html=True)
        for term, col_hex in DT_COLORS.items():
            st.markdown(
                f'<div style="color:{col_hex}">{term.replace("_"," ")} = {mu_dT[term]:.4f}</div>',
                unsafe_allow_html=True)
        for term, col_hex in KL_COLORS.items():
            st.markdown(
                f'<div style="color:{col_hex}">kl_{term} = {mu_kl[term]:.4f}</div>',
                unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_tbl:
        df_rules = pd.DataFrame([{
            "Jika DT": r["IF Selisih Suhu"],
            "Dan Kelembaban": r["DAN Kelembaban"],
            "Maka Kecepatan": r["MAKA Kecepatan"],
            "Output (%)": r["Nilai Output (%)"],
            "Kekuatan": r["Kekuatan Aktivasi"],
        } for r in all_rules])

        st.dataframe(
            df_rules.style.apply(
                lambda row: [
                    "background:#EFF6FF;font-weight:600;" if row["Kekuatan"] == max_fire else ""
                ] * len(row), axis=1
            ).format({"Kekuatan": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
            height=340,
        )

        active_count = sum(1 for r in all_rules if r["_firing"] > 0.001)
        if occ == 0:
            st.markdown("""
            <div class="info-box warn">
                Status = Kosong. Semua rules dilewati. Output = 0% (AC mati).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <strong>Rules aktif:</strong> {active_count} dari 15 &nbsp;&nbsp;
                <strong>Rule dominan:</strong> JIKA DT = {dom_rule["IF Selisih Suhu"]}
                DAN Kelembaban = {dom_rule["DAN Kelembaban"]}
                (kekuatan = {dom_rule["_firing"]:.4f})
                &rarr; MAKA Kecepatan = <strong>{dom_rule["MAKA Kecepatan"]}</strong>
                ({dom_rule["Nilai Output (%)"]}%)
            </div>
            """, unsafe_allow_html=True)

    # Grafik kekuatan aktivasi
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Kekuatan Aktivasi Setiap Rule</div>',
                unsafe_allow_html=True)

    rule_ids = [f"DT={r['IF Selisih Suhu'][:7]}\nKl={r['DAN Kelembaban'][:4]}"
                for r in all_rules]
    acts     = [r["Kekuatan Aktivasi"] for r in all_rules]
    b_clrs   = [PAL["manual"] if a == max(acts) else PAL["manual"] + "55" for a in acts]

    fig_r, ax_r = plt.subplots(figsize=(13, 2.8))
    fig_r.patch.set_facecolor("#FFFFFF")
    bars = ax_r.bar(range(len(acts)), acts,
                    color=b_clrs, edgecolor=PAL["manual"], lw=0.6, width=0.62)
    for b, a in zip(bars, acts):
        if a > 0.001:
            ax_r.text(b.get_x() + b.get_width()/2, a + 0.003,
                      f"{a:.3f}", ha="center", va="bottom", fontsize=7.5,
                      color="#1A2332", fontweight="600")
    ax_r.set_xticks(range(len(acts)))
    ax_r.set_xticklabels(rule_ids, fontsize=6.5)
    ax_r.set_ylabel("Kekuatan Aktivasi")
    ax_r.set_ylim(0, max(max(acts) * 1.3, 0.1))
    ax_r.grid(axis="y", alpha=0.3); ax_r.set_facecolor("#FAFBFC")
    plt.tight_layout(pad=0.8)
    st.pyplot(fig_r, use_container_width=True)
    plt.close()

    # Tabel Rule Matrix (ΔT × Kelembaban)
    st.markdown('<div class="section-label" style="margin-top:1rem">Matriks Rule (Selisih Suhu x Kelembaban)</div>',
                unsafe_allow_html=True)
    dT_terms = list(MANUAL_MF_PARAMS["delta_T"].keys())
    kl_terms = list(MANUAL_MF_PARAMS["kelembaban"].keys())
    matrix = pd.DataFrame(index=dT_terms, columns=kl_terms)
    for (d, k), s in RULE_TABLE.items():
        matrix.loc[d, k] = f"{s} ({MANUAL_MF_PARAMS['kecepatan'][s]}%)"
    matrix.index = [i.replace("_", " ") for i in matrix.index]
    st.dataframe(matrix, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — ANALISIS PERFORMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_perf:
    if metrics is None:
        st.error("File metrics.json tidak ditemukan. Jalankan sel export di notebook terlebih dahulu.")
    else:
        method_names = list(metrics.keys())
        m_colors     = [PAL["manual"], PAL["ga"], PAL["anfis"]]
        ref_mse      = metrics[method_names[0]]["MSE"]

        # ── Kartu Metrik ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Metrik Performa — Semua Model</div>',
                    unsafe_allow_html=True)
        cols_m = st.columns(len(method_names))
        for col, name, color in zip(cols_m, method_names, m_colors):
            m   = metrics[name]
            imp = (ref_mse - m["MSE"]) / ref_mse * 100
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label" style="color:{color}">{name}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.5rem">
                        <div>
                            <div style="font-size:0.63rem;color:#9CA3AF;font-weight:600;text-transform:uppercase">MSE</div>
                            <div style="font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:{color}">{m['MSE']:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size:0.63rem;color:#9CA3AF;font-weight:600;text-transform:uppercase">R2</div>
                            <div style="font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:{color}">{m['R2']:.4f}</div>
                        </div>
                        <div>
                            <div style="font-size:0.63rem;color:#9CA3AF;font-weight:600;text-transform:uppercase">RMSE</div>
                            <div style="font-size:0.95rem;font-weight:600;color:#3D4757">{m['RMSE']:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size:0.63rem;color:#9CA3AF;font-weight:600;text-transform:uppercase">MAE</div>
                            <div style="font-size:0.95rem;font-weight:600;color:#3D4757">{m['MAE']:.3f}</div>
                        </div>
                    </div>
                    <div style="margin-top:0.6rem;font-size:0.72rem;
                                color:{'#065F46' if imp > 0 else '#991B1B'};font-weight:600">
                        {"Meningkat" if imp > 0 else "Menurun"} {abs(imp):.1f}% vs {method_names[0]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar Chart Metrik & Scatter ─────────────────────────────────────────
        col_bar, col_sc = st.columns(2)

        with col_bar:
            st.markdown('<div class="section-label">Perbandingan Metrik</div>',
                        unsafe_allow_html=True)
            mk_list = ["MSE", "RMSE", "MAE", "R2"]
            fig_b, axes_b = plt.subplots(2, 2, figsize=(8, 6))
            fig_b.patch.set_facecolor("#FFFFFF")
            x_pos = np.arange(len(method_names))

            for ax_b, mk in zip(axes_b.flat, mk_list):
                vals = [metrics[n][mk] for n in method_names]
                bars_b = ax_b.bar(x_pos, vals,
                                  color=[c + "44" for c in m_colors],
                                  edgecolor=m_colors, linewidth=1.2, width=0.6)
                best_i = int(np.argmin(vals)) if mk != "R2" else int(np.argmax(vals))
                for i, (b, v) in enumerate(zip(bars_b, vals)):
                    ax_b.text(b.get_x() + b.get_width()/2,
                              b.get_height() + max(vals) * 0.01,
                              f"{v:.3f}", ha="center", va="bottom",
                              fontsize=7.5, color=m_colors[i],
                              fontweight="700" if i == best_i else "400")
                short_names = [n.replace(" FIS", "\nFIS") for n in method_names]
                ax_b.set_xticks(x_pos); ax_b.set_xticklabels(short_names, fontsize=7.5)
                ax_b.set_title(mk, fontsize=9); ax_b.set_ylabel(mk, fontsize=8)
                ax_b.grid(axis="y", alpha=0.3); ax_b.set_facecolor("#FAFBFC")

            plt.tight_layout(pad=1.2)
            st.pyplot(fig_b, use_container_width=True)
            plt.close()

        with col_sc:
            st.markdown('<div class="section-label">Aktual vs Prediksi (Scatter Plot)</div>',
                        unsafe_allow_html=True)
            if predictions is not None:
                y_true = predictions["Kecepatan_Kompresor"].values
                pred_cols = {
                    method_names[0]: "Pred_Manual_FIS",
                    method_names[1] if len(method_names) > 1 else "": "Pred_GA_FIS",
                    method_names[2] if len(method_names) > 2 else "": "Pred_ANN_FIS",
                }

                fig_sc, axes_sc = plt.subplots(len(method_names), 1,
                                                figsize=(5, 3.5 * len(method_names)))
                fig_sc.patch.set_facecolor("#FFFFFF")
                if len(method_names) == 1:
                    axes_sc = [axes_sc]

                for ax_sc, name, col_hex in zip(axes_sc, method_names, m_colors):
                    col_name = pred_cols.get(name, "")
                    if col_name not in predictions.columns:
                        ax_sc.text(0.5, 0.5, f"Data {name}\ntidak tersedia",
                                   ha="center", va="center", transform=ax_sc.transAxes)
                        continue
                    y_pred = predictions[col_name].values
                    ax_sc.scatter(y_true, y_pred, alpha=0.35, c=col_hex, s=14)
                    ax_sc.plot([0, 100], [0, 100], color="#374151", lw=1, ls="--", alpha=0.5)
                    ax_sc.set_title(f"{name}  |  R2 = {metrics[name]['R2']:.4f}",
                                    fontsize=8.5)
                    ax_sc.set_xlabel("Aktual (%)"); ax_sc.set_ylabel("Prediksi (%)")
                    ax_sc.set_xlim(0, 110); ax_sc.set_ylim(-5, 115)
                    ax_sc.grid(True, alpha=0.3); ax_sc.set_facecolor("#FAFBFC")

                plt.tight_layout(pad=1.2)
                st.pyplot(fig_sc, use_container_width=True)
                plt.close()
            else:
                st.info("File predictions.json belum tersedia.")

        # ── Plot Prediksi 50 Sampel ─────────────────────────────────────────
        if predictions is not None:
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Prediksi vs Aktual — 50 Sampel Pertama</div>',
                        unsafe_allow_html=True)
            n_show   = 50
            idx50    = np.arange(n_show)
            y_true50 = predictions["Kecepatan_Kompresor"].values[:n_show]
            pred_cfgs = [
                ("Pred_Manual_FIS", method_names[0], PAL["manual"]),
                ("Pred_GA_FIS",     method_names[1] if len(method_names) > 1 else "GA", PAL["ga"]),
                ("Pred_ANN_FIS",    method_names[2] if len(method_names) > 2 else "ANFIS", PAL["anfis"]),
            ]

            available = [(c, n, col) for c, n, col in pred_cfgs if c in predictions.columns]
            fig_lp, axes_lp = plt.subplots(len(available), 1,
                                            figsize=(13, 3.5 * len(available)), sharex=True)
            fig_lp.patch.set_facecolor("#FFFFFF")
            if len(available) == 1:
                axes_lp = [axes_lp]

            for ax_lp, (col_name, title, color) in zip(axes_lp, available):
                y_pred50 = predictions[col_name].values[:n_show]
                ax_lp.plot(idx50, y_true50, color="#374151", lw=2, label="Aktual", zorder=3)
                ax_lp.plot(idx50, y_pred50, color=color, lw=1.8, ls="--",
                           label=f"Prediksi {title}", zorder=2)
                ax_lp.fill_between(idx50, y_true50, y_pred50, alpha=0.1, color=color)
                ax_lp.set_ylabel("Kecepatan (%)", fontsize=8)
                ax_lp.set_title(f"{title}", fontsize=9)
                ax_lp.legend(fontsize=8, loc="upper right")
                ax_lp.grid(True, alpha=0.3); ax_lp.set_facecolor("#FAFBFC")

            if axes_lp.any():
                axes_lp[-1].set_xlabel("Indeks Sampel")
            plt.tight_layout(pad=1.2)
            st.pyplot(fig_lp, use_container_width=True)
            plt.close()

        # ── Riwayat GA ─────────────────────────────────────────────────────
        if ga_history is not None:
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Kurva Konvergensi GA (Pop=60, Gen=120)</div>',
                        unsafe_allow_html=True)
            gens = np.arange(1, len(ga_history["min"]) + 1)
            fig_ga, ax_ga = plt.subplots(figsize=(11, 3.5))
            fig_ga.patch.set_facecolor("#FFFFFF")
            ax_ga.plot(gens, ga_history["min"], color=PAL["ga"], lw=2.2, label="Terbaik (Min)")
            ax_ga.plot(gens, ga_history["avg"], color=PAL["ga"], lw=1.4, ls="--",
                       alpha=0.6, label="Rata-rata Populasi")
            ax_ga.fill_between(gens, ga_history["min"], ga_history["avg"],
                                alpha=0.08, color=PAL["ga"])

            # Tampilkan restart events
            stag_gens = ga_history.get("stagnation_gens", [])
            for i, sg in enumerate(stag_gens):
                ax_ga.axvline(x=sg, color="#7C3AED", lw=1.2, ls=":", alpha=0.7,
                              label="Random Restart" if i == 0 else "")

            if metrics:
                ref_rmse = metrics[method_names[0]]["RMSE"]
                ax_ga.axhline(ref_rmse, color=PAL["muted"], lw=1.2, ls="-.",
                              label=f"Manual FIS RMSE = {ref_rmse:.3f}")
            ax_ga.set_xlabel("Generasi"); ax_ga.set_ylabel("Fitness (RMSE + Penalti)")
            ax_ga.legend(fontsize=8.5); ax_ga.grid(True, alpha=0.3); ax_ga.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=1)
            st.pyplot(fig_ga, use_container_width=True)
            plt.close()

        # ── Riwayat Training ANFIS ─────────────────────────────────────────
        if anfis_hist is not None:
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Kurva Training Loss ANFIS (300 Epoch)</div>',
                        unsafe_allow_html=True)
            ep   = np.arange(1, len(anfis_hist) + 1)
            best_ep   = int(np.argmin(anfis_hist))
            best_loss = min(anfis_hist)

            fig_an, ax_an = plt.subplots(figsize=(11, 3.5))
            fig_an.patch.set_facecolor("#FFFFFF")
            ax_an.plot(ep, anfis_hist, color=PAL["anfis"], lw=2)
            ax_an.fill_between(ep, anfis_hist, alpha=0.12, color=PAL["anfis"])
            ax_an.annotate(
                f"Min = {best_loss:.5f}\n(Epoch {best_ep})",
                xy=(best_ep, best_loss),
                xytext=(best_ep + max(len(ep)//10, 5), best_loss + max(anfis_hist) * 0.05),
                arrowprops=dict(arrowstyle="->", color="#374151"),
                fontsize=8.5, color="#374151"
            )
            ax_an.set_xlabel("Epoch"); ax_an.set_ylabel("MSE Loss (ternormalisasi)")
            ax_an.set_title("ANFIS Training Loss")
            ax_an.grid(True, alpha=0.3); ax_an.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=1)
            st.pyplot(fig_an, use_container_width=True)
            plt.close()

        # ── Simpulan ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Kesimpulan Perbandingan</div>',
                    unsafe_allow_html=True)

        if metrics and len(method_names) >= 3:
            ga_imp  = (metrics[method_names[0]]["RMSE"] - metrics[method_names[1]]["RMSE"]) \
                       / metrics[method_names[0]]["RMSE"] * 100
            an_imp  = (metrics[method_names[0]]["RMSE"] - metrics[method_names[2]]["RMSE"]) \
                       / metrics[method_names[0]]["RMSE"] * 100

            st.markdown(f"""
            <div class="info-box">
                <strong>Manual FIS</strong> — Direkomendasikan ketika transparansi dan kemudahan
                audit oleh pakar menjadi prioritas utama. Berguna pada tahap desain awal atau
                ketika data pelatihan sangat terbatas.
            </div>
            <div class="info-box" style="border-left-color:{PAL['ga']}">
                <strong>GA-Tuned FIS</strong> — GA berhasil meningkatkan RMSE sebesar
                <strong>{ga_imp:.1f}%</strong> dibanding Manual FIS (R2 = {metrics[method_names[1]]['R2']:.4f}).
                Direkomendasikan ketika akurasi perlu dimaksimalkan namun interpretabilitas
                struktur rule masih diinginkan.
            </div>
            <div class="info-box" style="border-left-color:{PAL['anfis']}">
                <strong>ANFIS</strong> — Menggunakan Gaussian MF dan optimasi gradient end-to-end.
                Mencapai R2 = {metrics[method_names[2]]['R2']:.4f}
                ({an_imp:.1f}% perubahan RMSE vs Manual). ANFIS unggul dalam menangkap pola
                non-linear yang halus dan tidak dapat direpresentasikan oleh trapezoid kaku.
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — KONSUMSI ENERGI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_energy:
    st.markdown('<div class="section-label">Analisis Konsumsi Energi</div>',
                unsafe_allow_html=True)

    # Penjelasan formula
    st.markdown("""
    <div class="info-box">
        <strong>Formula Konsumsi Energi:</strong><br>
        E = (Kecepatan / 100) x P_rated x t_operasi x faktor_kelembaban<br>
        P_rated = 2.5 kW (AC 1 PK standar) &nbsp;|&nbsp;
        faktor_kelembaban = 1 + 0.1 x (Kelembaban / 50)<br>
        Semakin lembab udara, kompresor bekerja lebih keras sehingga energi meningkat.
    </div>
    """, unsafe_allow_html=True)

    # Estimasi real-time
    st.markdown('<div class="section-label">Estimasi Energi Real-Time (Input Sidebar)</div>',
                unsafe_allow_html=True)
    e_cols = st.columns(4)
    for col, (label, val_spd, color) in zip(
        e_cols,
        [("Manual FIS", fm, PAL["manual"]),
         ("GA-Tuned FIS", fg if fg else 0, PAL["ga"]),
         ("ANFIS", fa if fa else 0, PAL["anfis"]),
         ("Ensemble", ensemble, PAL["actual"])]
    ):
        e_val = hitung_energi(val_spd, kelembaban, durasi_est)
        e_tier, e_fg, e_bg = energy_tier(e_val)
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div style="font-size:1.5rem;font-weight:700;
                            font-family:'JetBrains Mono',monospace;color:{color}">
                    {e_val:.3f}
                    <span style="font-size:0.8rem;color:#9CA3AF;font-weight:400">kWh</span>
                </div>
                <div class="kpi-sub">Kecepatan: {val_spd:.1f}%</div>
                <div class="kpi-sub">Durasi: {durasi_est:.1f} jam</div>
                <span style="background:{e_bg};color:{e_fg};padding:0.15rem 0.4rem;
                             border-radius:3px;font-size:0.66rem;font-weight:600">{e_tier}</span>
            </div>
            """, unsafe_allow_html=True)

    # Data energi dari notebook (artifacts)
    if energy_m is not None and predictions is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ringkasan Energi dari Dataset (200 Sampel)</div>',
                    unsafe_allow_html=True)

        energy_keys = {
            "aktual":        ("Aktual", PAL["actual"]),
            "manual_fis":    ("Manual FIS", PAL["manual"]),
            "ga_tuned_fis":  ("GA-Tuned FIS", PAL["ga"]),
            "anfis":         ("ANFIS", PAL["anfis"]),
        }
        e_display_cols = st.columns(4)
        for col, (key, (label, color)) in zip(e_display_cols, energy_keys.items()):
            if key in energy_m:
                d = energy_m[key]
                mae_str = f"{d.get('mae_kwh', 0):.4f} kWh" if "mae_kwh" in d else "—"
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label" style="color:{color}">{label}</div>
                        <div style="font-size:1.3rem;font-weight:700;
                                    font-family:'JetBrains Mono',monospace;color:{color}">
                            {d['total_kwh']:.2f}
                            <span style="font-size:0.78rem;color:#9CA3AF;font-weight:400">kWh total</span>
                        </div>
                        <div class="kpi-sub">Rata-rata: {d['mean_kwh']:.4f} kWh/sesi</div>
                        <div class="kpi-sub">MAE Energi: {mae_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualisasi energi
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown('<div class="section-label">Scatter: Energi Aktual vs Prediksi</div>',
                        unsafe_allow_html=True)
            energy_pred_cols = [
                ("Energy_Manual_FIS", "Manual FIS",   PAL["manual"]),
                ("Energy_GA_FIS",     "GA-Tuned FIS", PAL["ga"]),
                ("Energy_ANN_FIS",    "ANFIS",         PAL["anfis"]),
            ]
            available_e = [(c, n, col) for c, n, col in energy_pred_cols
                           if c in predictions.columns]
            if available_e and "Konsumsi_Energi" in predictions.columns:
                fig_e, axes_e = plt.subplots(1, len(available_e),
                                              figsize=(4 * len(available_e), 4))
                fig_e.patch.set_facecolor("#FFFFFF")
                if len(available_e) == 1:
                    axes_e = [axes_e]
                y_e_true = predictions["Konsumsi_Energi"].values
                for ax_e, (col_n, name, color) in zip(axes_e, available_e):
                    y_e_pred = predictions[col_n].values
                    ax_e.scatter(y_e_true, y_e_pred, alpha=0.4, c=color, s=15)
                    mx = max(y_e_true.max(), y_e_pred.max()) * 1.05
                    ax_e.plot([0, mx], [0, mx], color="#374151", lw=1, ls="--", alpha=0.5)
                    mae_e = float(np.mean(np.abs(y_e_pred - y_e_true)))
                    ax_e.set_title(f"{name}\nMAE = {mae_e:.4f} kWh", fontsize=8.5)
                    ax_e.set_xlabel("Aktual (kWh)"); ax_e.set_ylabel("Prediksi (kWh)")
                    ax_e.grid(True, alpha=0.3); ax_e.set_facecolor("#FAFBFC")
                plt.tight_layout(pad=1.2)
                st.pyplot(fig_e, use_container_width=True)
                plt.close()

        with col_e2:
            st.markdown('<div class="section-label">Total & Distribusi Konsumsi Energi</div>',
                        unsafe_allow_html=True)
            model_lbls  = [energy_keys[k][0] for k in energy_keys if k in energy_m]
            total_vals  = [energy_m[k]["total_kwh"] for k in energy_keys if k in energy_m]
            bar_clrs_e  = [energy_keys[k][1] for k in energy_keys if k in energy_m]

            fig_eb, ax_eb = plt.subplots(figsize=(6, 4))
            fig_eb.patch.set_facecolor("#FFFFFF")
            bars_eb = ax_eb.bar(model_lbls, total_vals,
                                color=[c + "55" for c in bar_clrs_e],
                                edgecolor=bar_clrs_e, lw=1.2, width=0.55)
            for b, v in zip(bars_eb, total_vals):
                ax_eb.text(b.get_x() + b.get_width()/2, b.get_height() + max(total_vals)*0.01,
                           f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="600")
            ax_eb.set_ylabel("Total Konsumsi Energi (kWh)")
            ax_eb.set_title("Perbandingan Total Energi Dataset")
            ax_eb.grid(axis="y", alpha=0.3); ax_eb.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=1)
            st.pyplot(fig_eb, use_container_width=True)
            plt.close()

    elif predictions is None:
        st.info("File predictions.json belum tersedia. Jalankan notebook untuk menghasilkan data prediksi.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — ABLATION STUDY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ablation:
    if ablation is None:
        st.error("File ablation_results.json tidak ditemukan di folder artifacts/ablation/.")
    else:
        st.markdown('<div class="section-label">Ablation Study — Sensitivitas Parameter GA</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Ablation study menguji <strong>8 konfigurasi berbeda</strong> GA dengan variasi
            ukuran populasi dan jumlah generasi, untuk menjawab pertanyaan:
            Apa pengaruh konfigurasi GA terhadap kualitas solusi dan kecepatan konvergensi?
        </div>
        """, unsafe_allow_html=True)

        # Tentukan threshold dari metrics jika tersedia
        ref_rmse = metrics[method_names[0]]["RMSE"] if metrics else None
        target_thr = ref_rmse * 0.90 if ref_rmse else None

        # ── Kartu ringkasan ─────────────────────────────────────────────────
        best_r = min(ablation, key=lambda r: r["rmse"])
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE Manual FIS (Baseline)",
                  f"{ref_rmse:.4f}" if ref_rmse else "—")
        c2.metric("RMSE GA Terbaik",
                  f"{best_r['rmse']:.4f}",
                  delta=f"{best_r['rmse'] - ref_rmse:.4f}" if ref_rmse else None,
                  delta_color="inverse")
        c3.metric(
                    "Konfigurasi Terbaik",
                    f"Pop={best_r['pop']}\nGen={best_r['gen']}"  # pisah jadi 2 baris
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel hasil
        col_tbl2, col_chart2 = st.columns([1, 2])

        with col_tbl2:
            st.markdown('<div class="section-label">Tabel Hasil Ablation</div>',
                        unsafe_allow_html=True)
            abl_df = pd.DataFrame([{
                "Konfigurasi":     r["label"].strip().replace("★", "(Terbaik)"),
                "Pop Size":        r["pop"],
                "Generasi":        r["gen"],
                "RMSE Final":      round(r["rmse"], 4),
                "Gen Konvergensi": r.get("conv_gen", "—"),
                "Prematur":        "Ya" if r.get("premature", False) else "Tidak",
                "Waktu (s)":       r.get("waktu_s", "—"),
            } for r in ablation])
            st.dataframe(abl_df, use_container_width=True, hide_index=True, height=310)

        with col_chart2:
            st.markdown('<div class="section-label">RMSE per Konfigurasi</div>',
                        unsafe_allow_html=True)
            abl_labels = [f"Pop={r['pop']}\nGen={r['gen']}" for r in ablation]
            abl_rmse   = [r["rmse"] for r in ablation]
            abl_colors = [
                PAL["anfis"] if r["rmse"] == min(abl_rmse)
                else (PAL["danger"] if r.get("premature", False) else PAL["manual"])
                for r in ablation
            ]

            fig_abl, ax_abl = plt.subplots(figsize=(10, 4))
            fig_abl.patch.set_facecolor("#FFFFFF")
            bars_a = ax_abl.barh(abl_labels, abl_rmse,
                                  color=[c + "33" for c in abl_colors],
                                  edgecolor=abl_colors, lw=1.2, height=0.58)
            if ref_rmse:
                ax_abl.axvline(ref_rmse, color=PAL["muted"], lw=1.4, ls="-.",
                               label=f"Manual FIS RMSE = {ref_rmse:.3f}")
            for b, v in zip(bars_a, abl_rmse):
                ax_abl.text(b.get_width() + max(abl_rmse)*0.005,
                            b.get_y() + b.get_height()/2,
                            f"{v:.3f}", va="center", fontsize=8.5, fontweight="600")
            ax_abl.set_xlabel("RMSE Final")
            ax_abl.legend(fontsize=8)
            ax_abl.grid(axis="x", alpha=0.3); ax_abl.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=1.2)
            st.pyplot(fig_abl, use_container_width=True)
            plt.close()

        # ── Grafik scatter Pop vs RMSE ──────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Hubungan Pop Size dan Gen dengan RMSE</div>',
                    unsafe_allow_html=True)
        col_sc1, col_sc2 = st.columns(2)

        with col_sc1:
            pop_vals = [r["pop"] for r in ablation]
            gen_vals = [r["gen"] for r in ablation]
            fig_ps, ax_ps = plt.subplots(figsize=(5.5, 4))
            fig_ps.patch.set_facecolor("#FFFFFF")
            sc_ps = ax_ps.scatter(pop_vals, abl_rmse, c=gen_vals,
                                   cmap="RdYlGn", s=100, alpha=0.9,
                                   edgecolors="#374151", lw=0.7)
            plt.colorbar(sc_ps, ax=ax_ps, label="Jumlah Generasi")
            if ref_rmse:
                ax_ps.axhline(ref_rmse, color=PAL["muted"], lw=1.2, ls="--", alpha=0.8)
            ax_ps.set_xlabel("Ukuran Populasi")
            ax_ps.set_ylabel("RMSE Final")
            ax_ps.set_title("Pop Size vs RMSE (warna = jumlah generasi)")
            for r in ablation:
                ax_ps.annotate(f"G={r['gen']}", (r["pop"], r["rmse"]),
                               textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax_ps.grid(True, alpha=0.3); ax_ps.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=0.8)
            st.pyplot(fig_ps, use_container_width=True)
            plt.close()

        with col_sc2:
            fig_gs, ax_gs = plt.subplots(figsize=(5.5, 4))
            fig_gs.patch.set_facecolor("#FFFFFF")
            sc_gs = ax_gs.scatter(gen_vals, abl_rmse, c=pop_vals,
                                   cmap="Blues", s=100, alpha=0.9,
                                   edgecolors="#374151", lw=0.7)
            plt.colorbar(sc_gs, ax=ax_gs, label="Ukuran Populasi")
            if ref_rmse:
                ax_gs.axhline(ref_rmse, color=PAL["muted"], lw=1.2, ls="--", alpha=0.8)
            ax_gs.set_xlabel("Jumlah Generasi")
            ax_gs.set_ylabel("RMSE Final")
            ax_gs.set_title("Jumlah Generasi vs RMSE (warna = pop size)")
            for r in ablation:
                ax_gs.annotate(f"P={r['pop']}", (r["gen"], r["rmse"]),
                               textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax_gs.grid(True, alpha=0.3); ax_gs.set_facecolor("#FAFBFC")
            plt.tight_layout(pad=0.8)
            st.pyplot(fig_gs, use_container_width=True)
            plt.close()

        # ── Temuan Kunci ────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Temuan dan Interpretasi</div>',
                    unsafe_allow_html=True)

        premature_configs = [r for r in ablation if r.get("premature", False)]
        if premature_configs:
            st.markdown("""
            <div class="info-box warn">
                <strong>Perhatian — Konvergensi Prematur Terdeteksi:</strong><br>
                Konfigurasi dengan populasi kecil (Pop &le; 10) menunjukkan konvergensi prematur,
                artinya algoritma terjebak di solusi lokal sebelum mengeksplorasi ruang pencarian
                secara memadai. RMSE yang dihasilkan masih lebih tinggi dari threshold target.
            </div>
            """, unsafe_allow_html=True)

        best_is_standard = "60" in str(best_r.get("pop", "")) or "Standar" in best_r.get("label", "")
        st.markdown(f"""
        <div class="info-box">
            <strong>Populasi kecil (Pop &le; 10, Gen &le; 20):</strong> Konvergensi prematur.
            Diversitas populasi terlalu rendah untuk mengeksplorasi ruang parameter MF secara efektif.
        </div>
        <div class="info-box" style="border-left-color:{PAL['ga']}">
            <strong>Konfigurasi standar (Pop=60, Gen=120):</strong> Titik keseimbangan optimal
            antara kualitas solusi dan efisiensi komputasi. Direkomendasikan untuk penggunaan umum.
        </div>
        <div class="info-box">
            <strong>Konfigurasi besar (Pop &ge; 100, Gen &ge; 150):</strong> Peningkatan RMSE
            sangat marginal dibanding konfigurasi standar, namun waktu komputasi meningkat
            secara signifikan — menunjukkan <em>diminishing returns</em>.
        </div>
        <div class="info-box success">
            <strong>Konfigurasi Terbaik: Pop={best_r['pop']}, Gen={best_r['gen']}</strong><br>
            RMSE = {best_r['rmse']:.4f}
            {f"(penurunan {((ref_rmse - best_r['rmse'])/ref_rmse*100):.1f}% dari baseline)" if ref_rmse else ""}
        </div>
        """, unsafe_allow_html=True)