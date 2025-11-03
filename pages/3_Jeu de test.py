# pages/3_Jeu de test.py
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import re
import plotly.figure_factory as ff

# ---- Configuration Streamlit
st.set_page_config(page_title="Jeu de test (3D complet)", page_icon="🧊", layout="wide")
st.title("🧊 L'intérêt du jeu de test !")

# ---- Paramètres
DATA_PATH = "data/minecraft_70cube.csv"  # adapte si ton chemin diffère
MIN_ALPHA_FOR_ZERO = 0.2  # visibilité minimale si AA==00 lorsqu'on ne veut pas d'alpha nul invisible
RANDOM_SEED = 42          # graine pour reproductibilité de l'index aléatoire

# ---- Chargement du CSV
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    # Lecture tolérante: détecte le séparateur automatiquement
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # Harmonisation d'éventuels alias
    rename_map = {
        "x_coord": "x", "y_coord": "y", "z_coord": "z",
        "xs": "x", "ys": "y", "zs": "z"
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Conversion numérique (gère décimales ",")
    def _coerce_num(s):
        s = str(s).strip()
        if s.lower() in ("nan", "", "none"):
            return np.nan
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        s = re.sub(r"\s", "", s)
        try:
            return float(s)
        except Exception:
            return np.nan

    for c in ("x", "y", "z"):
        if c in df.columns:
            df[c] = df[c].map(_coerce_num)

    if "couleur" in df.columns:
        df["couleur"] = df["couleur"].astype(str)

    missing = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}. Colonnes trouvées: {list(df.columns)}")

    df = df.dropna(subset=["x", "y", "z"])
    return df


# ---- Normalisation des couleurs
def _normalize_hex_or_rgba(val: str, force_min_alpha: bool = True) -> str:
    """
    Convertit les notations de couleur en formats acceptés par Plotly:
      - #RRGGBB -> '#RRGGBB'
      - #RRGGBBAA -> 'rgba(r,g,b,a)'
      - #RGB -> '#RRGGBB'
      - #RGBA -> 'rgba(r,g,b,a)'
      - 'RRGGBB' / 'RRGGBBAA' (sans '#') -> géré
      - valeurs vides/NaN -> '#888888'
    """
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return "#888888"
    s = s.replace(" ", "")
    if not s.startswith("#"):
        s = "#" + s

    # #RRGGBB
    if len(s) == 7:
        return s

    # #RRGGBBAA -> rgba
    if len(s) == 9:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        a_raw = int(s[7:9], 16) / 255.0
        if force_min_alpha and a_raw == 0:
            a_raw = MIN_ALPHA_FOR_ZERO
        return f"rgba({r},{g},{b},{a_raw:.3f})"

    # #RGB -> #RRGGBB
    if len(s) == 4:
        r, g, b = s[1] * 2, s[2] * 2, s[3] * 2
        return f"#{r}{g}{b}"

    # #RGBA -> rgba
    if len(s) == 5:
        r = int(s[1] * 2, 16)
        g = int(s[2] * 2, 16)
        b = int(s[3] * 2, 16)
        a_raw = int(s[4] * 2, 16) / 255.0
        if force_min_alpha and a_raw == 0:
            a_raw = MIN_ALPHA_FOR_ZERO
        return f"rgba({r},{g},{b},{a_raw:.3f})"

    return "#888888"


# ---- Chargement du fichier
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Fichier introuvable : `{DATA_PATH}`. Place le CSV dans le dossier `data/`.")
    st.stop()

with st.spinner("Chargement des données…"):
    df = load_data(DATA_PATH)

# === (NOUVEAU) Inversion des axes Y et Z ===
# On échange les valeurs de y et z sans toucher aux noms d'axes utilisés plus loin
_y = df["y"].copy()
df["y"] = df["z"]
df["z"] = _y

# === (NOUVEAU) Index incrémental randomisé pour l'échantillonnage ===
# sample_idx est une permutation aléatoire de 1..len(df), stable via RANDOM_SEED
rng = np.random.default_rng(RANDOM_SEED)
df["sample_idx"] = rng.permutation(np.arange(1, len(df) + 1))

# ---- Nettoyage et normalisation couleurs
if "couleur" in df.columns:
    df["couleur"] = df["couleur"].apply(_normalize_hex_or_rgba)
else:
    df["couleur"] = "#888888"

# ---- Application alpha = 0.1 sur bleu ciel (#87CEEB)
def _set_alpha_on_skyblue(color: str) -> str:
    """Transforme le bleu ciel #87CEEB en rgba avec alpha=0.1"""
    s = str(color).strip().lower()
    if s in ("#87ceeb", "87ceeb"):
        return "rgba(135,206,235,0.1)"  # bleu ciel transparent
    return color

df["couleur"] = df["couleur"].map(_set_alpha_on_skyblue)
df = df[df["type_bloc"].astype(str).str.lower() != "ciel"]

# ---------------------------------------------
# 1. Le monde réel
# ---------------------------------------------
st.header("1. Le monde réel")

# ---- Infos et affichage
n_total = len(df)

# ---- Figure Plotly 3D
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        marker=dict(
            size=6,                   
            color=df["couleur"],      
            opacity=1.0        
        ),
        hoverinfo="text",
        text=(
            "x=" + df["x"].round(3).astype(str) +
            "<br>y=" + df["y"].round(3).astype(str) +
            "<br>z=" + df["z"].round(3).astype(str) +
            "<br>sample_idx=" + df["sample_idx"].astype(int).astype(str) +
            (("<br>type=" + df["type_bloc"].astype(str)) if "type_bloc" in df.columns else "")
        ),
        name="Points"
    )
)

fig.update_layout(
    template="ggplot2",
    title=f"Nuage de points 3D — {os.path.basename(DATA_PATH)}",
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
        aspectmode="cube",
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=10, r=10, t=60, b=10),
    height=600
)

st.plotly_chart(fig, width='stretch', key="minecraft_3d_allpoints")

with st.expander("🔎 Aperçu du DataFrame (50 premières lignes)"):
    st.caption(f"{n_total:,} points chargés — colonnes: {list(df.columns)}")
    st.dataframe(df[["x", "y", "z", "type_bloc"]].head(50), width='stretch')


# ---------------------------------------------
# 2. L'espace d'échantillonnage
# ---------------------------------------------
st.header("2. L'espace d'échantillonnage")

# UI: slider + bouton
col1, col2 = st.columns([3, 1])
with col1:
    pct = st.slider("Taille de l'échantillon (%)", min_value=0, max_value=100, value=5, step=1)
with col2:
    trigger = st.button("Échantillonnage", use_container_width=True)

# État pour n'afficher/mettre à jour que sur clic
if "sample_pct" not in st.session_state:
    st.session_state.sample_pct = None
if "sample_df" not in st.session_state:
    st.session_state.sample_df = None

if trigger:
    k = int(round(pct / 100 * n_total))
    # Filtre par sample_idx (1..n) => on garde les k premiers de l'ordre randomisé
    sample_df = df[df["sample_idx"] <= max(k, 0)].copy()
    st.session_state.sample_pct = pct
    st.session_state.sample_df = sample_df

# Affichage des résultats si un échantillon existe
if st.session_state.sample_df is not None:
    sample_df = st.session_state.sample_df
    pct_saved = st.session_state.sample_pct
    n_sample = len(sample_df)

    # 3D des échantillons
    fig_s = go.Figure()
    fig_s.add_trace(
        go.Scatter3d(
            x=sample_df["x"], y=sample_df["y"], z=sample_df["z"],
            mode="markers",
            marker=dict(size=4, color=sample_df["couleur"], opacity=1.0),
            hoverinfo="text",
            text=(
                "x=" + sample_df["x"].round(3).astype(str) +
                "<br>y=" + sample_df["y"].round(3).astype(str) +
                "<br>z=" + sample_df["z"].round(3).astype(str) +
                "<br>sample_idx=" + sample_df["sample_idx"].astype(int).astype(str) +
                (("<br>type=" + sample_df["type_bloc"].astype(str)) if "type_bloc" in sample_df.columns else "")
            ),
            name="Échantillon"
        )
    )
    fig_s.update_layout(
        template="ggplot2",
        title="Nuage 3D — Échantillon",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=600
    )
    st.plotly_chart(fig_s, use_container_width=True, key="minecraft_3d_sample")

    with st.expander("🔎 Aperçu de l'échantillon (50 premières lignes)"):
        st.caption(f"🎯 Échantillon courant : **{pct_saved}%** → **{n_sample:,} / {n_total:,}** points")
        cols = ["x", "y", "z", "sample_idx"] + (["type_bloc"] if "type_bloc" in sample_df.columns else [])
        st.dataframe(sample_df[cols].head(50), use_container_width=True)
else:
    # st.info("Choisis un pourcentage puis clique sur **Échantillonnage** pour afficher le nuage 3D des points échantillonnés.")
    lapin = 1

# ---------------------------------------------
# 3. Split Train/Test sur l'échantillon
# ---------------------------------------------
st.header("3. Split Train/Test")

# Base de split : l'échantillon si présent, sinon le dataset complet
base_df = st.session_state.sample_df if st.session_state.get("sample_df") is not None else df
n_base = len(base_df)

# Slider proportion Train (%)
test_pct = st.slider("Proportion du **Test set** (%)", min_value=0, max_value=100, value=20, step=1)
train_pct = 100 - test_pct 

# Conversion en nombre de points
k_train = int(round(train_pct / 100 * n_base))

# Ordre aléatoire stable via sample_idx : les k premiers -> Train, le reste -> Test
# (sample_idx est une permutation 1..N : on s’en sert comme ordre aléatoire reproductible)
ordered = base_df.sort_values("sample_idx")
train_df = ordered.iloc[:k_train]
test_df = ordered.iloc[k_train:]

# Comptes
n_train, n_test = len(train_df), len(test_df)
st.caption(f"Répartition → **Train**: {n_train:,} | **Test**: {n_test:,} (sur {n_base:,} points)")

# Figure 3D avec 2 traces : Train (bleu alpha 0.5) et Test (orange)
fig_split = go.Figure()

# Train en bleu semi-transparent
fig_split.add_trace(
    go.Scatter3d(
        x=train_df["x"], y=train_df["y"], z=train_df["z"],
        mode="markers",
        marker=dict(size=4, color="rgba(31,119,180,0.5)"),  # bleu, alpha=0.5
        hoverinfo="text",
        text=(
            "x=" + train_df["x"].round(3).astype(str) +
            "<br>y=" + train_df["y"].round(3).astype(str) +
            "<br>z=" + train_df["z"].round(3).astype(str) +
            "<br>set=Train"
        ),
        name="Train"
    )
)

# Test en orange (opaque)
fig_split.add_trace(
    go.Scatter3d(
        x=test_df["x"], y=test_df["y"], z=test_df["z"],
        mode="markers",
        marker=dict(size=4, color="rgba(255,127,14,1.0)"),  # orange
        hoverinfo="text",
        text=(
            "x=" + test_df["x"].round(3).astype(str) +
            "<br>y=" + test_df["y"].round(3).astype(str) +
            "<br>z=" + test_df["z"].round(3).astype(str) +
            "<br>set=Test"
        ),
        name="Test"
    )
)

fig_split.update_layout(
    template="ggplot2",
    title="Nuage 3D — Split Train/Test (basé sur l'échantillon courant ou le dataset complet)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=60, b=10),
    height=600
)

st.plotly_chart(fig_split, use_container_width=True, key="minecraft_3d_train_test")
