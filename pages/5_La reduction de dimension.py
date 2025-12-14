import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------- Config -------------------------
st.set_page_config(page_title="Réduction de dimension", page_icon="📐", layout="wide")
st.title("📐 Comprendre la réduction de dimension")

# ------------------------- Fonctions de chargement -------------------------
@st.cache_data(show_spinner=True)
def load_csv_tolerant(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=True)
def load_milo(path: str) -> pd.DataFrame:
    df = load_csv_tolerant(path)
    # Normalisation noms de colonnes
    lower = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=lower, inplace=True)
    # Aliases vers x,y,z
    rename_map = {"x_coord": "x", "y_coord": "y", "z_coord": "z", "xs": "x", "ys": "y", "zs": "z"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Conversion numérique robuste (gère virgules décimales, espaces)
    def _to_float(val):
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
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
            df[c] = df[c].map(_to_float)
    missing = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour milo.csv : {missing}")

    df = df.dropna(subset=["x", "y", "z"]).copy()
    return df

# ------------------------- 1.Partie 1 : Jeu - De 3D à 2D -------------------------
st.header("1. Exemple illustré de la notion de composante")

milo_path = os.path.join("data", "odin.csv")
try:
    milo = load_milo(milo_path)
except Exception as e:
    st.error(f"❌ Impossible de charger milo.csv — {e}")
    st.stop()

# Caméra codée en dur : vue par dessous (Z côté opposé à l'œil)
# Calcul dynamique du centre et d'une distance d'observation d'après l'enveloppe des données
x0, x1 = float(milo["x"].min()), float(milo["x"].max())
y0, y1 = float(milo["y"].min()), float(milo["y"].max())
z0, z1 = float(milo["z"].min()), float(milo["z"].max())
cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
span = max(x1 - x0, y1 - y0, z1 - z0)
d = 1.8 * (span if span > 0 else 1.0)

camera_bottom = dict(
    eye=dict(x=cx, y=cy, z=cz - 2*d),  # œil sous l'objet
    center=dict(x=cx, y=cy, z=cz),
    up=dict(x=0, y=1, z=0)  # Y vertical vers le haut
)

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=milo["x"], y=milo["y"], z=milo["z"],
    mode="markers",
    marker=dict(size=1, color="#888888", opacity=0.95),
    name="Points"
))

fig3d.update_layout(
    template="ggplot2",
    title="Nuage 3D — Vue par dessous (X horizontal, Y vertical, Z masqué)",
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        zaxis=dict(showgrid=False, zeroline=False, visible=False),  # Z invisible
        camera=camera_bottom,
        aspectmode="data",
    ),
    margin=dict(l=10, r=10, t=60, b=10),
    height=640,
)

st.plotly_chart(fig3d, use_container_width=True, theme=None, key="milo_3d_view")

# ------------------------- 2. Réduction de dimension — Exemple -------------------------
st.header("2. Réduction de dimension 3D vers 2D")

# PCA sur milo.csv (x, y, z) -> 2 composantes
X_milo = milo[["x", "y", "z"]].to_numpy()
# X_milo_std = StandardScaler().fit_transform(X_milo)
pca_milo = PCA(n_components=2, random_state=42)
X_milo_pca = pca_milo.fit_transform(X_milo)#_std)

# DataFrame des composantes
milo_pca_df = pd.DataFrame(X_milo_pca, columns=["PC1", "PC2"])

# Nuage 2D PC1 vs PC2 avec variance expliquée dans les labels d'axes
exp_var_milo = pca_milo.explained_variance_ratio_
fig_scatter_milo = go.Figure()
fig_scatter_milo.add_trace(go.Scatter(
    x=milo_pca_df["PC1"],
    y=milo_pca_df["PC2"],
    mode="markers",
    marker=dict(size=1, opacity=0.8),
    name="PC scores"
))
pourcentage_var = round(exp_var_milo[0]*100 + exp_var_milo[1]*100, 1)
fig_scatter_milo.update_layout(
    title="Projection PCA de 3D vers 2D (" + str(pourcentage_var) + "% var. expliquée)",
    xaxis_title=f"PC1 ({exp_var_milo[0]*100:.1f}% var. expliquée)",
    yaxis_title=f"PC2 ({exp_var_milo[1]*100:.1f}% var. expliquée)",
    margin=dict(l=10, r=10, t=60, b=10),
    height=500,
)
st.plotly_chart(fig_scatter_milo, use_container_width=True, theme=None)

with st.expander(label="Données brutes", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Jeu de données d'origine (3D)")
        st.dataframe(milo[["x", "y", "z"]].head(5), use_container_width=True)
    with c2:
        st.subheader("Jeu de données — Composantes principales (2D)")
        st.dataframe(milo_pca_df.head(5), use_container_width=True)


# ------------------------- 3. Exemple réel — diabetes.csv -------------------------
st.header("3. Exemple réel appliqué à la santé")

diab_path = os.path.join("data", "diabetes.csv")
try:
    diabetes = load_csv_tolerant(diab_path)
except Exception as e:
    st.error(f"❌ Impossible de charger diabetes.csv — {e}")
    st.stop()

# Colonnes numériques uniquement
numeric_cols = diabetes.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Le fichier diabetes.csv doit contenir au moins 2 colonnes numériques pour une PCA.")
    st.stop()

diab_clean = diabetes[numeric_cols].dropna().copy()

# Standardisation + PCA 2D
scaler = StandardScaler()
X_diab_std = scaler.fit_transform(diab_clean.values)
pca_diab = PCA(n_components=2, random_state=42)
X_diab_pca = pca_diab.fit_transform(X_diab_std)

diab_pca_df = pd.DataFrame(X_diab_pca, columns=["PC1", "PC2"])

# Nuage 2D PC1 vs PC2 avec variance expliquée dans les labels d'axes
exp_var_diab = pca_diab.explained_variance_ratio_
fig_scatter_diab = go.Figure()
fig_scatter_diab.add_trace(go.Scatter(
    x=diab_pca_df["PC1"],
    y=diab_pca_df["PC2"],
    mode="markers",
    marker=dict(size=7, opacity=0.8),
    name="PC scores"
))
pourcentage_var2 = round(exp_var_diab[0]*100 + exp_var_diab[1]*100,1)
fig_scatter_diab.update_layout(
    title="Projection PCA diabète (" + str(pourcentage_var2) + "% variances expliquée)",
    xaxis_title=f"PC1 ({exp_var_diab[0]*100:.1f}% var. expliquée)",
    yaxis_title=f"PC2 ({exp_var_diab[1]*100:.1f}% var. expliquée)",
    margin=dict(l=10, r=10, t=60, b=10),
    height=500,
)
st.plotly_chart(fig_scatter_diab, use_container_width=True, theme=None)

with st.expander(label="Données brutes", expanded=False):
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Dataset d'origine (8 variables réelles)")
        st.dataframe(diab_clean.head(5), use_container_width=True)
    with c4:
        st.subheader("Dataset PCA (2 variables synthétiques)")
        st.dataframe(diab_pca_df.head(5), use_container_width=True)

# ------------------------- 4. Exemple réel — diabetes.csv (t-SNE) -------------------------
st.header("4. Une alternative à la PCA, la t-SNE")

from sklearn.manifold import TSNE

diab_path = os.path.join("data", "diabetes.csv")
try:
    diabetes = load_csv_tolerant(diab_path)
except Exception as e:
    st.error(f"❌ Impossible de charger diabetes.csv — {e}")
    st.stop()

# Colonnes numériques uniquement
numeric_cols = diabetes.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Le fichier diabetes.csv doit contenir au moins 2 colonnes numériques pour une t-SNE.")
    st.stop()

diab_clean = diabetes[numeric_cols].dropna().copy()

# ---------------- Paramètres t-SNE ----------------
with st.expander("Paramètres t-SNE", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        perplexity = st.slider("Perplexity", min_value=5, max_value=100, value=30, step=1)
    with c2:
        n_iter = st.slider("Itérations (n_iter)", min_value=250, max_value=5000, value=1000, step=50)
    with c3:
        early_exaggeration = st.slider("Early exaggeration", min_value=4.0, max_value=20.0, value=12.0, step=0.5)

    c4, c5 = st.columns(2)
    with c4:
        learning_rate = st.selectbox("Learning rate", options=["auto", 10, 50, 100, 200, 500, 1000], index=0)
    with c5:
        metric = st.selectbox("Métrique", options=["euclidean", "manhattan", "cosine"], index=0)

# Standardisation
scaler = StandardScaler()
X_diab_std = scaler.fit_transform(diab_clean.values)

# Vérification perplexity < n_samples
n_samples = X_diab_std.shape[0]
if perplexity >= n_samples:
    st.error(f"La perplexity ({perplexity}) doit être strictement inférieure au nombre d'échantillons ({n_samples}).")
    st.stop()

# ---------------- t-SNE 2D ----------------
try:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        init="pca",
        metric=metric,
        random_state=42,
        verbose=0,
    )
    X_diab_tsne = tsne.fit_transform(X_diab_std)
except TypeError:
    # fallback pour compatibilité versions différentes de scikit-learn
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        init="pca",
        metric=metric,
        random_state=42,
    )
    X_diab_tsne = tsne.fit_transform(X_diab_std)

diab_tsne_df = pd.DataFrame(X_diab_tsne, columns=["tSNE1", "tSNE2"])

# ---------------- Nuage 2D ----------------
fig_scatter_diab_tsne = go.Figure()
fig_scatter_diab_tsne.add_trace(go.Scatter(
    x=diab_tsne_df["tSNE1"],
    y=diab_tsne_df["tSNE2"],
    mode="markers",
    marker=dict(size=7, opacity=0.85),
    name="t-SNE embedding"
))
fig_scatter_diab_tsne.update_layout(
    title=f"Projection t-SNE (perplexity={perplexity}, metric={metric})",
    xaxis_title="tSNE1",
    yaxis_title="tSNE2",
    margin=dict(l=10, r=10, t=60, b=10),
    height=500,
)
st.plotly_chart(fig_scatter_diab_tsne, use_container_width=True, theme=None)

with st.expander(label="Données brutes", expanded=False):
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Dataset d'origine (variables réelles)")
        st.dataframe(diab_clean.head(5), use_container_width=True)
    with c4:
        st.subheader("Embedding t-SNE (2 dimensions)")
        st.dataframe(diab_tsne_df.head(5), use_container_width=True)
