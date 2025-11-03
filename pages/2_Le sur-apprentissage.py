# pages/2_Surapprentissage.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline

# ------------------------- Config & Titre -------------------------
st.set_page_config(page_title="Surapprentissage (Overfitting)", page_icon="🎯", layout="wide")
st.title("🎯 Le phénomène de surapprentissage")

degre_max = 10

# ------------------------- Données -------------------------
@st.cache_data
def make_dataset(n=40, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)

    # Deux "tops" (pics) et un "bottom" (creux)
    top1   = np.exp(-0.5 * ((x - 2.0) / 0.7)**2)          # pic à gauche
    top2   = 0.9 * np.exp(-0.5 * ((x - 8.0) / 0.9)**2)    # pic à droite
    bottom = 1.2 * np.exp(-0.5 * ((x - 5.0) / 1.1)**2)    # creux au centre

    y_true = 0.9*top1 + 0.8*top2 - 1.1*bottom             # 2 tops + 1 bottom
    y = y_true + rng.normal(0, 0.15, size=n)              # bruit léger

    return x, y, y_true

@st.cache_data
def make_split(n, train_ratio=0.7, seed=422):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_ratio*n)
    train_idx = np.sort(idx[:n_train])
    val_idx   = np.sort(idx[n_train:])
    return train_idx, val_idx

x_raw, y_raw, y_true = make_dataset()
train_idx, val_idx = make_split(len(x_raw))
x_train, y_train = x_raw[train_idx], y_raw[train_idx]
x_val,   y_val   = x_raw[val_idx],   y_raw[val_idx]

# ------------------------- Utils ML -------------------------
def fit_poly_predict(x_train, y_train, degree, x_grid):
    coefs = np.polyfit(x_train, y_train, deg=degree)
    y_hat_train = np.polyval(coefs, x_train)
    y_hat_grid  = np.polyval(coefs, x_grid)
    return coefs, y_hat_train, y_hat_grid

def mse(y, y_hat):
    e = y - y_hat
    return float(np.mean(e*e))

# ------------------------- État -------------------------
DEFAULTS = {
    "degree": 1,
    "frame_time": 0.5,
    "is_playing": False,
    "show_val": True,          # ✅ visible par défaut
    "show_spline_left": False  # ❌ non visible par défaut
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "mse_train" not in st.session_state:
    st.session_state.mse_train = {}
if "mse_val" not in st.session_state:
    st.session_state.mse_val = {}
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

def reset_all():
    st.session_state.degree = 2
    st.session_state.frame_time = DEFAULTS["frame_time"]
    st.session_state.is_playing = False
    st.session_state.mse_train = {}
    st.session_state.mse_val = {}
    # réinitialise aussi les nouveaux interrupteurs
    st.session_state.show_val = True
    st.session_state.show_spline_left = False
    st.session_state.run_id = st.session_state.get("run_id", 0) + 1

# ------------------------- Lecture du mode "Play" (AVANT widgets) -------------------------
autoplay_step = False
if st.session_state.is_playing:
    if st.session_state.degree < degre_max :
        st.session_state.degree += 1
        autoplay_step = True
    else:
        st.session_state.is_playing = False

# ------------------------- Contrôles -------------------------
if "show_val" not in st.session_state:
    st.session_state.show_val = True
if "show_spline_left" not in st.session_state:
    st.session_state.show_spline_left = False

with st.container(border=True):
    st.markdown("### Réglages")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.slider("Durée d'apprentissage", 1, degre_max , step=1, key="degree")
    with c2:
        st.slider("Temps par frame (s)", 0.1, 2.0, step=0.1, key="frame_time")

    # ---- Nouveaux interrupteurs d'affichage ----
    t1, t2 = st.columns([1, 1], gap="large")
    with t1:
        st.checkbox("Afficher le Tuning Set", key="show_val")
    with t2:
        st.checkbox("Afficher une fonction moyenne", key="show_spline_left")

    b1, b2 = st.columns(2)
    with b1:
        play_clicked = st.button("▶️ Play", use_container_width=True)
    with b2:
        st.button("🔄 Reset", use_container_width=True, on_click=reset_all)

    if play_clicked:
        st.session_state.is_playing = True
        st.rerun()

# ------------------------- Fonctions de dessin -------------------------
colL, colR = st.columns(2, gap="large")
left_ph = colL.empty()
right_ph = colR.empty()

def update_errors_for_degree(deg: int):
    if deg in st.session_state.mse_train and deg in st.session_state.mse_val:
        return
    coefs = np.polyfit(x_train, y_train, deg=deg)
    yhat_train = np.polyval(coefs, x_train)
    yhat_val   = np.polyval(coefs, x_val)
    st.session_state.mse_train[deg] = mse(y_train, yhat_train)
    st.session_state.mse_val[deg]   = mse(y_val,   yhat_val)

alpha = 0.05
def draw_left(deg: int):
    x_grid = np.linspace(x_raw.min(), x_raw.max(), 400)
    coefs, _, y_hat_grid = fit_poly_predict(x_train, y_train, deg, x_grid)

    fig = go.Figure()

    # Points Train
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train, mode="markers", name="Train",
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="Train<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"
    ))
    # Points Val (optionnels)
    if st.session_state.show_val:
        fig.add_trace(go.Scatter(
            x=x_val, y=y_val, mode="markers", name="Val",
            marker=dict(size=8, color="#ff7f0e"),
            hovertemplate="Val<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"
        ))

    if st.session_state.show_spline_left:
        # --- Spline lissée (non-interpolante) ---
        order = np.argsort(x_train)
        xs = x_train[order]
        ys = y_train[order]

        # Paramètre de lissage s = α * n * Var(y)
        n = len(xs)
        s = float(alpha) * n * float(np.var(ys))
        # k=3 = spline cubique ; s>0 -> lissage
        spl = UnivariateSpline(xs, ys, k=3, s=s)
        y_smooth = spl(x_grid)

        fig.add_trace(go.Scatter(
            x=x_grid, y=y_smooth, mode="lines", name="Spline (lissée)",
            line=dict(width=3, color="#2ca02c", dash="dot"),
            hoverinfo="skip"
        ))
    else:
        # --- Modèle polynomial (si spline masquée) ---
        fig.add_trace(go.Scatter(
            x=x_grid, y=y_hat_grid, mode="lines", name="Modèle polynomial",
            line=dict(width=3, color="#2ca02c"),
            hoverinfo="skip"
        ))

    fig.update_layout(
        template="ggplot2",
        title="Modélisation (régression polynomiale)",
        xaxis_title="X", yaxis_title="y",
        margin=dict(l=10, r=10, t=60, b=10), height=540,
        legend=dict(
            x=0.02, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0)"
        )
    )
    left_ph.plotly_chart(fig, use_container_width=True, theme=None,
                         key=f"left_{st.session_state.run_id}_{deg}")

def draw_right():
    if len(st.session_state.mse_train) == 0:
        degrees, tr, va = [], [], []
    else:
        degrees = sorted(set(st.session_state.mse_train.keys()) | set(st.session_state.mse_val.keys()))
        tr = [st.session_state.mse_train[d] for d in degrees]
        va = [st.session_state.mse_val[d]   for d in degrees]

    fig = go.Figure()
    # Courbe Train (toujours affichée)
    fig.add_trace(go.Scatter(
        x=degrees, y=tr, mode="lines+markers", name="Train error",
        line=dict(width=3, color="#1f77b4")
    ))
    # Courbe Val (affichable/masquable)
    if st.session_state.show_val:
        fig.add_trace(go.Scatter(
            x=degrees, y=va, mode="lines+markers", name="Val error",
            line=dict(width=3, color="#ff7f0e")
        ))
    fig.update_layout(
        template="ggplot2",
        title="Visualisation de l'erreur (Train vs Val) au cours du temps",
        xaxis_title="Durée d'apprentissage",
        yaxis_title="Erreur quadratique moyenne (MSE)",
        margin=dict(l=10, r=10, t=60, b=10), height=540,
        legend=dict(
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0)"
        )
    )
    right_ph.plotly_chart(
        fig, use_container_width=True, theme=None,
        key=f"right_{st.session_state.run_id}_{len(degrees)}"
    )

# ------------------------- Rendu -------------------------
update_errors_for_degree(st.session_state.degree)
draw_left(st.session_state.degree)
draw_right()

# ------------------------- Animation (pilotée par rerun) -------------------------
if autoplay_step and st.session_state.is_playing:
    time.sleep(float(st.session_state.frame_time))
    st.rerun()
