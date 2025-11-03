# pages/1_Les hyperparamètres.py
import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ------------------------- Config & Titre -------------------------
st.set_page_config(page_title="Les paramètres et les hyperparamètres", page_icon="🛠️", layout="wide")
st.title("🛠️ Les paramètres et les hyperparamètres")

# ------------------------- Données & Pré-traitements -------------------------
@st.cache_data
def make_dataset(n=50, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    true_a, true_b = 2.3, 3.5
    noise = rng.normal(0, 1.2, size=n)
    y = true_a * x + true_b + noise
    return x, y, true_a, true_b

x_raw, y_raw, true_a, true_b = make_dataset()
y_mean = float(np.mean(y_raw))
sst = float(np.sum((y_raw - y_mean) ** 2))

# Normalisation
x_mu, x_sigma = float(np.mean(x_raw)), float(np.std(x_raw))
y_mu, y_sigma = float(np.mean(y_raw)), float(np.std(y_raw))
x_norm = (x_raw - x_mu) / (x_sigma if x_sigma != 0 else 1.0)
y_norm = (y_raw - y_mu) / (y_sigma if y_sigma != 0 else 1.0)

# ------------------------- Utilitaires ML -------------------------
def mse_norm(a_n, b_n):
    y_pred_n = a_n * x_norm + b_n
    return float(np.mean((y_norm - y_pred_n) ** 2))

def gradients_norm(a_n, b_n):
    n = len(x_norm)
    y_pred_n = a_n * x_norm + b_n
    da = (-2 / n) * np.sum(x_norm * (y_norm - y_pred_n))
    db = (-2 / n) * np.sum((y_norm - y_pred_n))
    return float(da), float(db)

def to_raw(a_n, b_n):
    a_raw = (y_sigma / (x_sigma if x_sigma != 0 else 1.0)) * a_n
    b_raw = y_sigma * b_n + y_mu - a_raw * x_mu
    return float(a_raw), float(b_raw)

def to_norm(a_raw, b_raw):
    a_n = a_raw * (x_sigma if x_sigma != 0 else 1.0) / (y_sigma if y_sigma != 0 else 1.0)
    b_n = (a_raw * x_mu + b_raw - y_mu) / (y_sigma if y_sigma != 0 else 1.0)
    return float(a_n), float(b_n)

def r2_raw(a_raw, b_raw):
    y_hat = a_raw * x_raw + b_raw
    sse = float(np.sum((y_raw - y_hat) ** 2))
    return 1.0 - (sse / sst if sst != 0 else 0.0)

# ------------------------- Paramètres & État -------------------------
DEFAULTS = {"a0_raw": -3, "b0_raw": -5, "lr": 0.01, "epochs": 100, "frame_time": 0.5}

# Init session_state pour les contrôles
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Identifiant de run (évite toute collision d'IDs sur les charts)
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# Early stopping
EPSILON, PATIENCE = 0.01, 5

def simulate_gd(a0_raw, b0_raw, lr, max_epochs):
    a_n, b_n = to_norm(a0_raw, b0_raw)
    a_hist, b_hist, loss_hist = [a_n], [b_n], [mse_norm(a_n, b_n)]
    best_loss, bad_epochs = loss_hist[0], 0
    for _ in range(max_epochs):
        da, db = gradients_norm(a_n, b_n)
        a_n -= lr * da
        b_n -= lr * db
        a_hist.append(a_n)
        b_hist.append(b_n)
        current_loss = mse_norm(a_n, b_n)
        loss_hist.append(current_loss)
        if best_loss - current_loss >= EPSILON:
            best_loss = current_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    return a_hist, b_hist, loss_hist

# Init historiques si absents
if "a_hist" not in st.session_state:
    st.session_state.a_hist, st.session_state.b_hist, st.session_state.loss_hist = simulate_gd(
        st.session_state.a0_raw, st.session_state.b0_raw, st.session_state.lr, 0
    )

# ------------------------- Fonctions d'affichage -------------------------
def draw_left(a_raw, b_raw, epoch_label="0"):
    x_line = np.array([x_raw.min(), x_raw.max()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode="markers",
                             marker=dict(size=7, color="#1f77b4", opacity=0.9), name="Données"))
    fig.add_trace(go.Scatter(x=x_line, y=a_raw * x_line + b_raw, mode="lines",
                             line=dict(width=3, color="#d62728"), name="Modèle"))
    fig.update_layout(template="ggplot2",
                      title="Entraînement d'un modèle de Régression Linéaire",
                      xaxis_title="x", yaxis_title="y",
                      margin=dict(l=10, r=10, t=60, b=10), height=540,
                      legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, 
                                  bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
    fig.add_annotation(xref="paper", yref="paper", x=0.06, y=0.92,
                       text=f"Époque: {epoch_label}<br>a = {a_raw:.3f}<br>b = {b_raw:.3f}",
                       showarrow=False, bgcolor="white", bordercolor="#555",
                       borderwidth=1, font=dict(color="#111"))
    fig.add_annotation(xref="paper", yref="paper", x=0.06, y=0.78,
                       text=f"Équation:<br>y = {a_raw:.3f}x + {b_raw:.3f}",
                       showarrow=False, bgcolor="#e6f0ff", bordercolor="#4a78c2",
                       borderwidth=1, font=dict(color="#0b2e66"))
    r2 = r2_raw(a_raw, b_raw)
    fig.add_annotation(xref="paper", yref="paper", x=0.06, y=0.68,
                       text=f"R² = {r2:.3f}",
                       showarrow=False, bgcolor="#e7f6e7", bordercolor="#2e7d32",
                       borderwidth=1, font=dict(color="#1b5e20"))
    left_ph.plotly_chart(fig, use_container_width=True, theme=None,
                         key=f"left_{st.session_state.run_id}_{epoch_label}")

def draw_right(a_hist, b_hist, loss_hist, epoch_label="0"):
    a_grid = np.linspace(min(-2, min(a_hist)-0.5), max(3, max(a_hist)+0.5), 60)
    b_grid = np.linspace(min(-2, min(b_hist)-0.5), max(3, max(b_hist)+0.5), 60)
    ZZ = np.empty((len(b_grid), len(a_grid)))
    for j, b_n in enumerate(b_grid):
        y_pred_grid = np.outer(a_grid, x_norm) + b_n
        ZZ[j, :] = np.mean((y_norm - y_pred_grid) ** 2, axis=1)

    traj = np.array([[a_hist[i], b_hist[i], loss_hist[i]] for i in range(len(a_hist))])
    surface = go.Surface(x=a_grid, y=b_grid, z=ZZ, opacity=0.4,
                         colorscale="Viridis", showscale=False, name="Surface MSE")
    traj_line = go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2],
                             mode="lines+markers", marker=dict(color="red", size=3), name="Trajectoire",
                             line=dict(color="red", width=4))
    current = go.Scatter3d(x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                           mode="markers", marker=dict(color="red", size=3, symbol="diamond"),
                           name="Position actuelle")

    fig = go.Figure(data=[surface, traj_line, current])
    fig.update_layout(template="ggplot2",
                      title="Surface d'Erreur et Trajectoire d'Optimisation",
                      scene=dict(xaxis_title="Paramètre a (pente)",
                                 yaxis_title="Paramètre b (ordonnée)",
                                 zaxis_title="Erreur (MSE)"),
                      legend=dict(
                            x=0.98,
                            y=0.98,
                            xanchor="right",
                            yanchor="top",
                            bgcolor="rgba(255,255,255,0.6)",
                            bordercolor="rgba(0,0,0,0)",
                            orientation="v"       
                        ),
                        margin=dict(l=0, r=0, t=60, b=0))           
                    #   margin=dict(l=10, r=10, t=60, b=10), height=540)
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98,
                       text=f"Loss (MSE):<br>{loss_hist[-1]:.3f}",
                       showarrow=False, bgcolor="#ffd180", bordercolor="#ff8f00",
                       borderwidth=1, font=dict(color="#4e2a00"))

    right_ph.plotly_chart(fig, use_container_width=True, theme=None,
                          key=f"right_{st.session_state.run_id}_{epoch_label}")

# ------------------------- Callback Reset -------------------------
def reset_all():
    # Remise à zéro des contrôles
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # Réinitialise les historiques
    st.session_state.a_hist, st.session_state.b_hist, st.session_state.loss_hist = simulate_gd(
        DEFAULTS["a0_raw"], DEFAULTS["b0_raw"], DEFAULTS["lr"], 0
    )
    # Nouveau run_id pour éviter toute collision d'éléments Streamlit
    st.session_state.run_id = st.session_state.get("run_id", 0) + 1

# ------------------------- UI : Contrôles -------------------------
if "initialized" not in st.session_state:
    reset_all()  # remet toutes les valeurs par défaut
    st.session_state.initialized = True  # évite de relancer le reset à chaque interaction

with st.container(border=True):
    st.markdown("### Réglages")

    c1, c2 = st.columns(2)
    with c1:
        st.slider("a₀ (pente initiale)", -10, 10, step=1, key="a0_raw")
    with c2:
        st.slider("b₀ (ordonnée initiale)", -10, 10, step=1, key="b0_raw")

    c3, c4, c5 = st.columns(3)
    with c3:
        # Sélecteur discret des LR
        st.select_slider(
            "Learning rate (η)",
            options=[0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 5, 10, 100],
            key="lr"
        )
    with c4:
        st.slider("Nombre d'epochs (max)", 10, 200, step=10, key="epochs")
    with c5:
        st.slider("Temps par frame (s)", 0.25, 2.0, step=0.25, key="frame_time")

    c6, c7 = st.columns(2)
    with c6:
        start = st.button("▶️ Lancer l'animation", use_container_width=True)
    with c7:
        st.button("🔄 Reset", use_container_width=True, on_click=reset_all)

# ------------------------- Layout d'affichage -------------------------
colL, colR = st.columns(2, gap="large")
left_ph = colL.empty()
right_ph = colR.empty()

# ------------------------- Affichage initial -------------------------
draw_left(st.session_state.a0_raw, st.session_state.b0_raw, epoch_label="init")
draw_right(st.session_state.a_hist, st.session_state.b_hist, st.session_state.loss_hist, epoch_label="init")

# ------------------------- Animation -------------------------
if start:
    # Nouveau run pour isoler les clés des charts
    st.session_state.run_id += 1

    a_hist, b_hist, loss_hist = simulate_gd(
        st.session_state.a0_raw, st.session_state.b0_raw,
        st.session_state.lr, st.session_state.epochs
    )
    for i in range(len(a_hist)):
        a_raw_i, b_raw_i = to_raw(a_hist[i], b_hist[i])
        draw_left(a_raw_i, b_raw_i, epoch_label=str(i))
        draw_right(a_hist[:i+1], b_hist[:i+1], loss_hist[:i+1], epoch_label=str(i))
        time.sleep(float(st.session_state.frame_time))

    st.session_state.a_hist = a_hist
    st.session_state.b_hist = b_hist
    st.session_state.loss_hist = loss_hist
