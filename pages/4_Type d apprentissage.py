# pages/4_Type apprentissage.py
# -*- coding: utf-8 -*-
"""
Nouvelle page Streamlit "4_Type apprentissage"
- Dataset attendu : data/Dataset_CPV.csv
- 4 onglets : Classification (Conformity C/NC), Régression (dissolution), PCA (3 composantes), Clustering (k=3).
- Pour chaque onglet : panneau de configuration d'un nouveau lot pour prédire/positionner.
- Modèles : XGBoost pour classification & régression.

Mises à jour (version simplifiée) :
- Plus d'exposition des paramètres d'entraînement (supprimés).
- Résultats d'entraînement (matrice de confusion + Accuracy & ROC AUC) uniquement dans un expander fermé.
- "Nouvelles données" limité à un sous-ensemble de variables (sliders) ; par défaut = moyenne par Produit.
- Pour toutes les autres variables non exposées : remplissage automatique par moyenne (ou mode) par Produit.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import qualitative

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# XGBoost (CPU)
from xgboost import XGBClassifier, XGBRegressor

# ------------------------- Config -------------------------
st.set_page_config(page_title="Type d'apprentissage", page_icon="🧰", layout="wide")
st.title("🧰 Les différents types d'apprentissage")

DATA_PATH = os.path.join("data", "Dataset_CPV.csv")
RANDOM_STATE = 42

# ------------------------- Chargement -------------------------
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Impossible de charger le dataset : {e}")
    st.stop()

with st.expander("🔎 Aperçu des données", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ------------------------- Utilitaires -------------------------
def rmse_score(y_true, y_pred):
    # compatible avec anciennes et nouvelles versions de scikit-learn
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
def _detect_target(df: pd.DataFrame, candidates: list):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _detect_product_col(df: pd.DataFrame):
    return _detect_target(df, ["Produit", "Product", "produit", "product"])

TARGET_CLASS = _detect_target(df, ["Conformity", "conformity", "CONFIRMITY"])
TARGET_REGR  = _detect_target(df, ["dissolution", "Dissolution"])
PRODUCT_COL  = _detect_product_col(df)

num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {TARGET_REGR}]
cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in {TARGET_CLASS}]
if TARGET_CLASS in cat_cols:
    cat_cols.remove(TARGET_CLASS)

# -------- Variables exposées dans "Nouvelles données" (sliders) --------
EXPOSED_NUMS = [  # affichées seulement si présentes
    "api_d90", "starch_water", "SREL", "api_content",
    "api_d50", "api_water", "compforce_mean", "api_d10"
]

def product_defaults(df: pd.DataFrame, product_col: str | None, product_value: str | None):
    """
    Retourne (num_mean, cat_mode) dictionnaires :
    - Si product_col/value fournis et valides -> moyens/modes sur le sous-ensemble du produit.
    - Sinon -> moyens/modes globaux.
    """
    if product_col and product_value and product_col in df.columns:
        sub = df[df[product_col].astype(str) == str(product_value)]
        if len(sub) == 0:
            sub = df
    else:
        sub = df

    num_mean = sub[num_cols].mean(numeric_only=True).to_dict() if len(num_cols) else {}
    cat_mode = {}
    for c in cat_cols:
        if c in sub.columns and sub[c].notna().any():
            cat_mode[c] = sub[c].mode(dropna=True).iloc[0]
        else:
            # fallback global
            if c in df.columns and df[c].notna().any():
                cat_mode[c] = df[c].mode(dropna=True).iloc[0]
            else:
                cat_mode[c] = ""
    return num_mean, cat_mode

def make_new_lot_form_reduced(key_prefix: str, df: pd.DataFrame):
    """
    Formulaire 'Nouvelles données' réduit :
    - Sélection du Produit (si colonne détectée)
    - Sliders uniquement pour EXPOSED_NUMS existantes
    - Pour tous les autres paramètres : remplissage auto à la moyenne/mode par Produit
    Retourne un dict complet pour toutes les features (num + cat).
    """
    with st.form(f"new_lot_form_{key_prefix}"):
        st.subheader("Effectuer une prédiction pour un nouveau lot")

        # Choix du produit (si dispo)
        product_value = None
        if PRODUCT_COL:
            choices = sorted([str(x) for x in df[PRODUCT_COL].dropna().unique().tolist()])
            default_idx = 0 if choices else None
            product_value = st.selectbox(
                "Produit", options=choices, index=default_idx, key=f"{key_prefix}_prod"
            ) if choices else None

        # Valeurs par défaut par produit
        num_mean, cat_mode = product_defaults(df, PRODUCT_COL, product_value)

        # Pré-remplit toutes les colonnes avec moyennes/modes par produit
        values = {}
        for c in num_cols:
            values[c] = float(num_mean.get(c, df[c].mean())) if c in df.columns else 0.0
        for c in cat_cols:
            values[c] = str(cat_mode.get(c, ""))

        # Ordre personnalisé des sliders
        slider_order = [
            "api_d10",
            "api_d50",
            "api_d90",
            "api_water",
            "api_content",
            "starch_water",
            "SREL",
            "compforce_mean"
        ]

        # Filtrage selon disponibilité réelle
        exposed_present = [c for c in slider_order if c in df.columns and c in num_cols]

        cols = st.columns(2) if len(exposed_present) > 1 else [st]
        for i, c in enumerate(exposed_present):
            series = df[c].dropna().astype(float)
            if series.empty:
                continue
            vmin, vmax = float(series.min()), float(series.max())
            if np.isclose(vmin, vmax):
                vmin = vmin * 0.9 if vmin != 0 else -1.0
                vmax = vmax * 1.1 if vmax != 0 else 1.0
            default = float(num_mean.get(c, series.mean()))
            step = (vmax - vmin) / 100.0 if vmax > vmin else 0.01
            with cols[i % len(cols)]:
                values[c] = st.slider(
                    c, min_value=vmin, max_value=vmax, value=default, step=step, key=f"{key_prefix}_num_{c}"
                )

        submitted = st.form_submit_button("Prédire / Positionner")

    return values if submitted else None


# ========================= Onglets ========================= #
TABS = st.tabs(["Classification", "Régression", "Réduction de dimension", "Clustering"])

# =========================================================== #
# 1) CLASSIFICATION — Conformity (C/NC)
# =========================================================== #
with TABS[0]:
    st.markdown("#### Objectif : Prédire si un nouveau lot est conforme ou non-conforme")

    if TARGET_CLASS is None:
        st.warning("Colonne cible **Conformity** introuvable. Assure-toi qu'elle existe (valeurs C/NC).")
    else:
        # y binaire
        y_raw = df[TARGET_CLASS].astype(str).str.upper().str.strip()
        y = y_raw.map({"C": 1, "NC": 0})
        if y.isna().any():
            st.warning("Certaines valeurs de Conformity ne sont pas C/NC — elles seront ignorées.")
        valid_idx = y.dropna().index
        X = df.loc[valid_idx, num_cols + cat_cols]
        y = y.loc[valid_idx].astype(int)

        # Pipelines
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        prepro = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])

        # Modèle XGBoost figé
        test_size = 0.2
        clf = Pipeline([
            ("prepro", prepro),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=-1,
            )),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        with st.spinner("Entraînement du modèle de classification…"):
            clf.fit(X_train, y_train)

        # Évaluation simple
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba, auc = None, None

        # Formulaire pour un nouveau lot
        new_vals = make_new_lot_form_reduced("clf", df)
        if new_vals is not None:
            X_new = pd.DataFrame([new_vals])
            try:
                pred = clf.predict(X_new)[0]
                proba = clf.predict_proba(X_new)[0, 1] if hasattr(clf[-1], "predict_proba") else None
                label = "C" if int(pred) == 1 else "NC"

                # ✅ Affichage conditionnel selon la conformité
                if label == "C":
                    if proba is not None:
                        st.success(f"🧪 Prédiction nouveau lot : **Conforme (C)** — probabilité = {proba:.2f}")
                    else:
                        st.success(f"🧪 Prédiction nouveau lot : **Conforme (C)**")
                else:
                    if proba is not None:
                        st.error(f"⚠️ Prédiction nouveau lot : **Non conforme (NC)** — probabilité = {proba:.2f}")
                    else:
                        st.error(f"⚠️ Prédiction nouveau lot : **Non conforme (NC)**")

            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# =========================================================== #
# 2) RÉGRESSION — dissolution
# =========================================================== #
with TABS[1]:
    st.markdown("#### Objectif : prédire **dissolution** (régression XGBoost)")

    if TARGET_REGR is None:
        st.warning("Colonne cible **dissolution** introuvable.")
    else:
        y = pd.to_numeric(df[TARGET_REGR], errors="coerce")
        valid_idx = y.dropna().index
        X = df.loc[valid_idx, num_cols + cat_cols]
        y = y.loc[valid_idx]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        prepro = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])

        # Paramètres figés (pas d'exposition)
        reg = Pipeline([
            ("prepro", prepro),
            ("model", XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=-1,
            )),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        with st.spinner("Entraînement du modèle de régression…"):
            reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        rmse = rmse_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{rmse:.3f}")
        m2.metric("MAE", f"{mae:.3f}")
        m3.metric("R²", f"{r2:.3f}")

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Points"))
        mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Idéal", line=dict(dash="dash")))
        fig_sc.update_layout(title="Vrai vs Prédit", xaxis_title="Vrai", yaxis_title="Prédit",
                             height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.divider()
        # Fenêtre "Nouvelles données" réduite + moyennes par Produit
        new_vals = make_new_lot_form_reduced("reg", df)
        if new_vals is not None:
            X_new = pd.DataFrame([new_vals])
            try:
                y_new = float(reg.predict(X_new)[0])
                st.success(f"🧪 Prédiction nouveau lot — dissolution : **{y_new:.3f}**")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# =========================================================== #
# 3) PCA — 3 composantes + visu 3D
# =========================================================== #
with TABS[2]:
    st.markdown("#### Objectif : projeter les données sur **3 composantes principales (PCA)**")

    if len(num_cols) < 3:
        st.warning("Moins de 3 variables numériques disponibles pour une PCA 3D.")
    else:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd

        # --- helpers de détection de colonnes ---
        def _find_col(candidates, df_cols):
            for c in df_cols:
                cl = c.lower()
                if any(pat in cl for pat in candidates):
                    return c
            return None

        dissolution_col = _find_col(
            ["dissol", "dissolution", "%diss", "dissolution_%", "diss_%"],
            df.columns
        )
        conformity_col = _find_col(
            ["conform", "conformité", "compliance", "status", "qc_pass", "conformity"],
            df.columns
        )

        # --- Imputation + standardisation systématiques ---
        Xnum = df[num_cols].copy()
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_imp = pd.DataFrame(imputer.fit_transform(Xnum), columns=num_cols)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=num_cols)

        # --- PCA 3D ---
        pca = PCA(n_components=3, random_state=RANDOM_STATE)
        comps = pca.fit_transform(X_scaled)
        evr = pca.explained_variance_ratio_
        pct = evr * 100.0
        evr_sum = float(np.sum(evr[:3]))

        # --- UI : activer/désactiver la coloration dissolution (défaut = Non) ---
        show_diss = st.checkbox("Colorer selon la dissolution (rouge → vert)", value=False)

        # --- Couleurs par dissolution : RdYlGn (faible=rouge -> fort=vert) ---
        colors = None
        colorbar_cfg = None
        colorscale = "RdYlGn"
        reversescale = False

        if show_diss:
            if dissolution_col is None:
                st.info("Aucune colonne de dissolution détectée — coloration désactivée.")
            elif not pd.api.types.is_numeric_dtype(df[dissolution_col]):
                st.info(f"Colonne '{dissolution_col}' non numérique — coloration désactivée.")
            else:
                colors = df[dissolution_col].values
                colorbar_cfg = dict(title=f"{dissolution_col}")

        # --- Mapping conformité pour overlay (vert=conforme, rouge=non conforme) ---
        mask_conforme = mask_non = None
        if conformity_col is not None:
            raw = df[conformity_col]

            def to_bool(x):
                if pd.isna(x): return None
                if isinstance(x, (int, np.integer)): return bool(x)
                s = str(x).strip().lower()
                if s in {"1","true","vrai","yes","oui","ok","conforme","pass","passed","valid"}: return True
                if s in {"0","false","faux","no","non","nok","non conforme","fail","failed","invalid"}: return False
                return None

            mapped = raw.map(to_bool)
            if mapped.notna().any():
                mask_conforme = mapped == True
                mask_non = mapped == False
            else:
                conformity_col = None  # valeurs non reconnues

        # --- Figure 3D ---
        fig = go.Figure()

        base_marker = dict(size=5, opacity=0.9)
        if colors is not None:
            base_marker.update(dict(
                color=colors,
                colorscale=colorscale,
                reversescale=reversescale,
                showscale=True,
                colorbar=colorbar_cfg or dict(title="Dissolution")
            ))
        else:
            base_marker.update(color="rgba(120,120,120,0.7)")

        # Nuage principal
        fig.add_trace(go.Scatter3d(
            x=comps[:, 0], y=comps[:, 1], z=comps[:, 2],
            mode="markers",
            marker=base_marker,
            name="Lots"
        ))

        # Overlay conformité (anneaux verts/rouges)
        if conformity_col is not None:
            if mask_conforme is not None and mask_conforme.any():
                fig.add_trace(go.Scatter3d(
                    x=comps[mask_conforme, 0], y=comps[mask_conforme, 1], z=comps[mask_conforme, 2],
                    mode="markers",
                    marker=dict(size=9, symbol="circle-open", line=dict(color="green", width=2)),
                    name="Conforme",
                    hoverinfo="skip",
                ))
            if mask_non is not None and mask_non.any():
                fig.add_trace(go.Scatter3d(
                    x=comps[mask_non, 0], y=comps[mask_non, 1], z=comps[mask_non, 2],
                    mode="markers",
                    marker=dict(size=9, symbol="circle-open", line=dict(color="red", width=2)),
                    name="Non conforme",
                    hoverinfo="skip",
                ))

        # Titre : somme de variance expliquée + détail PC1..PC3
        fig.update_layout(
            title=f"Analyse en Composantes Principales — ∑ variance expliquée = {evr_sum:.1%}",
            scene=dict(
                xaxis_title=f"PC1 ({pct[0]:.1f}%)",
                yaxis_title=f"PC2 ({pct[1]:.1f}%)",
                zaxis_title=f"PC3 ({pct[2]:.1f}%)"
            ),
            height=600,
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)


        # st.divider()
        # Formulaire nouveau lot + projection avec les MÊMES imputer/scaler/pca
        # new_vals = make_new_lot_form_reduced("pca", df)  # sliders limités + moyennes par Produit
        # if new_vals is not None:
        #     try:
        #         x_row = pd.DataFrame([new_vals])
        #         x_row_imp = pd.DataFrame(imputer.transform(x_row), columns=num_cols)
        #         x_row_scaled = pd.DataFrame(scaler.transform(x_row_imp), columns=num_cols)
        #         pt = pca.transform(x_row_scaled)[0]
        #         st.info(f"Coordonnées PCA nouveau lot → PC1={pt[0]:.3f}, PC2={pt[1]:.3f}, PC3={pt[2]:.3f}")

        #         fig2 = go.Figure(fig)
        #         fig2.add_trace(go.Scatter3d(
        #             x=[pt[0]], y=[pt[1]], z=[pt[2]],
        #             mode="markers",
        #             marker=dict(size=9, symbol="diamond", color="black", line=dict(width=1, color="black")),
        #             name="Nouveau lot"
        #         ))
        #         st.plotly_chart(fig2, use_container_width=True)
        #     except Exception as e:
        #         st.error(f"Erreur de projection PCA : {e}")

# =========================================================== #
# 4) CLUSTERING — KMeans (k=3) + couleurs sur PCA
# =========================================================== #
with TABS[3]:
    st.markdown("#### Objectif : Regrouper les lots de production en clusters cohérents")

    if len(num_cols) < 3:
        st.warning("Besoin d'au moins 3 variables numériques pour PCA/Clustering.")
    else:
        # Prétraitements systématiques
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xnum = df[num_cols].copy()
        X_imp = pd.DataFrame(imputer.fit_transform(Xnum), columns=num_cols)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=num_cols)

        # PCA 3D
        pca = PCA(n_components=3, random_state=RANDOM_STATE)
        Z = pca.fit_transform(X_scaled)

        

        # -------- Nouveau lot : réindexage strict sur num_cols --------
        new_vals = make_new_lot_form_reduced("clust", df)
        # Clustering
        k = st.slider("Nombre de clusters (k)", 2, 10, value=3)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        st.divider()

        # Silhouette optionnelle
        sil = np.nan
        if len(X_scaled) > k:
            try:
                sil = silhouette_score(X_scaled, labels)
            except Exception:
                pass
        # st.metric("Silhouette score", f"{sil:.3f}" if not np.isnan(sil) else "N/A")

        # Figure: une trace par cluster pour la légende "Cluster 1..n"
        palette = qualitative.Set1 * ((k // len(qualitative.Set1)) + 1)
        fig = go.Figure()
        for i in range(k):
            m = (labels == i)
            fig.add_trace(go.Scatter3d(
                x=Z[m, 0], y=Z[m, 1], z=Z[m, 2],
                mode="markers",
                marker=dict(size=5, color=palette[i]),
                name=f"Cluster {i+1}"
            ))

        # Centres noirs en 'x'
        centers_scaled = kmeans.cluster_centers_
        centers_pca = pca.transform(centers_scaled)
        fig.add_trace(go.Scatter3d(
            x=centers_pca[:, 0], y=centers_pca[:, 1], z=centers_pca[:, 2],
            mode="markers",
            marker=dict(size=2, symbol="x", color="black"),
            name="Centres"
        ))


        if new_vals is not None:
            try:
                x_row_all = pd.DataFrame([new_vals])

                # Conserver uniquement les colonnes attendues, dans le même ordre
                x_row_num = x_row_all.reindex(columns=num_cols)

                # Conversion en numérique (coercition si champs texte)
                for c in num_cols:
                    x_row_num[c] = pd.to_numeric(x_row_num[c], errors="coerce")

                # Même pipeline que pour le fit
                x_row_imp = pd.DataFrame(imputer.transform(x_row_num), columns=num_cols)
                x_row_scaled = pd.DataFrame(scaler.transform(x_row_imp), columns=num_cols)

                # Prédiction cluster + projection PCA
                pred_idx = int(kmeans.predict(x_row_scaled)[0])
                pred_label = pred_idx + 1
                pt = pca.transform(x_row_scaled)[0]

                # Info demandée entre divider et graphique
                st.info(f"🧪 Nouveau lot prédit → **Cluster {pred_label}**")

                # Nouveau point en violet (diamond)
                fig.add_trace(go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]],
                    mode="markers",
                    marker=dict(size=9, color="Black", symbol="diamond"),
                    name="Nouveau lot"
                ))
            except Exception as e:
                st.error(f"Erreur de clustering : {e}")

        fig.update_layout(
            title="KMeans sur PCA 3D",
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            height=600, margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)



# ------------------------- Conseils/bonnes pratiques -------------------------
# st.info(
#     "💡 **Conseils** :\n"
#     "- Vérifie les types de colonnes (numériques vs catégorielles) et l'absence de fuites de données (data leakage).\n"
#     "- Utilise un *jeu de test* externe pour l'évaluation finale.\n"
#     "- En production, conserve les objets entraînés (pipeline, scaler, PCA, KMeans) pour des prédictions cohérentes."
# )
