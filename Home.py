import os
import streamlit as st

st.set_page_config(page_title="Paramètres & Hyperparamètres", page_icon="📈", layout="wide")

st.title("📈 Comprendre l'apprentissage, les paramètres et les hyperparamètres")
st.markdown(
    """
    Bienvenue ! Cette application *multipages* illustre :
    - la **régression linéaire** optimisée par **descente de gradient**,
    - l'effet des **hyperparamètres** (taux d’apprentissage, nombre d’epochs),
    - un **jeu de test** pour manipuler les paramètres du modèle.

    👉 Utilisez le menu de gauche pour naviguer :
    - **Les hyperparamètres** : animation Plotly + surface de coût,
    - **Jeu de test** : ajustez la pente et l'ordonnée et observez la MSE.
    """
)

with st.expander("ℹ️ Conseils d'utilisation"):
    st.write(
        """
        - Sur **Les hyperparamètres**, placez les curseurs (*learning rate* et epochs) puis cliquez
          sur **Lancer la simulation** pour recalculer la trajectoire d’optimisation.
        - La figure de gauche montre les **données** et l’**évolution de la droite**.
        - La figure de droite montre la **surface d’erreur (MSE)** et la **trajectoire**.
        - Sur **Jeu de test**, ajustez `a` et `b` à la main pour voir l’impact sur la MSE.
        """
    )

st.divider()
st.subheader("Contenu pédagogique")
st.write(
    """
    - **Paramètres** du modèle : `a` (pente) et `b` (ordonnée à l'origine).
    - **Hyperparamètres** : **learning rate** (pas d'apprentissage) et **nombre d'epochs**.
    - **Objectif** : minimiser la **MSE** (Mean Squared Error) sur l'ensemble d'apprentissage.
    """
)

# (Optionnel) Illustration si disponible
img_path = "/mnt/data/53eace4e-c830-42a1-89bf-56a77d2e83d5.png"
if os.path.exists(img_path):
    st.image(img_path, caption="Illustration: optimisation et surface d'erreur", use_column_width=True)
