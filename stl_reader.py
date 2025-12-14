from stl import mesh
import pandas as pd

# Charger le fichier STL
fichier_stl = "data\milo.stl"
maillage = mesh.Mesh.from_file(fichier_stl)

# Extraire les coordonnées des sommets
x = maillage.vectors[:, :, 0].flatten()
y = maillage.vectors[:, :, 1].flatten()
z = maillage.vectors[:, :, 2].flatten()

# Créer un DataFrame
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

# Supprimer les doublons éventuels
df = df.drop_duplicates()

# Sauvegarder en CSV
df.to_csv("data\milo.csv", index=False)
