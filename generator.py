# Minecraft-like 70x70x70 voxel world generator with Perlin noise terrain + caves
# Produces a CSV with columns: x, y, z, type_bloc, couleur

import math
import random
import numpy as np
import pandas as pd

# -----------------------------
# Classic "improved Perlin" noise (Ken Perlin 2002) – pure Python/Numpy
# -----------------------------
perm = np.array([151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
    49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254], dtype=np.int32)
p = np.concatenate([perm, perm])

def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
def lerp(a, b, t): return a + t * (b - a)
def grad(hash_, x, y, z):
    h = hash_ & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h==12) | (h==14), x, z))
    return np.where((h & 1)==0, u, -u) + np.where((h & 2)==0, v, -v)
def perlin3(x, y, z):
    xi = np.floor(x).astype(np.int32) & 255
    yi = np.floor(y).astype(np.int32) & 255
    zi = np.floor(z).astype(np.int32) & 255
    xf = x - np.floor(x); yf = y - np.floor(y); zf = z - np.floor(z)
    u = fade(xf); v = fade(yf); w = fade(zf)
    aaa = p[p[p[  xi ] +  yi ] +  zi ]
    aba = p[p[p[  xi ] + yi+1] +  zi ]
    aab = p[p[p[  xi ] +  yi ] + zi+1]
    abb = p[p[p[  xi ] + yi+1] + zi+1]
    baa = p[p[p[xi+1] +  yi ] +  zi ]
    bba = p[p[p[xi+1] + yi+1] +  zi ]
    bab = p[p[p[xi+1] +  yi ] + zi+1]
    bbb = p[p[p[xi+1] + yi+1] + zi+1]
    x1 = lerp(grad(aaa, xf,   yf,   zf),   grad(baa, xf-1, yf,   zf),   u)
    x2 = lerp(grad(aba, xf,   yf-1, zf),   grad(bba, xf-1, yf-1, zf),   u)
    y1 = lerp(x1, x2, v)
    x1 = lerp(grad(aab, xf,   yf,   zf-1), grad(bab, xf-1, yf,   zf-1), u)
    x2 = lerp(grad(abb, xf,   yf-1, zf-1), grad(bbb, xf-1, yf-1, zf-1), u)
    y2 = lerp(x1, x2, v)
    return (lerp(y1, y2, w) + 1) / 2
def perlin2(x, y): return perlin3(x, y, 0.0)
def fbm2(x, y, octaves=4, lacunarity=2.0, gain=0.5):
    total = np.zeros_like(x, dtype=float); amp=1.0; freq=1.0
    for _ in range(octaves):
        total += amp * perlin2(x*freq, y*freq); amp*=gain; freq*=lacunarity
    return total / sum(gain**i for i in range(octaves))
def fbm3(x, y, z, octaves=3, lacunarity=2.0, gain=0.5):
    total = np.zeros_like(x, dtype=float); amp=1.0; freq=1.0
    for _ in range(octaves):
        total += amp * perlin3(x*freq, y*freq, z*freq); amp*=gain; freq*=lacunarity
    return total / sum(gain**i for i in range(octaves))

SIZE=70; SEA_LEVEL=24; BASE_HEIGHT=28; TERRAIN_AMPL=12
TREE_PROB=0.03; CAVE_STRENGTH=0.6; CAVE_MAX_Y=40
COLOR_BY_TYPE={"bois":"#8B5A2B","feuille":"#2E8B57","ciel":"#87CEEB",
               "herbe":"#3CB371","eau":"#1E90FF","pierre":"#808080","vide":"#00000000"}

xs=np.arange(SIZE); zs=np.arange(SIZE)
X,Z=np.meshgrid(xs, zs, indexing="ij")
height_noise=fbm2(X*0.06, Z*0.06, octaves=5, lacunarity=2.0, gain=0.5)
heightmap=(BASE_HEIGHT+TERRAIN_AMPL*(height_noise-0.5)*2).round().astype(int)
heightmap=np.clip(heightmap, 6, SIZE-6)

ys=np.arange(SIZE)
Y3=np.repeat(ys[np.newaxis, np.newaxis, :], SIZE, axis=0); Y3=np.repeat(Y3, SIZE, axis=1)
X3=np.repeat(xs[:, np.newaxis, np.newaxis], SIZE, axis=1); X3=np.repeat(X3, SIZE, axis=2)
Z3=np.repeat(zs[np.newaxis, :, np.newaxis], SIZE, axis=0); Z3=np.repeat(Z3, SIZE, axis=2)
cave_field=fbm3(X3*0.08, Y3*0.08, Z3*0.08, octaves=4, lacunarity=2.2, gain=0.55)

types=np.full((SIZE,SIZE,SIZE), "vide", dtype=object)

# Ciel
for x in range(SIZE):
    for z in range(SIZE):
        h=heightmap[x,z]
        if h < SIZE-1:
            types[x, h+1:, z]="ciel"

# Sol: pierre + surface
for x in range(SIZE):
    for z in range(SIZE):
        h=heightmap[x,z]
        if h<=0: continue
        types[x, :h, z]="pierre"
        types[x, h, z]="herbe" if h>=SEA_LEVEL else "pierre"

# Eau
for x in range(SIZE):
    for z in range(SIZE):
        h=heightmap[x,z]
        if h<SEA_LEVEL:
            y0=h+1
            types[x, y0:SEA_LEVEL+1, z]="eau"
            if SEA_LEVEL+1<SIZE:
                types[x, SEA_LEVEL+1:, z]="ciel"

# Grottes
for x in range(SIZE):
    for z in range(SIZE):
        h=heightmap[x,z]; ymax=min(h, CAVE_MAX_Y)
        if ymax<=1: continue
        cave_vals=cave_field[x, z, :ymax]
        not_water = types[x, :ymax, z]!="eau"
        solid     = types[x, :ymax, z]=="pierre"
        carve_idx = np.where((cave_vals > CAVE_STRENGTH) & not_water & solid)[0]
        if carve_idx.size>0:
            types[x, carve_idx, z]="vide"

# Arbres
rng=random.Random(42)
for x in range(2,SIZE-2):
    for z in range(2,SIZE-2):
        y=heightmap[x,z]
        if y<=SEA_LEVEL: continue
        if types[x,y,z]=="herbe" and rng.random()<TREE_PROB:
            height_trunk=rng.choice([3,4])
            for dy in range(1,height_trunk+1):
                if y+dy<SIZE-1 and types[x,y+dy,z]=="ciel":
                    types[x,y+dy,z]="bois"
            top_y=min(y+height_trunk, SIZE-2)
            for dx in (-1,0,1):
                for dz in (-1,0,1):
                    for dy in (0,1,2):
                        yy=top_y+dy; xx=x+dx; zz=z+dz
                        if 0<=yy<SIZE and 0<=xx<SIZE and 0<=zz<SIZE and types[xx,yy,zz]=="ciel":
                            types[xx,yy,zz]="feuille"

# Couleurs
def color_for(t): return COLOR_BY_TYPE.get(t, "#000000")
colors=np.vectorize(color_for)(types)

# Export
xs_col, ys_col, zs_col = np.meshgrid(np.arange(SIZE), np.arange(SIZE), np.arange(SIZE), indexing="ij")
df=pd.DataFrame({
    "x": xs_col.ravel(),
    "y": ys_col.ravel(),
    "z": zs_col.ravel(),
    "type_bloc": types.ravel(),
    "couleur": colors.ravel()
})
df.to_csv("minecraft_70cube.csv", index=False)
print("OK: minecraft_70cube.csv créé avec", len(df), "lignes")
