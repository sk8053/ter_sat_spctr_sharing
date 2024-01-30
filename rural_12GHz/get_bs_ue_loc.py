import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
import random
import seaborn as sns

num_colors = 100  # Change this to the number of colors you want

# Generate a list of distinct colors
colors = sns.color_palette("hsv", num_colors)
#print(colors)
if 0:
    df = pd.read_csv('parsed_data_bs_to_ue/bs_1.csv')
    rx, ry, rz = df['rx_x'], df['rx_y'], df['rx_z']

    tx_list, ty_list, tz_list = [], [], []
    for i in tqdm(range(104)):
        df = pd.read_csv(f'parsed_data_bs_to_ue/bs_{i+1}.csv', engine='python')
        tx, ty, tz = df['tx_x'], df['tx_y'], df['tx_z']
        tx_list.append(tx.to_numpy()[0])
        ty_list.append(ty.to_numpy()[0])
        tz_list.append(tz.to_numpy()[0])

    tx_xyz = np.column_stack((tx_list, ty_list, tz_list))
    rx_xyz = np.column_stack((rx, ry, rz))
    np.savetxt('rural_bs_loc.txt', tx_xyz)
    np.savetxt('rural_ue_loc.txt', rx_xyz)
    plt.scatter(rx.to_numpy(), ry.to_numpy(),s = 1)
    plt.scatter(tx_list, ty_list, s =5, color = 'r')
    plt.show()

tx_xyz = np.loadtxt('rural_bs_loc.txt')
rx_xyz = np.loadtxt('rural_ue_loc.txt')
num_colors = 104  # Change this to the number of colors you want

distinct_colors = [
    "#FF5733",  # Red
    "#33FF57",  # Green
    "#3366FF",  # Blue
    "#FF33AA",  # Pink
    "#33FFFF",  # Cyan
    "#FFCC33",  # Orange
    "#9933FF",  # Purple
    "#33FFCC",  # Teal
    "#FF33FF",  # Magenta
    "#66FF33",  # Lime
    '#FFB90F',
    '#DC143C',
    '#8EE5EE',
    '#000000',
    '#BDFCC9',
    '#FDF5E6',
    '#FF83FA',
    '#FFC1C1',
    '#54FF9F',
    '#FFE4B5',
    '#006400',
    '#00FFFF',
    '#00008B',
    '#8A360F',
    '#CAFF70',
    '#808A87',
    '#808A87'
    # Add more colors as needed
]
n_cluster = 30
num_colors = n_cluster  # Change this to the number of colors you want

# Repeat the distinct colors to have more than 100
colors = (distinct_colors * (num_colors // len(distinct_colors) + 1))[:num_colors]

xy = rx_xyz[:,:2]
kmeans = KMeans(n_clusters=n_cluster, random_state=560, n_init=10, max_iter = 1000, tol =1e-7).fit(xy)

clusters = {cluster_idx:[] for cluster_idx in range(n_cluster)}
bs_to_ue = np.zeros((n_cluster, len(rx_xyz)))
print(xy.shape, len(kmeans.labels_))

for i, label in enumerate(kmeans.labels_):
    clusters[label].append(xy[i])
    bs_to_ue[label, i] = 1

rx_xyz = np.loadtxt('rural_ue_loc.txt')
rxy = rx_xyz[:,:2]
tx_xyz = np.loadtxt('rural_bs_loc.txt')
txy = tx_xyz[:,:2]
bs_to_ue_cluster = {bs_idx:[] for bs_idx in range(len(txy))}
for i in range(len(txy)):
    txy_i = txy[i]
    dxy = np.linalg.norm(rxy - txy_i[None], axis = 1)
    I = np.argsort(dxy)[:87]
    bs_to_ue_cluster[i].append(I)
clusters = bs_to_ue_cluster
print(clusters.keys())
for i, key in enumerate(clusters.keys()):
    I = np.array(clusters[key])
    xy = rxy[I].squeeze()

    plt.scatter(xy[:,0], xy[:,1], s = 7)

np.savetxt('ue_clustering_matrix.txt', bs_to_ue)
plt.show()