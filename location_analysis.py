import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from utils import get_dataset, construct
import pickle

window = 30
fitted_model = pickle.load(open('HMM_tau_' + str(window), 'rb'))
dataset = get_dataset(window)

all_locs = np.vstack((dataset.g1_locations, dataset.g2_locations))
all_labels = fitted_model.decode(dataset.all_reduced_segments)[1]

n_modes = int(np.max(all_labels) + 1)

loc_counts = np.zeros((n_modes, 29, 29))
loc_prob = np.zeros_like(loc_counts)

for i in range(len(all_labels)):
    loc_counts[int(all_labels[i]), int(all_locs[i, 0] / 2) + 14, int(all_locs[i, 1] / 2) + 14] += 1

for i in range(n_modes):
    loc_prob[i, :, :] = loc_counts[i, :, :] / (np.sum(loc_counts[i, :, :]) + 1e-6)


plt.figure(figsize=(2.5, 2))
plt.imshow(loc_counts[2, :, :].T / (np.sum(loc_counts, axis=0).T + 1))
plt.colorbar()
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.tight_layout()






p_comps = dataset.pca_results.components_
motifs = fitted_model.means_

motif = construct(2, p_comps, motifs)
plt.figure(figsize=(2, 2))
plt.plot(motif[1, :], -motif[0, :], color='blue')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.xlabel('a', fontsize=0)
plt.ylabel('a', fontsize=0)
#plt.xlim(-0.5, 0.5)
#plt.ylim(-0.4, 2.3)
plt.scatter(0, 0, color='green')
plt.scatter(motif[1, -1], -motif[0, -1], color='red')
plt.tight_layout()