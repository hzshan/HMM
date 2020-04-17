import pickle
import numpy as np
from utils import get_dataset
import matplotlib.pyplot as plt
from utils import construct

window = 30
fitted_model = pickle.load(open('HMM_tau_' + str(window), 'rb'))

dataset = get_dataset(window)

p_comps = dataset.pca_results.components_
motifs = fitted_model.means_

all_labels = fitted_model.decode(dataset.all_reduced_segments)[1]
unique_labels, counts = np.unique(all_labels, return_counts=True)

sorted_unique_labels = unique_labels[np.argsort(counts)[::-1]]



""" Plot most common three motifs (judged by marginal prob)"""
motif_fig = plt.figure(figsize=(3, 3))
all_motifs = np.array([])
for i in range(10):
    true_ind = int(sorted_unique_labels[i])
    motif = construct(true_ind, p_comps, motifs)
    plt.plot(motif[1, :], -motif[0, :], label=str(i))
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.scatter(0, 0, color='g')
plt.text(0.05, 0, 'Start')
# plt.xlim(-0.5, 2)
# plt.ylim(-0.5, 2)
plt.legend()
plt.tight_layout()

example_motif_fig = plt.figure(figsize=(2.5, 2.5))

true_ind = int(sorted_unique_labels[5])
motif = construct(true_ind, p_comps, motifs)
plt.scatter(motif[1, :], -motif[0, :], color='blue', s=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.xlabel('a', fontsize=0)
plt.ylabel('a', fontsize=0)
#plt.xlim(-0.1, 0.5)
#plt.ylim(-0.4, 2.3)
plt.scatter(0, 0, color='green')
plt.scatter(motif[1, -1], -motif[0, -1], color='red')
plt.title('"circling"', fontsize=20)
plt.tight_layout()

"""Plot marginal probability of all motifs"""
plt.figure(figsize=(5, 3))
plt.plot(np.sort(counts)[::-1] / len(all_labels))
plt.ylabel('Marginal Prob')
plt.xlabel('Motif (sorted by marginal probability)')
plt.tight_layout()


all_labels = fitted_model.decode(dataset.all_reduced_segments)[1]
unique_labels, counts = np.unique(all_labels, return_counts=True)

sorted_unique_labels = unique_labels[np.argsort(counts)[::-1]]

def relabel_transmat(transmat, sorted_labels):
    results = np.zeros_like(transmat)
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            results[i, j] = transmat[int(sorted_labels[i]), int(sorted_labels[j])]
    return results

sorted_transmat = relabel_transmat(fitted_model.transmat_, sorted_unique_labels)
plt.figure()
plt.imshow(sorted_transmat)

"""Sort transmat manually to show blocks"""
dict_30 = np.array([0, 1, 5, 8, 9, 11, 12, 2, 3, 4, 6, 7, 10])


dict_60 = np.array([0, 6, 4, 7, 11, 13, 1, 2, 8, 3, 5, 10, 12, 9])

dict_120 = np.array([0, 1, 3, 4, 2, 5, 6])

dict_10 = np.array([0, 3, 17, 2, 4, 1, 5, 7, 8, 11, 6, 9, 10, 14, 12, 13, 15, 20, 25, 16, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29])

#                  0, 1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
dict_5 = np.array([0, 2, 5, 4, 15, 1, 13, 6, 11, 14, 3, 12, 17, 18, 7, 8, 9, 10, 16, 19, 20, 21, 22, 23, 24, 25])
np.unique(dict_10, return_counts=True)
def implement_manual_sort(mat, dict):
    assert np.all(np.unique(dict) == np.arange(len(dict)))
    results = np.zeros_like(mat)
    for i in range(len(dict)):
        for j in range(len(dict)):
            results[i, j] = mat[int(dict[i]), int(dict[j])]
    return results

block_mat = implement_manual_sort(sorted_transmat, dict_30)

plt.figure(figsize=(2.5, 2.5))
plt.imshow(block_mat)
plt.title('$\mathbf{T}_{30}$', fontsize=20)
#plt.xticks(fontsize=0)
#plt.yticks(fontsize=0)
plt.colorbar()

new_mat = np.zeros((3, 3))
groups = [np.array([0, 1, 2]), np.array([3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12])]
for i in range(3):
    for j in range(3):
        x_indices = groups[i]
        y_indices = groups[j]
        for k in x_indices:
            for l in y_indices:
                new_mat[i, j] += block_mat[int(k), int(l)]

    new_mat[i, :] /= np.sum(new_mat[i, :])


plt.figure(figsize=(3, 3))
plt.imshow(new_mat)
plt.xticks([0, 1, 2], ['idle', 'circling', 'dart'], fontsize=14, rotation=45)
plt.yticks([0, 1, 2], ['idle', 'circling', 'dart'], fontsize=14, rotation=45)
plt.colorbar()
plt.tight_layout()