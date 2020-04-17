import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from utils import get_dataset, construct
import pickle
import scipy.stats

window = 1
n_modes = 39

fitted_model = pickle.load(open('HMM_tau_' + str(window), 'rb'))
dataset = get_dataset(window)

all_locs = np.vstack((dataset.g1_locations, dataset.g2_locations))
opener_labels = fitted_model.decode(dataset.g1_reduced_segments)[1]
non_opener_labels = fitted_model.decode(dataset.g2_reduced_segments)[1]

all_labels = np.append(opener_labels, non_opener_labels)
all_lbs, all_counts = np.unique(np.append(all_labels, np.arange(0, n_modes)), return_counts=True)

opener_lbs, opener_counts = np.unique(np.append(opener_labels, np.arange(0, n_modes)), return_counts=True)
nonopener_lbs, nonopener_counts = np.unique(np.append(non_opener_labels, np.arange(0, n_modes)), return_counts=True)

opener_counts = opener_counts[np.argsort(all_counts)[::-1]]
nonopener_counts = nonopener_counts[np.argsort(all_counts)[::-1]]

plt.figure()
plt.plot(opener_counts / np.sum(opener_counts))
plt.plot(nonopener_counts / np.sum(nonopener_counts))

def make_session_mask(counts, index):
    assert index < len(counts)
    mask = np.zeros(int(np.sum(counts)))
    prev = np.sum(counts[0:index])
    mask[int(prev + 1):int(prev + counts[index])] = 1
    return mask

opener_counts_over_session = np.zeros((n_modes, len(dataset.g1_segment_counts)))

for i in range(len(dataset.g1_segment_counts)):
    sub_labels = np.ma.masked_where(make_session_mask(dataset.g1_segment_counts, i) == 0, opener_labels).compressed()
    labels, counts = np.unique(np.append(sub_labels, np.arange(0, int(np.max(opener_labels)))), return_counts=True)
    opener_counts_over_session[:, i] = counts - 1


nonopener_counts_over_session = np.zeros((n_modes, len(dataset.g2_segment_counts)))

for i in range(len(dataset.g2_segment_counts)):
    sub_labels = np.ma.masked_where(make_session_mask(dataset.g2_segment_counts, i) == 0, non_opener_labels).compressed()
    labels, counts = np.unique(np.append(sub_labels, np.arange(0, n_modes)), return_counts=True)
    nonopener_counts_over_session[:, i] = counts - 1
    nonopener_counts_over_session[:, i]

opener_stds = np.std(opener_counts_over_session, axis=1)
non_opener_stds = np.std(nonopener_counts_over_session, axis=1)

opener_stds = opener_stds[np.argsort(all_counts)[::-1]]
non_opener_stds = non_opener_stds[np.argsort(all_counts)[::-1]]
opener_means = opener_counts / len(dataset.g1_segment_counts)
non_opener_means = nonopener_counts / len(dataset.g2_segment_counts)

x_axis = all_lbs
plt.figure(figsize=(4, 3))
plt.plot(x_axis, opener_means, label='opener', color='r')
plt.plot(x_axis, non_opener_means, label='non-opener', color='k')
plt.fill_between(x_axis, opener_means, opener_stds / np.sqrt(len(dataset.g1_segment_counts)) + opener_means,
                 color='pink')
plt.fill_between(x_axis, opener_means, -opener_stds / np.sqrt(len(dataset.g1_segment_counts)) + opener_means,
                 color='pink')
plt.fill_between(x_axis, non_opener_means, non_opener_stds / np.sqrt(len(dataset.g2_segment_counts)) + non_opener_means,
                 color='gray')
plt.fill_between(x_axis, non_opener_means, -non_opener_stds / np.sqrt(len(dataset.g2_segment_counts)) + non_opener_means,
                 color='gray')

plt.legend()
plt.ylabel('Usage per session', fontsize=14)
plt.xlabel('Behavior Index', fontsize=14)
plt.grid()
plt.tight_layout()



unique_labels, counts = np.unique(all_labels, return_counts=True)

sorted_unique_labels = unique_labels[np.argsort(counts)[::-1]]
true_label = int(sorted_unique_labels[0])

p_comps = dataset.pca_results.components_
motifs = fitted_model.means_

motif = construct(true_label, p_comps, motifs)
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

p_values = np.zeros(n_modes)
for i in range(n_modes):
    test = scipy.stats.ttest_ind(opener_counts_over_session[i, :], nonopener_counts_over_session[i, :])
    p_values[i] = test.pvalue

corrected_p = p_values * n_modes
corrected_p = corrected_p[np.argsort(counts)[::-1]]