from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_utils import get_AS_lib, mass_load_AS_lib, sampled_data, processed_data
from HMM_util import HMM_GMM_models, generate_groups, label_a_session, get_colorbar_labels
from HMM_util import get_models, rearrange_matrix, incre_granular, HMM_place_analysis, translate_labels
import scipy.stats
import cPickle as pickle
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter as gaussian_2d
from math_util import mutual_information_2d
# %% Grab AS library and fit HMM

HMM_AS_lib = get_AS_lib(1)
hmm_model = pickle.load(open('hmm_1.model','rb'))
# HMM_AS_lib = pickle.load(open('AS_lib1', 'rb'))
combined_data = np.vstack((HMM_AS_lib.DR_vectors_group1,
                           HMM_AS_lib.DR_vectors_group2))
combined_length = np.hstack((HMM_AS_lib.session_durations_g1,
                            HMM_AS_lib.session_durations_g2))
combined_length = combined_length[combined_length != 0]
combined_labels = hmm_model.decode(combined_data)[1]
combined_locs = np.vstack((HMM_AS_lib.locations_g1, HMM_AS_lib.locations_g2))
translated_labels = translate_labels(combined_labels)
translated_transmat = rearrange_matrix(hmm_model.transmat_,
                                       combined_labels)

all_hist = np.zeros((hmm_model.n_components, 10, 10))
for i in xrange(hmm_model.n_components):

    X_component0 = np.ma.masked_where(translated_labels != i,
                                      combined_locs[:, 0])
    X_component0 = X_component0.compressed()
    Y_component0 = np.ma.masked_where(translated_labels != i,
                                      combined_locs[:, 1])
    Y_component0 = Y_component0.compressed()

    hist, xedges, yedges = np.histogram2d(
        X_component0, Y_component0, bins=[10, 10],
        range=[[-30, 30], [-30, 30]])
    all_hist[i, :, :] = hist


# %% reconstruct modes
reconstruction = plt.figure(figsize=(4.5, 4.5))
k = 0 # just setting a counter to zero
for i in xrange(3):

    old_mode_index = combined_labels[np.where(translated_labels==i)[0][0]]
    recon_mode = np.zeros(len(HMM_AS_lib.vector_PCA.components_[k, :]))
    for k in xrange(5):
        recon_mode += hmm_model.means_[old_mode_index][k] * HMM_AS_lib.vector_PCA.components_[k, :]
    d2_mode = np.split(recon_mode.T, 2)
    plt.plot(-d2_mode[0], d2_mode[1],label=str(i+1), linewidth=3)

plt.yticks(fontsize=16, rotation=60)
plt.xticks(fontsize=16, rotation=30)
plt.legend(fontsize=16)
#plt.text()
plt.grid()
# %% First-order location-based marginal


marginals = all_hist / np.sum(all_hist, axis=0)

reshaped_marginals = np.reshape(marginals,
                                (hmm_model.n_components, 100))

graph_x = range(hmm_model.n_components)
dist_graph = plt.figure()
for i in xrange(100):
    plt.plot(range(hmm_model.n_components), reshaped_marginals[:, i])
plt.grid()
plt.xticks(graph_x, np.linspace(1, 33, 33).astype(int))

mean_usage = np.mean(reshaped_marginals, axis=1)
std_usage = np.std(reshaped_marginals, axis=1)

plt.figure(figsize=(3, 3))
plt.plot(graph_x, mean_usage - std_usage, 'lightblue', linewidth=2)
plt.plot(graph_x, mean_usage + std_usage, 'lightblue', linewidth=2)
plt.plot(graph_x, np.max(reshaped_marginals, axis=1),
         'lightsalmon', linewidth=3, label="Extreme")
plt.plot(graph_x, np.min(reshaped_marginals, axis=1),
         'lightsalmon', linewidth=3)

plt.fill_between(graph_x, mean_usage, mean_usage - std_usage,
                 color='lightblue', label="Standard dev.")
plt.fill_between(graph_x, mean_usage, mean_usage + std_usage,
                 color='lightblue')
plt.plot(graph_x, mean_usage, 'blue', linewidth=2, label='Mean')

#plt.legend(fontsize=12)
plt.xticks(rotation='horizontal', fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.ylabel(r'$p(S_t=k)$', fontsize=16)
plt.xlabel('k', fontsize=16)
# plt.legend(fontsize=25)
#%%

assembled_obs = incre_granular(2, data=translated_labels, redundant=True)

edges = np.linspace(-0.5, hmm_model.n_components - 0.5,
                    hmm_model.n_components+1)
observed_transition, xedges, yedges = np.histogram2d(
    assembled_obs[:, 0], assembled_obs[:, 1], bins=edges)

observed_tran_p = observed_transition / np.sum(observed_transition)
mode_usage_count = np.unique(translated_labels, return_counts=True)[1]
marginal_prob = mode_usage_count.astype(float) / sum(mode_usage_count)
posterior_trans_mat = np.zeros_like(translated_transmat)

for i in xrange(hmm_model.n_components):
    posterior_trans_mat[i, :] = translated_transmat[i, :] * marginal_prob[i]

plt.figure(figsize=(4.5, 3))
plt.imshow(observed_tran_p,
           interpolation='None', cmap='Blues')
plt.xticks((0 + 1 / 2, hmm_model.n_components - 1 / 2),
           (1, hmm_model.n_components), fontsize=16)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=16)



plt.title(r'$Observed\:p(S_{t+1} , S_t)$', fontsize=16)
plt.figure(figsize=(4.5, 3))

plt.imshow(translated_transmat, interpolation='None',
           cmap='Reds')
plt.colorbar()
plt.title(r'$p(S_{t+1} | S_t)$', fontsize=16)
plt.xticks((0 + 1 / 2, hmm_model.n_components - 1 / 2),
           (1, hmm_model.n_components), fontsize=16)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=16)

plt.figure(figsize=(4.5, 3))
plt.imshow(rearrange_matrix(posterior_trans_mat, translated_labels),
           interpolation='None', cmap='Blues')
plt.title(r'$p(S_{t+1}|S_{t})p(S_{t})$', fontsize=16)
plt.colorbar()
plt.xticks((0 + 1 / 2, hmm_model.n_components - 1 / 2),
           (1, hmm_model.n_components), fontsize=16)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=16)

plt.figure(figsize=(4.5, 3))
plt.imshow(observed_tran_p - posterior_trans_mat,
           interpolation='None')
plt.colorbar()
plt.title(r'${p(S_{t+1} , S_t)} - p(S_{t+1}|S_{t})p(S_{t})$',
          fontsize=16)
plt.xticks((0 + 1 / 2, hmm_model.n_components - 1 / 2),
           (1, hmm_model.n_components), fontsize=16)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=16)

# %% Figure making


figure, ax = plt.subplots(figsize=(8, 6))

gra1 = plt.imshow(translated_transmat,
                  interpolation='None', cmap='Reds')

cb1 = plt.colorbar(gra1, ticks=[0, 0.37, 0.74], orientation='vertical')
cb1.ax.tick_params(labelsize=30)

plt.ylabel(r'$S_t$', fontsize=30)
plt.xlabel(r'$S_{t+1}$', fontsize=30)
plt.xticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=24)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=24)
ax.xaxis.set_label_position('top') 
ax.xaxis.tick_top()


fig = plt.figure(figsize=(4, 6))
modes, counts = np.unique(translated_labels, return_counts=True)
arranged_counts = np.sort(counts)

y = arranged_counts[::-1]
x = np.linspace(0, hmm_model.n_components-1, hmm_model.n_components)
width = 1/1.5
barlist = plt.barh((x+0.5), y, width, color='darksalmon')
plt.yticks(fontsize=30)
plt.xticks((0,50000), (0, 50000), fontsize=24)
plt.ylim([0,hmm_model.n_components+0.5])

figure, ax = plt.subplots(figsize=(8, 6))
gra2 = plt.imshow(posterior_trans_mat,
                  interpolation='None', cmap='Reds')
plt.ylabel(r'$S_t$', fontsize=30)
plt.xlabel(r'$S_{t+1}$', fontsize=30)
ax.xaxis.set_label_position('top') 
ax.xaxis.tick_top()


cb2 = plt.colorbar(gra2, ticks=[0, 0.05, 0.11], orientation='vertical')
cb2.ax.set_xticklabels(['0', '0.05', '0.11'])
cb2.ax.tick_params(labelsize=30)
plt.xticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=24)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=24)


# %% Second order location-based marg (spatial heterogeneity)


# labels are acquired with place analysis
current_mode = translated_labels[np.arange(len(translated_labels))!=len(translated_labels)-1]
next_mode = translated_labels[np.arange(len(translated_labels))!=0]
coordinates = combined_locs[np.arange(len(translated_labels))!=len(translated_labels)-1, :]
x_coor = coordinates[:, 0]
y_coor = coordinates[:, 1]

edges = np.linspace(-30, 30, 61)

entropy_storage = np.zeros((hmm_model.n_components,
                        hmm_model.n_components))
hists = np.zeros((hmm_model.n_components,
                  hmm_model.n_components, 60, 60))
total_hist, xedges, yedges = np.histogram2d(combined_locs[:, 0],
            combined_locs[:, 1], bins=edges)

for i in xrange(hmm_model.n_components):
    for k in xrange(hmm_model.n_components):
        subsetx = np.ma.masked_where(current_mode != i, x_coor)
        subsetx = np.ma.masked_where(next_mode != k, subsetx).compressed()
        subsety = np.ma.masked_where(current_mode != i, y_coor)
        subsety = np.ma.masked_where(next_mode != k, subsety).compressed()
        hist, xedges, yedges = np.histogram2d(subsetx, subsety, bins=edges)

        if np.sum(hist) == 0:
            entropy_storage[i, k] = 0
        else:
            hist = hist / (total_hist + 1)
            hists[i, k, :, :] = hist
            entropy_storage[i, k] = scipy.stats.entropy(hist.ravel())

spatial_hetero, ax = plt.subplots(figsize=(8, 6))
spa_het = plt.imshow(entropy_storage, interpolation='nearest')
ax.xaxis.set_label_position('top') 
ax.xaxis.tick_top()
plt.ylabel(r'$S_t$', fontsize=30)
plt.xlabel(r'$S_{t+1}$', fontsize=30)
plt.xticks((0 + 1 / 2, hmm_model.n_components - 1 / 2),
           (1, hmm_model.n_components), fontsize=24)
plt.yticks((0, hmm_model.n_components-1),
           (1, hmm_model.n_components), fontsize=24)
cb1 = plt.colorbar(spa_het, orientation='vertical')
cb1.ax.tick_params(labelsize=24) 
cb1.set_label('Entropy (nats)', fontsize=24)

# %% Example graphs of the spatial heterogeneity
# i, k = np.random.choice(5, 2)


i = 1
k = 3
smoothing_sigma = 5

X_component0 = np.ma.masked_where(translated_labels != i,
                                  combined_locs[:, 0])
X_component0 = X_component0.compressed()
Y_component0 = np.ma.masked_where(translated_labels != i,
                                  combined_locs[:, 1])
Y_component0 = Y_component0.compressed()

mode_usage, xedges, yedges = np.histogram2d(
    X_component0, Y_component0, bins=[60, 60],
    range=[[-30, 30], [-30, 30]])
mode_prob = plt.figure(figsize=(8, 6)) # mode i usage

mode_img = plt.imshow(gaussian_2d(mode_usage / (total_hist + 1),
                                  sigma=smoothing_sigma).T)
plt.xlabel('Coordinate (Y)', fontsize=24)
plt.ylabel('Coordinate (X)', fontsize=24)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)

cb2 = plt.colorbar(mode_img, orientation='vertical')
cb2.ax.tick_params(labelsize=24) 
cb_labels = get_colorbar_labels(gaussian_2d(mode_usage / (total_hist + 1),
                                  sigma=smoothing_sigma))
cb2.ax.tick_params(labelsize=24) 
cb2.ax.set_yticklabels(cb_labels, rotation=30)


trans_img = hists[i, k, :, :] * (total_hist + 1) / (mode_usage + 1)
hetero_example = plt.figure(figsize=(8, 6)) # transition from i to k
example_img = plt.imshow(gaussian_2d(trans_img,
                                     sigma=smoothing_sigma).T)
cb1 = plt.colorbar(example_img, orientation='vertical')
cb_labels = get_colorbar_labels(gaussian_2d(trans_img,
                                     sigma=smoothing_sigma))
cb1.ax.tick_params(labelsize=24) 
cb1.ax.set_yticklabels(cb_labels, rotation=30)
plt.xlabel('Coordinate (Y)', fontsize=24)
plt.ylabel('Coordinate (X)', fontsize=24)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)





plt.figure(figsize=(8, 6)) # Overall placement heatmap
overall_placement = plt.imshow(gaussian_2d(total_hist, smoothing_sigma).T)
plt.xlabel('Coordinate (Y)', fontsize=24)
plt.ylabel('Coordinate (X)', fontsize=24)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
cb_labels = get_colorbar_labels(gaussian_2d(total_hist, smoothing_sigma), 0)

cb3 = plt.colorbar(overall_placement, orientation='vertical')
cb3.ax.tick_params(labelsize=24) 
cb3.ax.set_yticklabels(cb_labels, rotation=0)

# %%
error_fig = plt.figure(figsize=(13,12))
sub1 = error_fig.add_subplot(221)
img = plt.imshow(posterior_trans_mat, interpolation='None',
           cmap='Blues', clim=[0, 0.12])
cb1 = plt.colorbar(img, orientation='vertical')
cb1.set_ticks([0, 0.06, 0.12])
cb1.set_ticklabels(['0', '.06', '.12'])
cb1.ax.tick_params(labelsize=24) 
sub1.xaxis.set_label_position('top') 
sub1.xaxis.tick_top()


plt.ylabel(r'$S_t$', fontsize=30)
plt.xlabel(r'$S_{t+1}$', fontsize=30)
plt.xticks((0 + 1 / 2, HMM_GMM.HMM_model.n_components - 1 / 2),
           (1, HMM_GMM.HMM_model.n_components), fontsize=24)
plt.yticks((0, HMM_GMM.HMM_model.n_components-1),
           (1, HMM_GMM.HMM_model.n_components), fontsize=24)

plt.title('Observed', fontsize=30)

sub2 = error_fig.add_subplot(222)
plt.imshow(observed_tran_p,
           interpolation='None', cmap='Blues', clim=[0, 0.12])
plt.xticks((0 + 1 / 2, HMM_GMM.HMM_model.n_components - 1 / 2),
           (1, HMM_GMM.HMM_model.n_components), fontsize=24)
plt.yticks((0, HMM_GMM.HMM_model.n_components-1),
           (1, HMM_GMM.HMM_model.n_components), fontsize=24)
cb2 = plt.colorbar(img, orientation='vertical')
cb2.set_ticks([0, 0.06, 0.12])
cb2.set_ticklabels(['0', '.06', '.12'])
cb2.ax.tick_params(labelsize=24)
plt.title('Computed', fontsize=30)
sub2.xaxis.set_label_position('top') 
sub2.xaxis.tick_top()

ax3 = error_fig.add_subplot(223)
im3 = plt.imshow((observed_tran_p - posterior_trans_mat),
           interpolation='None', cmap='coolwarm')
          
plt.title('Error', fontsize=30)
plt.xticks((0 + 1 / 2, HMM_GMM.HMM_model.n_components - 1 / 2),
           (1, HMM_GMM.HMM_model.n_components), fontsize=24)
plt.yticks((0, HMM_GMM.HMM_model.n_components-1),
           (1, HMM_GMM.HMM_model.n_components), fontsize=24)
cb3 = plt.colorbar(im3)
cb3.set_ticks([-0.004, 0.002, 0.008])
cb3.set_ticklabels(['-0.004', '.002', '.008'])
cb3.ax.tick_params(labelsize=24) 

sub4 = error_fig.add_subplot(224)
im4 = plt.imshow((observed_tran_p - posterior_trans_mat),
           interpolation='None', cmap='coolwarm')

ax3.add_patch(
    patches.Rectangle(
        (- 0.5, - 0.5),
        16,
        16,
        fill=False, linewidth=3))    # remove background
ax3.text(2, 20, 'Enlarged region', fontsize=20,
        bbox={'facecolor':'w', 'alpha':1, 'pad':10})
ax3.xaxis.set_label_position('top') 
ax3.xaxis.tick_top()


plt.title('Error (enlarged)', fontsize=30)
plt.xlim([- 0.5, 15])
plt.ylim([15, -0.5])
plt.xticks((1 / 2, 15 + 1 / 2),
           (1, 16), fontsize=24)
plt.yticks((1 / 2, 15 + 1 / 2),
           (1, 16), fontsize=24)
sub4.xaxis.set_label_position('top') 
sub4.xaxis.tick_top()

Hamming_loss_lb = np.sum(np.abs(observed_tran_p - posterior_trans_mat)) * 0.5
print('Hamming loss lower bound is %s') % Hamming_loss_lb

np.sum(np.dot(observed_tran_p, posterior_trans_mat))
np.sum(np.dot(observed_tran_p, observed_tran_p))
DKL = scipy.stats.entropy(observed_tran_p.ravel(), posterior_trans_mat.ravel())

print('DKL is %s.') % DKL

# %% Is covariance of each mode related to prediction error?
covariance = HMM_GMM.HMM_model._covars_
total_dispersion = np.sum(covariance.reshape(33, 25), axis=1)
plt.figure()
plt.plot(range(HMM_GMM.HMM_model.n_components), total_dispersion)

# %% Making a figure for MI at different jumps and different timescales.
fig = plt.figure(figsize=(9, 5))


lengths = np.array([1, 5, 10, 15, 30, 60])

for k, time_length in enumerate(lengths):
    HMM_AS_lib = get_AS_lib(time_length)
    model_name = 'hmm_' + str(lengths[k]) + '.model'
    hmm_model = pickle.load(open(model_name,'rb'))

    combined_labels = hmm_model.decode(combined_data)[1]
    translated_labels = translate_labels(combined_labels)
    mi_Ns = np.zeros(10)
    control_mi = np.zeros(10)
    first_label = translated_labels
    second_label = np.append(0, first_label)[0: len(first_label)]

    for i in xrange(10):
        mi_Ns[i] = mutual_information_2d(first_label, second_label)
        control_mi[i] = mutual_information_2d(
            np.random.permutation(first_label),
            np.random.permutation(first_label))

        second_label = np.append(0, second_label)[0: len(second_label)]

    plt.plot(range(10), mi_Ns, str(0.7 - k * 0.1), label=str(time_length), 
             linewidth=4)
    #plt.plot(range(10), control_mi, label='mutual information')
plt.legend(fontsize=16)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
          # ncol=4, mode="expand", borderaxespad=0.,fontsize = 18)
plt.xticks(np.linspace(0, 9, 9),
           np.linspace(1, 10, 10).astype(int), fontsize=17)
plt.yticks((0.2, 0.4, 0.6), ('.2', '.4', '.6'), fontsize=17)
plt.xlabel('Jump size', fontsize=18)
plt.ylabel('MI (nats)', fontsize=18)
plt.grid()
# %% Testing Chapman_Kolmogorov

lengths = np.array([1, 5, 10, 15, 30, 60])
colors = np.array(['k', 'r', 'chocolate', 'limegreen', 'cyan',
                   'slategrey', 'b', 'm'])

DKLs = np.zeros((6, 50))
Hamming = np.zeros_like(DKLs)

for k, time_length in enumerate(lengths):
    HMM_AS_lib = get_AS_lib(time_length)
    model_name = 'hmm_' + str(lengths[k]) + '.model'
    hmm_model = pickle.load(open(model_name,'rb'))

    combined_labels = hmm_model.decode(combined_data)[1]
    translated_labels = translate_labels(combined_labels)
    translated_transmat = rearrange_matrix(hmm_model.transmat_,
                                       combined_labels)

    for jump in xrange(50):
        windowed_data = incre_granular(jump + 2, translated_labels,
                                       redundant=True)
        jump_trans = np.linalg.matrix_power(translated_transmat,
                                            jump + 1)
        edges = np.linspace(-0.5, hmm_model.n_components - 0.5,
        hmm_model.n_components+1)
        observed_transition, xedges, yedges = np.histogram2d(
            windowed_data[:, 0], windowed_data[:, jump+1], bins=edges)
        observed_tran_p = observed_transition / np.sum(observed_transition)
        mode_usage_count = np.unique(translated_labels, return_counts=True)[1]
        if hmm_model.n_components != len(mode_usage_count):
            mode_usage_count = np.append(mode_usage_count,
                                         np.zeros(hmm_model.n_components - len(mode_usage_count)))
        marginal_prob = mode_usage_count.astype(float) / sum(mode_usage_count)
        posterior_trans_mat = np.zeros_like(translated_transmat)

        for i in xrange(hmm_model.n_components):
            posterior_trans_mat[i, :] = jump_trans[i, :] * marginal_prob[i]
        DKLs[k, jump] = scipy.stats.entropy(observed_tran_p.ravel(),
                                            posterior_trans_mat.ravel())
        Hamming[k, jump] = 0.5 * np.sum(np.abs(
                        observed_tran_p.ravel() - posterior_trans_mat.ravel()))
# %%
DKL_graph = plt.figure(figsize=(9, 4))

ax = DKL_graph.add_subplot(111)
for i in xrange(6):
    plt.plot(np.linspace(0, 49, 50).astype(int), DKLs[i, :], colors[i],
             linewidth=2,
             label=lengths[i])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,fontsize = 18)
plt.xticks(np.linspace(0, 49, 10),
           np.linspace(1, 50, 10).astype(int), fontsize=18)
plt.xlabel('Jump size', fontsize=18)
plt.ylabel(r'$D_{KL}$', fontsize=18)
'''
enlarged = ax.add_patch(
    patches.Rectangle(
        (0, 0),
        3,
        0.1,
        fill=False, linestyle='dashed', linewidth=3))   
'''
plt.grid()
'''
DKL_graph2 = plt.figure(figsize=(4, 6))
for k in xrange(6):
    i = 5 - k
    plt.plot(np.linspace(0, 49, 50).astype(int), DKLs[i, :], colors[i],
             linewidth=4,
             label=lengths[i])
plt.xlim([0, 3])
plt.ylim([0, 0.1])
plt.xticks(np.linspace(0, 3, 4),
           np.linspace(1, 4, 4).astype(int), fontsize=24)
plt.yticks((0.05, 0.10), ('.05', '.10'), fontsize=24)
plt.xlabel('Jump size', fontsize=30)
plt.grid()
'''
Hamming_graph = plt.figure(figsize=(9, 4))
ax = Hamming_graph.add_subplot(111)
for i in xrange(6):
    plt.plot(np.linspace(0, 49, 50).astype(int), Hamming[i, :], colors[i],
             linewidth=2,
             label=lengths[i])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,fontsize = 18)
plt.xticks(np.linspace(0, 49, 10),
           np.linspace(1, 50, 10).astype(int), fontsize=18)
plt.xlabel('Jump size', fontsize=18)
plt.ylabel(r'$\mathcal{L}^{H}_{min}$', fontsize=18)
'''
enlarged = ax.add_patch(
    patches.Rectangle(
        (0, 0),
        3,
        0.1,
        fill=False, linestyle='dashed', linewidth=3))   
'''
plt.grid()
'''
Hamming_graph2 = plt.figure(figsize=(4, 6))
for k in xrange(6):
    i = 5 - k
    plt.plot(np.linspace(0, 49, 50).astype(int), Hamming[i, :], colors[i],
             linewidth=4,
             label=lengths[i])
plt.xlim([0, 3])
plt.ylim([0, 0.1])
plt.xticks(np.linspace(0, 3, 4),
           np.linspace(1, 4, 4).astype(int), fontsize=24)
plt.yticks((0.05, 0.10), ('.05', '.10'), fontsize=24)
plt.xlabel('Jump size', fontsize=30)
plt.grid()
'''
# %% Mode duration analysis
lengths = np.array([1, 5, 10, 15, 30, 60, 100, 200])
colors = np.array(['k', 'r', 'chocolate', 'limegreen', 'cyan',
                   'slategrey', 'b', 'm'])


HMM_AS_library = get_AS_lib(200)
HMM_GMM, combined_labels, combined_locs, combined_dataset, combined_length = get_models(
    True, 200, HMM_AS_library)
translated_labels = translate_labels(combined_labels)
gmm_labels = HMM_GMM.GMM_control.predict(combined_dataset)
gmm_translated = translate_labels(gmm_labels)
translated_transmat = rearrange_matrix(HMM_GMM.HMM_model.transmat_,
                                   combined_labels)

# %%
HMM_AS_lib = get_AS_lib(60)
hmm_model = pickle.load(open('hmm_60.model','rb'))
# HMM_AS_lib = pickle.load(open('AS_lib1', 'rb'))
combined_data = np.vstack((HMM_AS_lib.DR_vectors_group1,
                           HMM_AS_lib.DR_vectors_group2))
combined_length = np.hstack((HMM_AS_lib.session_durations_g1,
                            HMM_AS_lib.session_durations_g2))
combined_length = combined_length[combined_length != 0]
combined_labels = hmm_model.decode(combined_data)[1]
combined_locs = np.vstack((HMM_AS_lib.locations_g1, HMM_AS_lib.locations_g2))
translated_labels = translate_labels(combined_labels)
translated_transmat = rearrange_matrix(hmm_model.transmat_,
                                       combined_labels)

r_squared_storage = np.zeros(hmm_model.n_components)

for i in xrange(hmm_model.n_components):
    indices = np.where(translated_labels == i)
    switches = np.zeros_like(translated_labels)
    switches[indices] = 1
    double_switch = incre_granular(2, switches, redundant=True)
    second_switches = double_switch[:, 1]
    on_switch = np.ones_like(second_switches)
    off_switch = np.ones_like(second_switches)
    on_switch[double_switch[:, 0] == 1] = 0
    on_switch[double_switch[:, 1] == 0] = 0
    off_switch[double_switch[:, 0] == 0] = 0
    off_switch[double_switch[:, 1] == 1] = 0
    on_loc = np.where(on_switch==1)[0]
    off_loc = np.where(off_switch==1)[0]
    if double_switch[0, 0] == 1:
        on_loc = np.append(0, on_loc)
    if double_switch[len(double_switch) - 2, 1] == 1:
        off_loc = np.append(off_loc, len(double_switch) - 2)
    mode_durations = off_loc - on_loc
    
    
#    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 20, 21)
    bin_centers = np.mean(incre_granular(2, bins, True), axis=1)
    bin_centers[len(bin_centers) - 1] = bin_centers[len(bin_centers) - 2] + 1
    counts, edges = np.histogram(mode_durations, bins=bins)
    x_intercept = np.argmax(counts)
    truncated_counts = counts[x_intercept: len(bins)]
    truncated_bin_centers = bin_centers[x_intercept: len(bins)] - x_intercept
#    plt.scatter(bin_centers, counts)

    
    def exponenial_func(x, a, b, c):
        return a*np.exp(-b * (x)) + c

   
    def inverse_func(x, a, b):
        return a / x + b


    def log_function(x, a ,b ,c):
        return b*np.log(a*x)+c
    
    
    
    popt, pcov = curve_fit(exponenial_func, truncated_bin_centers,
                       truncated_counts, p0=(1, 1e-6, 1))
    xxx = truncated_bin_centers
    yyy = exponenial_func(xxx, *popt)
    xxx = x_intercept + xxx
#    plt.plot(xxx, yyy, linewidth=3)
    
    residual_SS = np.sum(np.power(
            yyy[x_intercept: len(bins)] - truncated_counts[x_intercept: len(bins)], 2))
    total_SS = np.sum(np.power(
        truncated_counts[x_intercept: len(bins)]-np.mean(truncated_counts[x_intercept: len(bins)]), 2))
    r_squared_2 = 1 - residual_SS / total_SS
    print(r_squared_2)
    r_squared_storage[i] = r_squared_2

# %% Figure making of summary
exp_figure = plt.figure(figsize=(3, 3))
r_squared_storage = r_squared_storage[r_squared_storage != 0]
print(np.mean(r_squared_storage))
print(np.median(r_squared_storage))
plt.hist(r_squared_storage, bins=np.linspace(0.45, 1, 12), color='skyblue')
plt.xlabel(r'$r^{2}$', fontsize=16)
plt.xticks(np.linspace(0.4, 1, 7), np.linspace(0.4, 1, 7), fontsize=16)
plt.yticks((0, 10, 20, 30), (0, 10, 20, 30), fontsize=0)
exp_figure.autofmt_xdate()
plt.grid()
text = 'Mean: %2f;\nMedian: %2f' % (np.mean(r_squared_storage[0:30]), np.median(r_squared_storage))
plt.text(0.45, 12, text, fontsize=16)

# %% Figure making of example
i = 1
indices = np.where(translated_labels == i)
switches = np.zeros_like(translated_labels)
switches[indices] = 1
double_switch = incre_granular(2, switches, redundant=True)
second_switches = double_switch[:, 1]
on_switch = np.ones_like(second_switches)
off_switch = np.ones_like(second_switches)
on_switch[double_switch[:, 0] == 1] = 0
on_switch[double_switch[:, 1] == 0] = 0
off_switch[double_switch[:, 0] == 0] = 0
off_switch[double_switch[:, 1] == 1] = 0
on_loc = np.where(on_switch==1)[0]
off_loc = np.where(off_switch==1)[0]
if double_switch[0, 0] == 1:
    on_loc = np.append(0, on_loc)
if double_switch[len(double_switch) - 2, 1] == 1:
    off_loc = np.append(off_loc, len(double_switch) - 2)
mode_durations = off_loc - on_loc


plt.figure(figsize=(6, 3))
bins = np.linspace(0, 20, 21)
bin_centers = np.mean(incre_granular(2, bins, True), axis=1)
bin_centers[len(bin_centers) - 1] = bin_centers[len(bin_centers) - 2] + 1
counts, edges = np.histogram(mode_durations, bins=bins)
x_intercept = np.argmax(counts)
truncated_counts = counts[x_intercept: len(bins)]
truncated_bin_centers = bin_centers[x_intercept: len(bins)] - x_intercept
plt.scatter(bin_centers+0.5, counts / np.sum(counts), s=50, color = 'k')


def exponenial_func(x, a, b, c):
    return a*np.exp(-b * (x)) + c


popt, pcov = curve_fit(exponenial_func, truncated_bin_centers,
                       truncated_counts, p0=(1, 1e-6, 1))
xxx = np.linspace(0, 20, 100)
yyy = exponenial_func(xxx, *popt)
xxx = x_intercept + xxx
plt.plot(xxx+0.5, yyy / np.sum(counts), linewidth=3)

xxx = truncated_bin_centers
yyy = exponenial_func(xxx, *popt)

residual_SS = np.sum(np.power(
        yyy[x_intercept: len(bins)] - truncated_counts[x_intercept: len(bins)], 2))
total_SS = np.sum(np.power(
    truncated_counts[x_intercept: len(bins)]-np.mean(truncated_counts[x_intercept: len(bins)]), 2))
r_squared_2 = 1 - residual_SS / total_SS
print(r_squared_2)
plt.xlabel('Duration(sec)', fontsize=16)
plt.ylabel('Prob Density', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.hlines(0, 0, 25)
plt.xlim([0, 21])
plt.ylim([-0.01, np.max(yyy / np.sum(counts))+0.01])
plt.grid()
plt.text(10, 0.4, r'$r^{2}=.998$', fontsize=16)

# %% Goodness of fit at different timescales
means = np.array([.999, .998, .997, .997, .991, .979])
medians = np.array([1.000, .999, 1.000, .999, .995, .993])
lengths = np.array([1, 5, 10, 15, 30, 60])

fit_figure = plt.figure(figsize=(9, 3))
plt.plot(range(6), means, linewidth=5, label=r'$\bar{r^{2}} Mean$')
plt.plot(range(6), medians, linewidth=5, label=r'$\tilde{r^{2}} Median$')
plt.legend(fontsize=18, loc=3)
plt.grid()
plt.ylim([0.64, 1.01])
plt.xticks(range(6), lengths, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(r'$r^{2}$', fontsize=18)
plt.xlabel('Timescale (sec)', fontsize=18)