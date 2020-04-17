import numpy as np


# data for T=1 are in "temp_store"

data_storage = np.zeros((3, 7, 15))

lengths = np.array([1, 5, 10, 15, 30, 60, 100])

for k, time_length in enumerate(lengths):
    
    HMM_AS_lib = get_AS_lib(time_length)
    
    for m in xrange(15):
        HMM_GMM, combined_labels, combined_locs, combined_dataset, combined_length = get_models(False, time_length, HMM_AS_lib)
        translated_labels = translate_labels(combined_labels)
        translated_transmat = rearrange_matrix(HMM_GMM.HMM_model.transmat_,
                                           combined_labels)
        assembled_obs = incre_granular(2, data=translated_labels, redundant=True)
        
        edges = np.linspace(-0.5, HMM_GMM.HMM_model.n_components - 0.5,
                            HMM_GMM.HMM_model.n_components+1)
        observed_transition, xedges, yedges = np.histogram2d(
            assembled_obs[:, 0], assembled_obs[:, 1], bins=edges)
        
        observed_tran_p = observed_transition / np.sum(observed_transition)
        mode_usage_count = np.unique(translated_labels, return_counts=True)[1]
        marginal_prob = mode_usage_count.astype(float) / sum(mode_usage_count)
        posterior_trans_mat = np.zeros_like(translated_transmat)
        
        for i in xrange(HMM_GMM.HMM_model.n_components):
            posterior_trans_mat[i, :] = translated_transmat[i, :] * marginal_prob[i]
    
        Hamming_loss_lb = np.sum(np.abs(observed_tran_p - posterior_trans_mat)) * 0.5
        print(time_length)
        print('Hamming loss lower bound is %s') % Hamming_loss_lb
        
        np.sum(np.dot(observed_tran_p, posterior_trans_mat))
        np.sum(np.dot(observed_tran_p, observed_tran_p))
        DKL = scipy.stats.entropy(observed_tran_p.ravel(), posterior_trans_mat.ravel())
        
        print('DKL is %s.') % DKL
        data_storage[0, k, m] = time_length
        data_storage[1, k, m] = Hamming_loss_lb
        data_storage[2, k, m] = DKL

# %%
plt.figure()
plt.scatter(data_storage[0,:,:], data_storage[1,:,:])
plt.figure()
plt.scatter(data_storage[0,:,:], data_storage[2,:,:])

# pickle.dump(data_storage, open('error_data', 'wb'))
