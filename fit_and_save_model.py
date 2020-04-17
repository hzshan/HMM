import pickle
import numpy as np
from utils import get_dataset, generate_groups
from hmmlearn import hmm

choice = 5
windows = np.array([1, 5, 10, 30, 60, 120])
best_k = np.array([39, 26, 30, 13, 14, 7])

data = get_dataset(time_window=windows[choice])


fit_vec, test_vec, fit_len, test_len = generate_groups(data, subset=1, probs=(0.7, 0.3))


HMM_model = hmm.GaussianHMM(n_components=best_k[choice],
                            covariance_type='diag',
                            n_iter=150,
                            verbose=True)
HMM_model.fit(fit_vec, lengths=fit_len)
test_score = HMM_model.score(test_vec, lengths=test_len)

pickle.dump(HMM_model, open('HMM_tau_' + str(windows[choice]), 'wb'))


