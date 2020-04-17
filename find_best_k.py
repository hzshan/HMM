import numpy as np
from utils import get_dataset
from utils import generate_groups
from hmmlearn import hmm
import matplotlib.pyplot as plt

time_window = 120
ks = np.arange(1, 15, 2)
LL = np.zeros_like(ks)
BIC_scores = np.zeros_like(ks)

data = get_dataset(time_window=time_window)

ratio = 1

fit_vec, test_vec, fit_len, test_len = generate_groups(data, subset=ratio)

for i in range(len(ks)):

    n_components = ks[i]
    print('current k', n_components)
    HMM_model = hmm.GaussianHMM(n_components=n_components,
                                covariance_type='diag',
                                n_iter=150,
                                verbose=True)
    HMM_model.fit(fit_vec, lengths=fit_len)
    test_score = HMM_model.score(test_vec, lengths=test_len)
    n_parameters = HMM_model.n_components + HMM_model.n_components * fit_vec.shape[1]
    + np.power(HMM_model.n_components, 2) + HMM_model.n_components * np.power(fit_vec.shape[1], 2)
    LL[i] = test_score
    BIC_scores[i] = np.log(np.mean(test_len)) * n_parameters - 2 * test_score
    print('BIC_score', BIC_scores[i])

plt.figure()
plt.plot(ks, BIC_scores)
plt.title('BIC')
plt.xlabel('k')

"""
Best K:
1s - 39
5s - 26
10s - 30
30s - 13
60s - 14
120s - 7
"""
