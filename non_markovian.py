import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from utils import get_dataset
import pickle


def prob_from_labels(labels, step):
    n_mode = int(np.max(labels) + 1)
    _transmat = np.zeros((n_mode, n_mode))
    _marginal = np.zeros(n_mode)
    for i in range(len(labels)):
        if i + step == len(labels):
            break
        _transmat[int(labels[i]), int(labels[i + step])] += 1
        _marginal[int(labels[i])] += 1
    for i in range(n_mode):
        if np.sum(_transmat[i, :]) == 0:
            continue
        _transmat[i, :] /= np.sum(_transmat[i, :])
    return _transmat, _marginal / np.sum(_marginal)



def make_joint(transmat, marginals):
    assert transmat.shape[0] == len(marginals)
    results = np.zeros_like(transmat)
    for i in range(len(marginals)):
        results[i, :] = transmat[i, :] * marginals[i]
    return results


def dkl(joint1, joint2):
    flat_1 = joint1.flatten() + 1e-4
    flat_2 = joint2.flatten() + 1e-4

    result = 0
    for i in range(len(flat_1)):
        result += flat_1[i] * np.log(flat_1[i] / flat_2[i])
    return result


windows = [1, 5, 10, 30, 60, 120]


jumps = np.arange(1, 10, 1)
all_dkls = np.zeros((6, len(jumps)))
for window_ind in range(6):

    window = windows[window_ind]
    fitted_model = pickle.load(open('HMM_tau_' + str(window), 'rb'))
    dataset = get_dataset(window)

    all_labels = fitted_model.decode(dataset.all_reduced_segments)[1]

    for i in range(len(jumps)):

        M = fitted_model.transmat_
        recovered_transmat, marginals = prob_from_labels(all_labels, jumps[i])
        recovered_joint = make_joint(recovered_transmat, marginals)
        predicted_joint = make_joint(M**int(jumps[i]), marginals)
        all_dkls[window_ind, i] = dkl(recovered_joint, predicted_joint)

plt.figure(figsize=(4, 3))
for i in range(6):
    plt.plot(jumps, all_dkls[i, :],
             label='$\\tau=$' + str(windows[i]),
             linewidth=1.5)
plt.xlabel('jump size', fontsize=14)
plt.ylabel('Divergence', fontsize=14)
plt.xticks(jumps, fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid()
plt.tight_layout()





