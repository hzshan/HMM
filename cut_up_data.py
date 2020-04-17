import numpy as np
import pickle
from utils import SegmentedData

one_sec = SegmentedData(window_length=1,
                        first_session_id=1,
                        last_session_id=12,
                        selection_starts=60,
                        selection_ends=2400)

one_sec.read_data(sampling_redundancy=0.5, pickle_file_header='propranolol_data', rotation=True)

one_sec.pca(n_components=5)

window_length = np.array([1, 5, 10, 30, 60, 120])

for window in window_length:
    data = SegmentedData(window_length=window,
                         first_session_id=1,
                         last_session_id=144,
                         selection_starts=60,
                         selection_ends=2400)

    data.read_data(sampling_redundancy=0.5, pickle_file_header='propranolol_data', rotation=True)

    data.pca(n_components=5)

    database_name = 'segments_tau_%s' % int(window)
    pickle.dump(data, open(database_name, 'wb'))
