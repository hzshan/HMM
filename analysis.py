import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from utils import generate_grid_count, normalize_trace
import scipy.stats


class Session:
    def __init__(self, pickle_title, video_number, time_start,
                 time_end, grid_size):
        self.pickle_name = pickle_title + "%s.pkl" % video_number
        self.data = pickle.load(open(self.pickle_name,'rb'))
        (self.x_coors,
        self.y_coors,
        self.selected_time) = normalize_trace(data=self.data,
                                              selection_starts=time_start,
                                              selection_ends=time_end)


        self.grid,\
        self.x_edges,\
        self.y_edges,\
        self.x_center,\
        self.y_center = generate_grid_count(self.data, grid_size=grid_size)


    def make_hist(self, figsize=(4, 4)):
        plt.figure(figsize=figsize)
        plt.imshow(ind_session.grid, interpolation='none')
        plt.xlabel('X grid count')
        plt.ylabel('Y grid count')
        plt.colorbar()
        plt.title('Headmap for the rat position')
        plt.tight_layout()

ind_session = Session(pickle_title='propranolol_data',
                      video_number=2,
                      time_start=100,
                      time_end=500,
                      grid_size=2)

ind_session.make_hist()


#%% GROUP-WISE ANALYSIS (DO ABOVE ANALYSIS FIRST)
#=============================================================
start_grouped_analysis_at=1 #for multi-session analysis
end_grouped_analysis_at=64
select_time_segment=True
selection_starts=1
selection_ends=600 #in seconds
#=============================================================

processing_start_time = time.time()

open_sessions = []
close_sessions = []
open_grids = None
close_grids = None
for session_ind in range(start_grouped_analysis_at, end_grouped_analysis_at + 1):
    session = Session(pickle_title='propranolol_data',
                      video_number=session_ind,
                      time_start=100,
                      time_end=500,
                      grid_size=2)

    if session.data.o == 'opener':
        print('Session #' + str(session_ind) + ' is sorted into ' + session.data.o)
        if not open_sessions:
            open_grids = session.grid
        else:
            open_grids = np.dstack((open_grids, session.grid))
        open_sessions.append(session)

    if session.data.o == 'nonopener':
        print('Session #' + str(session_ind) + ' is sorted into ' + session.data.o)
        if not close_sessions:
            close_grids = session.grid
        else:
            close_grids = np.dstack((close_grids, session.grid))
        close_sessions.append(session)

open_count = len(open_sessions)
close_count = len(close_sessions)

open_mean_grid = np.sum(open_grids, axis=2) / open_count
close_mean_grid = np.sum(close_grids, axis=2) / close_count
between_group_mean = np.mean(np.append(open_grids, close_grids, axis=2),axis=2)
open_grid_ste = scipy.stats.sem(open_grids,axis=2)
close_grid_ste = scipy.stats.sem(close_grids,axis=2)

compare_grids = plt.figure(figsize=(6, 3))
compare_grids.add_subplot(121)
plt.imshow(open_mean_grid)
plt.colorbar()
plt.title('openers')
plt.tight_layout()
compare_grids.add_subplot(122)
plt.imshow(close_mean_grid)
plt.colorbar()
plt.title('non-openers')
plt.tight_layout()
