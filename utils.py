import numpy as np
import matplotlib.pyplot as plt
import pickle
import time



class ProcessedData:
    def __init__(self, filename, condition, opener, opening_time,
                 group, rat_id, date, timeline, x_coors, y_coors, timestep, arena_size):
        self.n = filename
        self.c = condition
        self.o = opener
        self.ot = opening_time
        self.g = group
        self.rat_id = rat_id
        self.d = date
        self.t = timeline
        self.x = x_coors
        self.y = y_coors
        self.timestep = timestep
        self.arena_size = arena_size


def PCA_summary(self, sampled_data):
        print(sampled_data.vector_PCA.explained_variance_ratio_)
        print("The first five principles components explain %02s of the variance.") % \
        np.sum(sampled_data.vector_PCA.explained_variance_ratio_[0:5])

        fig = plt.figure(figsize=(20, 4))

        figure_bound = 0.05
        neg_bound = '-A.U.'
        pos_bound = 'A.U.'
        sampled_data.PC1_x = sampled_data.vector_PCA.components_[0, 0:sampled_data.window_frame_count]
        sampled_data.PC1_y = sampled_data.vector_PCA.components_[0,
                             sampled_data.window_frame_count:(2 * sampled_data.window_frame_count)]
        ax1 = fig.add_subplot(151)
        ax1.plot(sampled_data.PC1_x, sampled_data.PC1_y, 'k', linewidth=3)
        ax1.scatter(0, 0, facecolors='none', edgecolors='k', s=120, label='start')
        ax1.scatter(sampled_data.PC1_x[-1], sampled_data.PC1_y[-1], color='k',
                    linewidth='8', label='end')
        plt.xticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20)
        plt.yticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20, rotation='vertical')

        plt.title('PC1', fontsize=30)

        sampled_data.PC2_x = sampled_data.vector_PCA.components_[1, 0:sampled_data.window_frame_count]
        sampled_data.PC2_y = sampled_data.vector_PCA.components_[1,
                             sampled_data.window_frame_count:(2 * sampled_data.window_frame_count)]
        ax2 = fig.add_subplot(152)
        ax2.plot(sampled_data.PC2_x, sampled_data.PC2_y, 'k', linewidth=3)
        ax2.scatter(0, 0, facecolors='none', edgecolors='k', s=120, label='start')
        ax2.scatter(sampled_data.PC2_x[-1], sampled_data.PC2_y[-1], color='k',
                    linewidth='8', label='end')
        plt.xticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20)
        plt.yticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20, rotation='vertical')
        plt.title('PC2', fontsize=30)

        sampled_data.PC3_x = sampled_data.vector_PCA.components_[2, 0:sampled_data.window_frame_count]
        sampled_data.PC3_y = sampled_data.vector_PCA.components_[2,
                             sampled_data.window_frame_count:(2 * sampled_data.window_frame_count)]
        ax3 = fig.add_subplot(153)
        ax3.plot(sampled_data.PC3_x, sampled_data.PC3_y, 'k', linewidth=3)
        ax3.scatter(0, 0, facecolors='none', edgecolors='k', s=120, label='start')
        ax3.scatter(sampled_data.PC3_x[-1], sampled_data.PC3_y[-1], color='k',
                    linewidth='8', label='end')
        plt.xticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20)
        plt.yticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20, rotation='vertical')
        plt.title('PC3', fontsize=30)

        sampled_data.PC4_x = sampled_data.vector_PCA.components_[3, 0:sampled_data.window_frame_count]
        sampled_data.PC4_y = sampled_data.vector_PCA.components_[3,
                             sampled_data.window_frame_count:(2 * sampled_data.window_frame_count)]
        ax4 = fig.add_subplot(154)
        ax4.plot(sampled_data.PC4_x, sampled_data.PC4_y, 'k', linewidth=3)
        ax4.scatter(0, 0, facecolors='none', edgecolors='k', s=120, label='start')
        ax4.scatter(sampled_data.PC4_x[-1], sampled_data.PC4_y[-1], color='k',
                    linewidth='8', label='end')
        plt.xticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20)
        plt.yticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20, rotation='vertical')
        plt.title('PC4', fontsize=30)

        sampled_data.PC5_x = sampled_data.vector_PCA.components_[4, 0:sampled_data.window_frame_count]
        sampled_data.PC5_y = sampled_data.vector_PCA.components_[4,
                             sampled_data.window_frame_count:(2 * sampled_data.window_frame_count)]
        ax5 = fig.add_subplot(155)
        ax5.plot(sampled_data.PC5_x, sampled_data.PC5_y, 'k', linewidth=3)
        ax5.scatter(0, 0, facecolors='none', edgecolors='k', s=120, label='start')
        ax5.scatter(sampled_data.PC5_x[-1], sampled_data.PC5_y[-1], color='k',
                    linewidth='8', label='end')
        plt.xticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20)
        plt.yticks((-figure_bound, 0, figure_bound), (neg_bound, 0, pos_bound),
                   fontsize=20, rotation='vertical')
        plt.title('PC5', fontsize=30)


def generate_groups(dataset, probs=(0.7, 0.3), subset=0.5):



    all_counts = dataset.segment_counts
    n_segments_used = int(subset * all_counts.shape[0])
    all_counts = all_counts[0: n_segments_used]
    all_segments = dataset.all_reduced_segments
    all_segments = all_segments[0:int(np.sum(all_counts)), :]

    seg_length = all_segments.shape[1]  # length of each segment

    random_grouping_mask = np.random.choice((1, 0), p=probs,
                                            size=len(all_counts))

    n_session = len(random_grouping_mask)
    n_training_session = int(np.sum(random_grouping_mask))
    n_testing_session = n_session - n_training_session

    segment_mask = np.array([])
    for _ind in range(n_session):
        segment_mask = np.append(segment_mask,
                                 np.ones(int(all_counts[_ind])) * random_grouping_mask[_ind])

    full_mask = np.repeat(np.matrix(segment_mask), seg_length, axis=0).T

    training_counts = np.ma.masked_where(random_grouping_mask == 1,
                                         all_counts).compressed()  # number of segments in each session
    testing_counts = np.ma.masked_where(random_grouping_mask == 0,
                                        all_counts).compressed()

    training_set = np.ma.masked_where(
        full_mask == 1, all_segments).compressed().reshape(int(np.sum(training_counts)), seg_length)
    testing_set = np.ma.masked_where(
        full_mask == 0, all_segments).compressed().reshape(int(np.sum(testing_counts)), seg_length)

    training_counts = training_counts[training_counts != 0]
    testing_counts = testing_counts[testing_counts != 0]

    return training_set, testing_set, training_counts, testing_counts


def load_series(first_row, last_row, row_title, worksheet):
    _counter = 0
    target = np.zeros((last_row - first_row + 1))
    data_range = '%s%i:%s%i' % (row_title, first_row, row_title, last_row)

    for row in worksheet[data_range]:
        for cell in row:
            if cell.value == '-':
                target[_counter] = 0
            else:
                target[_counter] = cell.value
                _counter = _counter + 1

    return target


def generate_grid_count(data,
                        grid_size):
    _y_center_coor = int(data.arena_size / grid_size / 2)
    _x_center_coor = _y_center_coor

    x_edges = np.arange(-data.arena_size / 2, data.arena_size / 2, grid_size)
    y_edges = x_edges.copy()
    h, x_edges, y_edges = np.histogram2d(
        data.x, data.y, bins=(x_edges, y_edges))

    grid = h * data.timestep

    center_ub = 5 + _y_center_coor
    center_lb = - 5 + _y_center_coor
    center_left = - 5 + _x_center_coor
    center_right = 5 + _x_center_coor
    central_region = grid[center_left:center_right,
                     center_lb:center_ub]
    central_region = np.clip(central_region, np.min(central_region),
                             np.percentile(central_region, 80))

    grid[center_left:center_right,
    center_lb:center_ub] = central_region

    grid /= np.sum(grid)  # normalize count such that it's probability

    return grid, x_edges, y_edges, _x_center_coor, _y_center_coor


def normalize_trace(data,
                    selection_starts,
                    selection_ends,
                    left_edge_smoothing_sample_size=100,
                    right_edge_smoothing_sample_size=100,
                    upper_edge_smoothing_sample_size=100,
                    lower_edge_smoothing_sample_size=100):
    _x_coors = data.x
    _y_coors = data.y

    _x_axis_lb = np.mean(
        np.sort(_x_coors, axis=None)[:left_edge_smoothing_sample_size])

    _x_axis_ub = np.mean(np.sort(
        _x_coors, axis=None)[::-1][:right_edge_smoothing_sample_size])

    _y_axis_lb = np.mean(np.sort(
        _y_coors, axis=None)[:lower_edge_smoothing_sample_size])

    _y_axis_ub = np.mean(np.sort(
        _y_coors, axis=None)[::-1][:upper_edge_smoothing_sample_size])

    for i in range(int(len(_x_coors))):
        if _x_coors[i] < _x_axis_lb or \
                _x_coors[i] > _x_axis_ub or \
                _y_coors[i] < _y_axis_lb or \
                _y_coors[i] > _y_axis_ub:
            _x_coors[i] = 0
            _y_coors[i] = 0
    _x_center_coor = 0.5 * (np.max(_x_coors) + np.min(_x_coors))
    _y_center_coor = 0.5 * (np.max(_y_coors) + np.min(_y_coors))
    _x_scale = data.arena_size / (np.max(_x_coors) - np.min(_x_coors))
    _y_scale = data.arena_size / (np.max(_y_coors) - np.min(_y_coors))
    _rescaled_x = (_x_coors - _x_center_coor) * _x_scale
    _rescaled_y = (_y_coors - _y_center_coor) * _y_scale

    selection_starts = int(selection_starts / data.timestep)
    selection_ends = int(selection_ends / data.timestep)
    _rescaled_x = _rescaled_x[selection_starts:selection_ends]
    _rescaled_y = _rescaled_y[selection_starts:selection_ends]
    selected_time_series = data.t[selection_starts:selection_ends]

    return _rescaled_x, _rescaled_y, selected_time_series


def get_dataset(time_window):
    filename = 'segments_tau_' + str(time_window)
    dataset = pickle.load(open(filename, 'rb'))

    return dataset

def construct(motif_index, principle_components, motifs):
    _raw = np.zeros_like(principle_components[0, :])
    for _i in range(5):
        _raw += motifs[motif_index, _i] * principle_components[_i, :]
    true_len = int(len(_raw) / 2)
    _output = np.zeros((2, true_len))
    _output[0, :] = _raw[0:true_len]
    _output[1, :] = _raw[true_len:]

    return _output