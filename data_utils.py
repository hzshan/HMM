import numpy as np
import time
import pickle
from utils import normalize_trace
import sklearn
import os

class SegmentedData:

    def __init__(self, window_length, first_session_id, last_session_id, selection_starts, selection_ends):

        self.window_length = window_length
        self.start_grouped_analysis_at = first_session_id
        self.end_grouped_analysis_at = last_session_id
        self.tracking_resolution = 0.066

        self.selection_starts = selection_starts
        self.selection_ends = selection_ends

        self.processing_start_time = time.time()

        self.window_frame_count = int(self.window_length / self.tracking_resolution)

        self.g1_segment_counts = np.zeros(self.end_grouped_analysis_at - self.start_grouped_analysis_at + 2)
        self.g2_segment_counts = np.zeros_like(self.g1_segment_counts)

        self.g1_total_count = 0
        self.g2_total_count = 0

        self.segment_counts = np.zeros_like(self.g1_segment_counts)
        self.g1_segments = np.zeros((int(1000000 / window_length), self.window_frame_count * 2))
        self.g2_segments = np.zeros_like(self.g1_segments)

        self.g1_locations = np.zeros((1000000, 2))
        self.g2_locations = self.g1_locations.copy()

        self.g1_reduced_segments = None
        self.g2_reduced_segments = None

        self.all_segments = None
        self.all_reduced_segments = None
        self.pca_results = None

    def read_data(self, sampling_redundancy, pickle_file_header, rotation=True):

        processing_start_time = time.time()

        for session_ind in range(self.start_grouped_analysis_at, self.end_grouped_analysis_at + 1):

            file_name = pickle_file_header + '%s.pkl' % session_ind

            if os.path.isfile(file_name) is False:
                print('File ' + file_name + ' not found.')

            loaded_data = pickle.load(open(file_name, 'rb'))  # Load pickle file for analysis
            print("Currently analyzing " + file_name)

            if int(loaded_data.ot) < 35:
                print('Session excluded because there is a recorded opening')
                continue  # skip sessions where an opening occurred

            x_coors, y_coors, time_series = \
                normalize_trace(loaded_data,
                                self.selection_starts,
                                self.selection_ends)

            pos = 0  # position of cutter
            total_potential_seg = 0
            total_used_seg = 0
            for i in range(len(x_coors) - self.window_frame_count):

                if pos > len(x_coors) - self.window_frame_count:  # stop generating segments when running out of data
                    break

                origin_position = np.matrix((x_coors[pos], y_coors[pos]))

                raw_segment = np.array((x_coors[pos: pos + self.window_frame_count],
                                        y_coors[pos: pos + self.window_frame_count]))

                pos += int(self.window_frame_count * (1 - sampling_redundancy))  # move cutter

                first_x = np.min(raw_segment[0, :])
                last_x = np.max(raw_segment[0, :])
                first_y = np.min(raw_segment[1, :])
                last_y = np.max(raw_segment[1, :])
                displacement = np.sqrt((last_x - first_x) ** 2 + (last_y - first_y) ** 2)

                total_potential_seg += 1

                if displacement < 1:
                    continue

                moved_segment = raw_segment - np.tile(origin_position.T, (1, self.window_frame_count))

                direction_basis_vector = np.array(np.asarray(moved_segment[:, 1]))

                third_edge = np.sqrt(np.sum(direction_basis_vector ** 2))

                if third_edge == 0:
                    continue

                total_used_seg += 1  # count segments saved into data

                cos = direction_basis_vector[0, 0] / third_edge
                sin = direction_basis_vector[1, 0] / third_edge

                rotation_matrix = np.matrix([[cos, sin], [-sin, cos]])
                rotated_segment = np.zeros_like(moved_segment)

                for m in range(len(moved_segment.T)):
                    rotated_segment[:, m] = np.matmul(rotation_matrix, moved_segment[:, m])
                self.segment_counts[session_ind] += 1

                if rotation is False:
                    rotated_segment = moved_segment

                if loaded_data.o == 'opener':
                    self.g1_segment_counts[session_ind] += 1
                    g1_total_count = int(np.sum(self.g1_segment_counts))

                    self.g1_segments[g1_total_count, :] = np.hstack((rotated_segment[0, :], rotated_segment[1, :]))

                    self.g1_locations[g1_total_count, :] = np.mean(raw_segment, axis=1)

                elif loaded_data.o == 'nonopener':
                    self.g2_segment_counts[session_ind] += 1
                    g2_total_count = int(np.sum(self.g2_segment_counts))

                    self.g2_segments[g2_total_count, :] = np.hstack((rotated_segment[0, :], rotated_segment[1, :]))
                    self.g2_locations[g2_total_count, :] = np.mean(raw_segment, axis=1)

                else:
                    print('Neither opener nor non-opener??')
                    print(loaded_data.o)

            print("Data segments are made and stored.")

            print(str(total_used_seg) + ' / ' + str(total_potential_seg) + ' of all segments used.')

            self.g1_total_count = int(np.sum(self.g1_segment_counts))
            self.g2_total_count = int(np.sum(self.g2_segment_counts))

            print('Total number of segments so far ', self.g1_total_count + self.g2_total_count)

        self.g1_segment_counts = \
            np.ma.masked_where(self.g1_segment_counts == 0, self.g1_segment_counts).compressed()
        self.g2_segment_counts = \
            np.ma.masked_where(self.g2_segment_counts == 0, self.g2_segment_counts).compressed()

        self.g1_segments = self.g1_segments[0:self.g1_total_count, :]
        self.g2_segments = self.g2_segments[0:self.g2_total_count, :]

        self.g1_locations = self.g1_locations[0:self.g1_total_count, :]
        self.g2_locations = self.g2_locations[0:self.g2_total_count, :]

        print("################# Segmentation: Time elapsed:" +
              str(np.round((time.time() - processing_start_time))) + ' sec. ')

    def pca(self, n_components):

        """
        Performs principle component analysis on the coordinate trajectories.

        Argument:
        n_components: number of components in the PCA.

        Return:
        self.vector_PCA PCA object
        self.DR_vectors_group1 dimensionality - reduced coordinate trajectories
        self.DR_vectors_group2 dimensionality - reduced coordinate trajectories
        """

        self.all_segments = np.vstack((self.g1_segments, self.g2_segments))

        self.pca_results = sklearn.decomposition.PCA(n_components=n_components)
        self.pca_results.fit(self.all_segments)
        self.g1_reduced_segments = self.pca_results.transform(self.g1_segments)
        self.g2_reduced_segments = self.pca_results.transform(self.g2_segments)
        self.all_reduced_segments = self.pca_results.transform(self.all_segments)
