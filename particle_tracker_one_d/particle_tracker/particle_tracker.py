import numpy as np
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from .trajectory import Trajectory


class ParticleTracker:
    def __init__(self, intensity, time):
        self.time = time
        self._intensity = intensity
        self._expected_width_of_particle = 1
        self._boxcar_width = 0
        self._feature_point_threshold = 1
        self._particle_discrimination_threshold = 0
        self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 1
        self._maximum_distance_a_particle_can_travel_between_frames = 1
        self._averaged_intensity = intensity
        self._trajectories = []
        self._association_matrix = {}
        self._cost_matrix = {}
        self._particle_positions = np.empty((0, 2), dtype=np.int16)
        self._update_averaged_intensity()
        self._update_particle_positions()
        self._update_association_matrix()
        self._update_trajectories()

    @property
    def intensity(self):
        return self._averaged_intensity

    @property
    def sigma_0(self):
        return 0.1 * np.pi * self.expected_width_of_particle ** 2

    @property
    def sigma_2(self):
        return 0.1 * np.pi * self.expected_width_of_particle ** 2

    @property
    def boxcar_width(self):
        return self._boxcar_width

    @boxcar_width.setter
    def boxcar_width(self, width):
        if not width == self._boxcar_width:
            self._boxcar_width = width
            self._update_averaged_intensity()
            self._update_particle_positions()
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def expected_width_of_particle(self):
        return self._expected_width_of_particle

    @expected_width_of_particle.setter
    def expected_width_of_particle(self, width):
        if not width == self._expected_width_of_particle:
            self._expected_width_of_particle = width
            self._update_particle_positions()
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def feature_point_threshold(self):
        return self._feature_point_threshold

    @feature_point_threshold.setter
    def feature_point_threshold(self, threshold):
        if not threshold == self._feature_point_threshold:
            self._feature_point_threshold = threshold
            self._update_particle_positions()
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def particle_discrimination_threshold(self):
        return self._particle_discrimination_threshold

    @particle_discrimination_threshold.setter
    def particle_discrimination_threshold(self, threshold):
        if not threshold == self._particle_discrimination_threshold:
            self._particle_discrimination_threshold = threshold
            # self._discriminate_particles

    @property
    def maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles(self):
        return self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles

    @maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles.setter
    def maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles(self, number_of_frames):
        if not number_of_frames == self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:
            self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = number_of_frames
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def maximum_distance_a_particle_can_travel_between_frames(self):
        return self._maximum_distance_a_particle_can_travel_between_frames

    @maximum_distance_a_particle_can_travel_between_frames.setter
    def maximum_distance_a_particle_can_travel_between_frames(self, distance):
        if not distance == self._maximum_distance_a_particle_can_travel_between_frames:
            self._maximum_distance_a_particle_can_travel_between_frames = distance
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def association_matrix(self):
        return self._association_matrix

    @property
    def particle_positions(self):
        return self._particle_positions

    def get_intensity_at_time(self, time):
        index = self._find_index_of_nearest(self.time, time)
        return self._averaged_intensity[index]

    def plot_averaged_intensity(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.axes()
        ax.imshow(self._averaged_intensity, **kwargs)
        return ax

    def plot_intensity_at_time(self, time, ax=None, **kwargs):
        intensity = self.get_intensity_at_time(time)
        if ax is None:
            ax = plt.axes()
        ax.plot(intensity, **kwargs)
        return ax

    def plot_intensity_of_frame(self, frame_nr, ax=None, **kwargs):
        intensity = self._averaged_intensity[frame_nr]
        if ax is None:
            ax = plt.axes()
        ax.plot(intensity, **kwargs)
        return ax

    def _update_trajectories(self):
        self._trajectories = []
        count = 0
        for index, point in enumerate(self._particle_positions):
            if not self._is_particle_position_already_used_in_trajectory(point):
                self._trajectories.append(Trajectory())
                self._trajectories[count].append_position(point)
                for index_future_points, future_point in enumerate(self._particle_positions[index + 1:]):
                    if self._points_are_linked(point, future_point):
                        self._trajectories[count].append_position(future_point)
                        point = future_point
                count += 1

    def _update_association_matrix(self):
        self._initialise_empty_association_matrix()
        self._initialise_empty_cost_matrix()
        self._calculate_cost_matrix()
        self._create_initial_links_in_association_matrix()
        self._optimise_association_matrix()

    def _update_averaged_intensity(self):
        if self.boxcar_width == 0:
            self._averaged_intensity = self._intensity
        else:
            self._averaged_intensity = np.empty(self._intensity.shape)
            kernel = Box1DKernel(self.boxcar_width)
            for row_index, row_intensity in enumerate(self._intensity):
                self._averaged_intensity[row_index] = convolve(row_intensity, kernel)

    def _update_particle_positions(self):
        initial_particle_positions = self._get_positions_of_intensity_maximas()
        refined_particle_positions = self._refine_particle_positions(initial_particle_positions)
        self._particle_positions = self._perform_particle_discrimination(refined_particle_positions)

    def _get_positions_of_intensity_maximas(self):
        positions = np.empty((0, 2), dtype=np.uint16)
        for row_index, row_intensity in enumerate(self._averaged_intensity):
            columns_with_local_maximas = np.r_[row_intensity[:-1] > row_intensity[1:], True] & \
                                         np.r_[True, row_intensity[1:] > row_intensity[:-1]] & \
                                         np.r_[row_intensity > self.feature_point_threshold]
            for col_index, is_max in enumerate(columns_with_local_maximas):
                if is_max:
                    positions = np.append(positions, np.array([[row_index, col_index]]), axis=0)
        return positions

    def _refine_particle_positions(self, particle_positions):
        if self.expected_width_of_particle != 0:
            for row_index, position in enumerate(particle_positions):
                particle_positions[row_index, 1] = self._find_center_of_mass_close_to_position(position)
        return particle_positions

    def _find_center_of_mass_close_to_position(self, particle_position):
        if particle_position[1] == 0:
            return 0
        if particle_position[1] <= self.expected_width_of_particle:
            width = particle_position[1]
        elif particle_position[1] >= self._averaged_intensity.shape[1] - self.expected_width_of_particle:
            width = self._averaged_intensity.shape[1] - particle_position[1]
        else:
            width = self._expected_width_of_particle
        intensity = self._averaged_intensity[particle_position[0], particle_position[1] - width:particle_position[1] + width]
        return particle_position[1] + self._calculate_center_of_mass(intensity) - width

    def _perform_particle_discrimination(self, particle_positions):
        particle_positions = self._remove_particles_with_wrong_moment(particle_positions)
        particle_positions = self._remove_particles_too_closely_together(particle_positions)
        return particle_positions

    def _remove_particles_with_wrong_moment(self, particle_positions):
        if self.particle_discrimination_threshold == 0:
            return particle_positions
        index_of_particles_to_be_kept = []
        for row_index, position in enumerate(particle_positions):
            if self._calculate_discrimination_score_for_particle(position) >= self.particle_discrimination_threshold:
                index_of_particles_to_be_kept.append(row_index)
        return particle_positions[index_of_particles_to_be_kept]

    def _calculate_discrimination_score_for_particle(self, particle_position):
        score = 0
        particle_positions_of_particles_in_same_frame = self._get_particle_positions_in_frame(frame_index=particle_position[0])
        particle_positions_of_particles_in_same_frame = particle_positions_of_particles_in_same_frame[
            np.where(particle_positions_of_particles_in_same_frame[:, 1] != particle_position[1])]
        for index, position in enumerate(particle_positions_of_particles_in_same_frame):
            score += self._calculate_gaussian_moment(particle_position, position)
        return score

    def _calculate_gaussian_moment(self, particle_position_1, particle_position_2):
        return 1 / (2 * np.pi * self.sigma_0 * self.sigma_2 * self._particle_positions.shape[0]) * \
               np.exp(-(self._calculate_first_order_intensity_moment(particle_position_1) - self._calculate_first_order_intensity_moment(particle_position_2)) ** 2 / (
                       2 * self.sigma_0)
                      - (self._calculate_second_order_intensity_moment(particle_position_1) - self._calculate_second_order_intensity_moment(particle_position_2)) ** 2 / (
                              2 * self.sigma_2))

    def _calculate_second_order_intensity_moment(self, particle_position):
        if particle_position[1] == 0:
            return 0
        if particle_position[1] < self.expected_width_of_particle:
            w = particle_position[1]
        elif particle_position[1] > self._intensity.shape[1] - self.expected_width_of_particle:
            w = self._intensity.shape[1] - particle_position[1]
        else:
            w = self.expected_width_of_particle
        return np.sum(
            np.arange(-w, w) ** 2 * self.intensity[particle_position[0], particle_position[1] - w: particle_position[1] + w]) / \
               self._calculate_first_order_intensity_moment(particle_position)

    def _calculate_first_order_intensity_moment(self, particle_position):
        if particle_position[1] == 0:
            return particle_position[0]
        if particle_position[1] < self.expected_width_of_particle:
            w = particle_position[1]
        elif particle_position[1] > self._intensity.shape[1] - self.expected_width_of_particle:
            w = self._intensity.shape[1] - particle_position[1]
        else:
            w = self.expected_width_of_particle
        return np.sum(self.intensity[particle_position[0], particle_position[1] - w: particle_position[1] + w])

    def _get_particle_positions_in_frame(self, frame_index):
        return self._particle_positions[np.where(self._particle_positions[:, 0] == frame_index)]

    def _remove_particles_too_closely_together(self, particle_positions):
        index_of_particles_to_be_removed = []
        for index_1, p1 in enumerate(particle_positions[:-1]):
            for index_2, p2 in enumerate(particle_positions[index_1 + 1:]):
                if self._particles_are_too_close(p1, p2):
                    index_of_particles_to_be_removed.append(self._index_of_particle_with_lowest_first_order_moment(index_1, index_1 + index_2, particle_positions))
        return np.delete(particle_positions, index_of_particles_to_be_removed, axis=0)

    def _index_of_particle_with_lowest_first_order_moment(self, particle_index_1, particle_index_2, particle_positions):
        if self._calculate_first_order_intensity_moment(particle_positions[particle_index_1]) < self._calculate_first_order_intensity_moment(
                particle_positions[particle_index_2]):
            return particle_index_1
        return particle_index_2

    def _particles_are_too_close(self, position1, position2):
        return position1[0] == position2[0] and np.abs(position2[1] - position1[1]) < self.expected_width_of_particle

    def _initialise_empty_association_matrix(self):
        self._association_matrix = {}
        for index, t in enumerate(self.time):
            number_of_particles_at_t = np.count_nonzero(self._particle_positions[:, 0] == index)
            self._association_matrix[str(index)] = {}
            for r in range(1, self.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles + 1):
                if r + index < len(self.time):
                    number_of_particles_at_t_plus_r = np.count_nonzero(self._particle_positions[:, 0] == index + r)
                    self._association_matrix[str(index)][str(r)] = np.zeros(
                        (number_of_particles_at_t + 1, number_of_particles_at_t_plus_r + 1), dtype=np.int16)

    def _create_initial_links_in_association_matrix(self):
        for frame_index, frame_key in enumerate(self._association_matrix.keys()):
            for future_frame_index, future_frame_key in enumerate(self._association_matrix[frame_key].keys()):
                self._association_matrix[frame_key][future_frame_key] = self._initialise_link_matrix(self._association_matrix[frame_key][future_frame_key], frame_key,
                                                                                                     future_frame_key)

    def _initialise_empty_cost_matrix(self):
        self._cost_matrix = {}
        for frame_index, frame_key in enumerate(self._association_matrix.keys()):
            self._cost_matrix[frame_key] = {}
            for future_frame_index, future_frame_key in enumerate(self._association_matrix[frame_key].keys()):
                self._cost_matrix[frame_key][future_frame_key] = np.zeros(self._association_matrix[frame_key][future_frame_key].shape, dtype=np.float32)

    def _calculate_cost_matrix(self):
        for frame_index, frame_key in enumerate(self._cost_matrix.keys()):
            for future_frame_index, future_frame_key in enumerate(self._cost_matrix[frame_key].keys()):
                cost_for_association_with_dummy_particle = self._calculate_cost_for_association_with_dummy_particle(future_frame_index + 1)
                particle_positions_in_current_frame = self._get_particle_positions_in_frame(frame_index)
                particle_positions_in_future_frame = self._get_particle_positions_in_frame(frame_index + future_frame_index + 1)
                for row_index, row in enumerate(self._cost_matrix[frame_key][future_frame_key]):
                    for col_index, value in enumerate(self._cost_matrix[frame_key][future_frame_key][row_index]):
                        if row_index == 0 or col_index == 0:
                            self._cost_matrix[frame_key][future_frame_key][row_index][col_index] = cost_for_association_with_dummy_particle
                        else:
                            position1 = particle_positions_in_current_frame[row_index - 1]
                            position2 = particle_positions_in_future_frame[col_index - 1]
                            self._cost_matrix[frame_key][future_frame_key][row_index][col_index] = self._calculate_linking_cost(position1, position2)

    def _initialise_link_matrix(self, link_matrix, frame_key, future_frame_key):
        for row_index, costs in enumerate(self._cost_matrix[frame_key][future_frame_key]):
            if not row_index == 0:
                col_index = np.where(costs == np.amin(costs))[0][0]
                if (not (link_matrix[:, col_index] == 1).any()) or col_index == 0:
                    link_matrix[row_index][col_index] = 1
        return self._fill_in_empty_rows_and_columns(link_matrix)

    def _calculate_linking_cost(self, position1, position2):
        return (
                (position1[1] - position2[1]) ** 2 +
                (self._calculate_first_order_intensity_moment(position1) - self._calculate_first_order_intensity_moment(position2)) ** 2 +
                (self._calculate_second_order_intensity_moment(position1) - self._calculate_second_order_intensity_moment(position2)) ** 2
        )

    def _calculate_cost_for_association_with_dummy_particle(self, future_frame_index):
        return (self.maximum_distance_a_particle_can_travel_between_frames * future_frame_index) ** 2

    def _optimise_association_matrix(self):
        for frame_index, frame_key in enumerate(self._association_matrix.keys()):
            for future_frame_index, future_frame_key in enumerate(self._association_matrix[frame_key].keys()):
                link_matrix = self._association_matrix[frame_key][future_frame_key]
                self._association_matrix[frame_key][future_frame_key] = self._optimise_link_matrix(link_matrix, frame_key, future_frame_key)
        return

    def _optimise_link_matrix(self, link_matrix, frame_key, future_frame_key):
        link_matrix_is_optimal = False
        while not link_matrix_is_optimal:
            link_matrix_is_optimal = True
            for row_index, row in enumerate(link_matrix):
                for col_index, val in enumerate(row):
                    if val == 0:
                        if col_index > 0 and row_index > 0:
                            introduction_cost = self._cost_matrix[frame_key][future_frame_key][row_index][col_index]
                            row_index_with_link = np.where(link_matrix[:, col_index] == 1)[0][0]
                            col_index_with_link = np.where(link_matrix[row_index, :] == 1)[0][0]
                            reduction_cost_row = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][col_index]
                            reduction_cost_col = self._cost_matrix[frame_key][future_frame_key][row_index][col_index_with_link]
                            introduction_row_col = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][col_index_with_link]
                            total_cost = introduction_cost - reduction_cost_row - reduction_cost_col + introduction_row_col

                            if total_cost < 0:
                                link_matrix[row_index][col_index] = 1
                                link_matrix[row_index][col_index_with_link] = 0
                                link_matrix[row_index_with_link][col_index] = 0
                                link_matrix[row_index_with_link][col_index_with_link] = 1
                                link_matrix_is_optimal = False

                        elif row_index == 0 and col_index > 0:
                            introduction_cost = self._cost_matrix[frame_key][future_frame_key][row_index][col_index]
                            row_index_with_link = np.where(link_matrix[:, col_index] == 1)[0][0]
                            reduction_cost_row = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][col_index]
                            introduction_row = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][0]
                            total_cost = introduction_cost - reduction_cost_row + introduction_row

                            if total_cost < 0:
                                link_matrix[row_index][col_index] = 1
                                link_matrix[row_index_with_link][col_index] = 0
                                link_matrix[row_index_with_link][0] = 1
                                link_matrix_is_optimal = False

                        elif row_index > 0 and col_index == 0:
                            introduction_cost = self._cost_matrix[frame_key][future_frame_key][row_index][col_index]
                            col_index_with_link = np.where(link_matrix[row_index][:] == 1)[0][0]
                            reduction_cost_col = self._cost_matrix[frame_key][future_frame_key][row_index][col_index_with_link]
                            introduction_col = self._cost_matrix[frame_key][future_frame_key][0][col_index_with_link]
                            total_cost = introduction_cost - reduction_cost_col + introduction_col

                            if total_cost < 0:
                                link_matrix[row_index][col_index] = 1
                                link_matrix[row_index][col_index_with_link] = 0
                                link_matrix[0][col_index_with_link] = 1
                                link_matrix_is_optimal = False

        return link_matrix

    def _is_particle_position_already_used_in_trajectory(self, particle_position):
        for trajectory in self._trajectories:
            if trajectory.position_exists_in_trajectory(particle_position):
                return True
        return False

    def _points_are_linked(self, point, future_point):
        if point[0] == future_point[0]:
            return False
        nr_of_frames_between_points = self._calculate_number_of_frames_between_points(point, future_point)
        if nr_of_frames_between_points <= self.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:

            time_key = str(point[0])
            r_key = str(nr_of_frames_between_points)

            link_matrix = self._association_matrix[time_key][r_key]

            points_in_same_frame_as_point = self._get_particle_positions_in_frame(point[0])
            points_in_same_frame_as_future_point = self._get_particle_positions_in_frame(future_point[0])

            index_of_point = np.where(points_in_same_frame_as_point[:, 1] == point[1])[0][0]
            index_of_future_point = np.where(points_in_same_frame_as_future_point[:, 1] == future_point[1])[0][0]
            return int(link_matrix[index_of_point + 1][index_of_future_point + 1]) == 1
        else:
            return False

    @staticmethod
    def _calculate_center_of_mass(y):
        x = np.arange(0, y.shape[0])
        return np.sum(x * y) / np.sum(y)

    @staticmethod
    def _calculate_number_of_frames_between_points(p1, p2):
        return int(p2[0] - p1[0])

    @staticmethod
    def _fill_in_empty_rows_and_columns(link_matrix):
        for row_index, row in enumerate(link_matrix):
            if not (row == 1).any():
                link_matrix[row_index][0] = 1
        for col_index, col in enumerate(link_matrix.T):
            if not (col == 1).any():
                link_matrix[0][col_index] = 1
        return link_matrix

    @staticmethod
    def normalise_intensity(intensity):
        intensity = intensity - np.amin(intensity)
        return intensity / np.amax(intensity)

    @staticmethod
    def _find_index_of_nearest(array, value):
        return (np.abs(np.array(array) - value)).argmin()
