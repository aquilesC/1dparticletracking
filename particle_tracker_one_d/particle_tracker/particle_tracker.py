import numpy as np
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from .trajectory import Trajectory


class ParticleTracker:
    """
    Dynamic Particle tracker object which finds trajectories in the frames. Trajectories are automatically updated when properties are changed.

    Parameters
    ----------
    frames: np.array
        The frames in which trajectories are to be found. The shape of the np.array should be (nFrames,xPixels). The intensity of the frames should be normalised according to
        :math:`I_n = (I-I_{min})/(I_{max}-I_{min})`, where :math:`I` is the intensity of the frames, :math:`I_{min}`, :math:`I_{max}` are the global intensity minima and maxima of the
        frames.
    time: np.array
        The corresponding time of each frame.

    Attributes
    ----------
    frames
    time
    boxcar_width
    integration_radius_of_intensity_peaks
    feature_point_threshold
    particle_discrimination_threshold
    maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles
    maximum_distance_a_particle_can_travel_between_frames
    particle_positions
    """

    def __init__(self, frames, time):
        self._time = time
        self._frames = frames
        self._integration_radius_of_intensity_peaks = 1
        self._boxcar_width = 0
        self._feature_point_threshold = 1
        self._particle_discrimination_threshold = 0
        self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 1
        self._maximum_distance_a_particle_can_travel_between_frames = 1
        self._averaged_intensity = frames
        self._trajectories = []
        self._association_matrix = {}
        self._cost_matrix = {}
        self._particle_positions = np.empty(1, dtype=[('frame_index', np.int16), ('time', np.float32),
                                                      ('integer_position', np.int16), ('refined_position', np.float32)])
        self._update_averaged_intensity()
        self._update_particle_positions()
        self._update_association_matrix()
        self._update_trajectories()

    @property
    def frames(self):
        """
        np.array:
            The frames which the particle tracker tries to find trajectories in. If the property boxcar_width!=0 it will return the smoothed frames.
        """
        return self._averaged_intensity

    @property
    def _sigma_0(self):
        return 0.1 * np.pi * self.integration_radius_of_intensity_peaks ** 2

    @property
    def _sigma_2(self):
        return 0.1 * np.pi * self.integration_radius_of_intensity_peaks ** 2

    @property
    def boxcar_width(self):
        """
        int:
            Number of values used in the boxcar averaging of the frames.
        """
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
    def integration_radius_of_intensity_peaks(self):
        """
        int:
            Number of pixels used when integrating the intensity peaks. No particles closer than twice this value will be found. If two peaks are found within twice this value,
            the one with highest intensity moment will be kept.
        """
        return self._integration_radius_of_intensity_peaks

    @integration_radius_of_intensity_peaks.setter
    def integration_radius_of_intensity_peaks(self, width):
        if not width == self._integration_radius_of_intensity_peaks:
            self._integration_radius_of_intensity_peaks = width
            self._update_particle_positions()
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def feature_point_threshold(self):
        """
        float:
            Defines the threshold value for finding intensity peaks. Local maximas below this threshold will not be
            considered as particles. Should be a value between 0 and 1.
        """
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
        """
        float:
            TODO
        """
        return self._particle_discrimination_threshold

    @particle_discrimination_threshold.setter
    def particle_discrimination_threshold(self, threshold):
        if not threshold == self._particle_discrimination_threshold:
            self._particle_discrimination_threshold = threshold
            self._update_particle_positions()

    @property
    def maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles(self):
        """
        int:
            Number of frames a particle can be invisible and still be linked in a trajectory.
        """
        return self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles

    @maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles.setter
    def maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles(self,
                                                                                                 number_of_frames):
        if not number_of_frames == self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:
            self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = number_of_frames
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def maximum_distance_a_particle_can_travel_between_frames(self):
        """
        int:
            Max number of pixels a particle can travel between two consecutive frames.
        """
        return self._maximum_distance_a_particle_can_travel_between_frames

    @maximum_distance_a_particle_can_travel_between_frames.setter
    def maximum_distance_a_particle_can_travel_between_frames(self, distance):
        if not distance == self._maximum_distance_a_particle_can_travel_between_frames:
            self._maximum_distance_a_particle_can_travel_between_frames = distance
            self._update_association_matrix()
            self._update_trajectories()

    @property
    def trajectories(self):
        """
        list:
            Returns a list with all found trajectories of type class: Trajectory.
        """
        return self._trajectories

    @property
    def particle_positions(self):
        """
        np.array:
            Numpy array with all particle positions on the form `np.array((nParticles,), dtype=[('frame_index', np.int16),
            ('time', np.float32),('integer_position', np.int16), ('refined_position', np.float32)])`
        """
        return self._particle_positions

    @property
    def time(self):
        """
        np.array:
            The time for each frame.
        """
        return self._time

    def get_frame_at_time(self, time):
        """
        time: float
            Time of the frame which you want to get.

        Returns
        -------
            np.array
                Returns the frame which corresponds to the input time.
        """
        index = self._find_index_of_nearest(self.time, time)
        return self._averaged_intensity[index]

    def plot_all_frames(self, ax=None, **kwargs):
        """
        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.imshow method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        ax.imshow(self._averaged_intensity, **kwargs)
        return ax

    def plot_frame_at_time(self, time, ax=None, **kwargs):
        """
        time: float
            The time of the frame you want to plot.
        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        intensity = self.get_frame_at_time(time)
        if ax is None:
            ax = plt.axes()
        ax.plot(intensity, **kwargs)
        return ax

    def plot_frame(self, frame_nr, ax=None, **kwargs):
        """
        time: float
            The time of the frame you want to plot.
        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        intensity = self._averaged_intensity[frame_nr]
        if ax is None:
            ax = plt.axes()
        ax.plot(intensity, **kwargs)
        return ax

    def _update_trajectories(self):
        self._trajectories = []
        count = 0
        for index, position in enumerate(self._particle_positions):
            if not self._is_particle_position_already_used_in_trajectory(position):
                self._trajectories.append(Trajectory())
                self._trajectories[count]._append_position(position)
                for index_future_points, future_point in enumerate(self._particle_positions[index + 1:]):
                    if self._points_are_linked(position, future_point):
                        self._trajectories[count]._append_position(future_point)
                        position = future_point
                count += 1

    def _update_association_matrix(self):
        self._initialise_empty_association_matrix()
        self._initialise_empty_cost_matrix()
        self._calculate_cost_matrix()
        self._create_initial_links_in_association_matrix()
        self._optimise_association_matrix()

    def _update_averaged_intensity(self):
        if self.boxcar_width == 0:
            self._averaged_intensity = self._frames
        else:
            self._averaged_intensity = np.empty(self._frames.shape)
            kernel = Box1DKernel(self.boxcar_width)
            for row_index, row_intensity in enumerate(self._frames):
                self._averaged_intensity[row_index] = convolve(row_intensity, kernel)

    def _update_particle_positions(self):
        self._find_integer_particle_positions()
        self._refine_particle_positions()
        self._perform_particle_discrimination()

    def _find_integer_particle_positions(self):
        frame_indexes = []
        times = []
        integer_positions = []
        for row_index, row_intensity in enumerate(self._averaged_intensity):
            indexes_of_local_maximas = self._find_indexes_of_local_maximas_with_intensity_higher_than_threshold(
                row_intensity)
            integer_positions += indexes_of_local_maximas
            frame_indexes += [row_index for index in indexes_of_local_maximas]
            times += [self._time[row_index] for index in indexes_of_local_maximas]
        self._particle_positions = np.empty((len(frame_indexes),),
                                            dtype=[('frame_index', np.int16), ('time', np.float32),
                                                   ('integer_position', np.int16), ('refined_position', np.float32)])
        self._particle_positions['frame_index'] = frame_indexes
        self._particle_positions['time'] = times
        self._particle_positions['integer_position'] = integer_positions

    def _find_indexes_of_local_maximas_with_intensity_higher_than_threshold(self, array):
        columns_with_local_maximas = np.r_[array[:-1] > array[1:], True] & \
                                     np.r_[True, array[1:] > array[:-1]] & \
                                     np.r_[array > self.feature_point_threshold]
        return np.argwhere(columns_with_local_maximas).flatten().tolist()

    def _refine_particle_positions(self):
        if self.integration_radius_of_intensity_peaks != 0:
            for row_index, position in enumerate(self._particle_positions):
                refined_position = self._find_center_of_mass_close_to_position(position)
                self._particle_positions['refined_position'][row_index] = refined_position
                self._particle_positions['integer_position'][row_index] = round(refined_position)
        else:
            self._particle_positions['refined_position'] = self._particle_positions['integer_position']

    def _find_center_of_mass_close_to_position(self, particle_position):
        if particle_position['integer_position'] == 0:
            return 0
        if particle_position['integer_position'] <= self.integration_radius_of_intensity_peaks:
            width = particle_position['integer_position']
        elif particle_position['integer_position'] >= self._averaged_intensity.shape[
            1] - self.integration_radius_of_intensity_peaks:
            width = self._averaged_intensity.shape[1] - particle_position['integer_position']
        else:
            width = self._integration_radius_of_intensity_peaks
        intensity = self._averaged_intensity[particle_position[0],
                    particle_position['integer_position'] - width:particle_position['integer_position'] + width]
        return particle_position['integer_position'] + self._calculate_center_of_mass(intensity-np.min(intensity)) - width

    def _perform_particle_discrimination(self):
        self._remove_particles_with_wrong_intensity_moment()
        self._remove_particles_too_closely_together()

    def _remove_particles_with_wrong_intensity_moment(self):
        if self.particle_discrimination_threshold != 0:
            index_of_particles_to_be_kept = []
            for row_index, position in enumerate(self._particle_positions):
                if self._calculate_discrimination_score_for_particle(
                        position) >= self.particle_discrimination_threshold:
                    index_of_particles_to_be_kept.append(row_index)
            self._particle_positions = self._particle_positions[index_of_particles_to_be_kept]

    def _calculate_discrimination_score_for_particle(self, particle_position):
        score = 0
        particle_positions_of_particles_in_same_frame = self._get_particle_positions_in_frame(
            frame_index=particle_position['frame_index'])
        particle_positions_of_particles_in_same_frame = particle_positions_of_particles_in_same_frame[
            np.where(particle_positions_of_particles_in_same_frame['integer_position'] != particle_position[
                'integer_position'])]
        for index, position in enumerate(particle_positions_of_particles_in_same_frame):
            score += self._calculate_gaussian_moment(particle_position, position)
        return score

    def _calculate_gaussian_moment(self, particle_position_1, particle_position_2):
        return 1 / (2 * np.pi * self._sigma_0 * self._sigma_2 * self._particle_positions.shape[0]) * \
               np.exp(-(self._calculate_first_order_intensity_moment(
                   particle_position_1) - self._calculate_first_order_intensity_moment(particle_position_2)) ** 2 / (
                              2 * self._sigma_0)
                      - (self._calculate_second_order_intensity_moment(
                   particle_position_1) - self._calculate_second_order_intensity_moment(particle_position_2)) ** 2 / (
                              2 * self._sigma_2))

    def _calculate_second_order_intensity_moment(self, particle_position):
        if particle_position['integer_position'] == 0:
            return 0
        if particle_position['integer_position'] < self.integration_radius_of_intensity_peaks:
            w = particle_position['integer_position']
        elif particle_position['integer_position'] > self._frames.shape[1] - self.integration_radius_of_intensity_peaks:
            w = self._frames.shape[1] - particle_position['integer_position']
        else:
            w = self.integration_radius_of_intensity_peaks
        return np.sum(
            np.arange(-w, w) ** 2 * self.frames[particle_position['frame_index'],
                                    particle_position['integer_position'] - w: particle_position[
                                                                                   'integer_position'] + w]) / \
               self._calculate_first_order_intensity_moment(particle_position)

    def _calculate_first_order_intensity_moment(self, particle_position):
        if particle_position['integer_position'] == 0:
            return self._averaged_intensity[particle_position['frame_index'], particle_position['integer_position']]
        if particle_position['integer_position'] < self.integration_radius_of_intensity_peaks:
            w = particle_position['integer_position']
        elif particle_position['integer_position'] > self._frames.shape[1] - self.integration_radius_of_intensity_peaks:
            w = self._frames.shape[1] - particle_position['integer_position']
        else:
            w = self.integration_radius_of_intensity_peaks
        return np.sum(self.frames[particle_position['frame_index'],
                      particle_position['integer_position'] - w: particle_position['integer_position'] + w])

    def _get_particle_positions_in_frame(self, frame_index):
        return self._particle_positions[np.where(self._particle_positions['frame_index'] == frame_index)]

    def _remove_particles_too_closely_together(self):
        for index, first_position in enumerate(self._particle_positions[:-1]):
            second_position = self._particle_positions[index + 1]
            if self._particles_are_too_close(first_position, second_position):
                first_order_moment_for_first_position = self._calculate_first_order_intensity_moment(first_position)
                first_order_moment_for_second_position = self._calculate_first_order_intensity_moment(second_position)
                if first_order_moment_for_first_position < first_order_moment_for_second_position:
                    self._particle_positions = np.delete(self._particle_positions, index, axis=0)
                    return self._remove_particles_too_closely_together()
                else:
                    self._particle_positions = np.delete(self._particle_positions, index + 1, axis=0)
                    return self._remove_particles_too_closely_together()

    def _particles_are_too_close(self, position1, position2):
        return position1['frame_index'] == position2['frame_index'] and (
                np.abs(position2['integer_position'] - position1[
                    'integer_position']) < self.integration_radius_of_intensity_peaks)

    def _initialise_empty_association_matrix(self):
        self._association_matrix = {}
        for index, t in enumerate(self._time):
            number_of_particles_at_t = np.count_nonzero(self._particle_positions['frame_index'] == index)
            self._association_matrix[str(index)] = {}
            for r in range(1,
                           self.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles + 1):
                if r + index < len(self._time):
                    number_of_particles_at_t_plus_r = np.count_nonzero(
                        self._particle_positions['frame_index'] == index + r)
                    self._association_matrix[str(index)][str(r)] = np.zeros(
                        (number_of_particles_at_t + 1, number_of_particles_at_t_plus_r + 1), dtype=np.int16)

    def _create_initial_links_in_association_matrix(self):
        for frame_index, frame_key in enumerate(self._association_matrix.keys()):
            for future_frame_index, future_frame_key in enumerate(self._association_matrix[frame_key].keys()):
                self._association_matrix[frame_key][future_frame_key] = self._initialise_link_matrix(
                    self._association_matrix[frame_key][future_frame_key], frame_key,
                    future_frame_key)

    def _initialise_empty_cost_matrix(self):
        self._cost_matrix = {}
        for frame_index, frame_key in enumerate(self._association_matrix.keys()):
            self._cost_matrix[frame_key] = {}
            for future_frame_index, future_frame_key in enumerate(self._association_matrix[frame_key].keys()):
                self._cost_matrix[frame_key][future_frame_key] = np.zeros(
                    self._association_matrix[frame_key][future_frame_key].shape, dtype=np.float32)

    def _calculate_cost_matrix(self):
        for frame_index, frame_key in enumerate(self._cost_matrix.keys()):
            for future_frame_index, future_frame_key in enumerate(self._cost_matrix[frame_key].keys()):
                cost_for_association_with_dummy_particle = self._calculate_cost_for_association_with_dummy_particle(
                    future_frame_index + 1)
                particle_positions_in_current_frame = self._get_particle_positions_in_frame(frame_index)
                particle_positions_in_future_frame = self._get_particle_positions_in_frame(
                    frame_index + future_frame_index + 1)
                for row_index, row in enumerate(self._cost_matrix[frame_key][future_frame_key]):
                    for col_index, value in enumerate(self._cost_matrix[frame_key][future_frame_key][row_index]):
                        if row_index == 0 or col_index == 0:
                            self._cost_matrix[frame_key][future_frame_key][row_index][
                                col_index] = cost_for_association_with_dummy_particle
                        else:
                            position1 = particle_positions_in_current_frame[row_index - 1]
                            position2 = particle_positions_in_future_frame[col_index - 1]
                            self._cost_matrix[frame_key][future_frame_key][row_index][
                                col_index] = self._calculate_linking_cost(position1, position2)

    def _initialise_link_matrix(self, link_matrix, frame_key, future_frame_key):
        for row_index, costs in enumerate(self._cost_matrix[frame_key][future_frame_key]):
            if not row_index == 0:
                col_index = np.where(costs == np.amin(costs))[0][0]
                if (not (link_matrix[:, col_index] == 1).any()) or col_index == 0:
                    link_matrix[row_index][col_index] = 1
        return self._fill_in_empty_rows_and_columns(link_matrix)

    def _calculate_linking_cost(self, position1, position2):
        return (
                (position1['refined_position'] - position2['refined_position']) ** 2 +
                (self._calculate_first_order_intensity_moment(position1) - self._calculate_first_order_intensity_moment(
                    position2)) ** 2 +
                (self._calculate_second_order_intensity_moment(
                    position1) - self._calculate_second_order_intensity_moment(position2)) ** 2
        )

    def _calculate_cost_for_association_with_dummy_particle(self, future_frame_index):
        return (self.maximum_distance_a_particle_can_travel_between_frames * future_frame_index) ** 2

    def _optimise_association_matrix(self):
        for frame_index, frame_key in enumerate(self._association_matrix.keys()):
            for future_frame_index, future_frame_key in enumerate(self._association_matrix[frame_key].keys()):
                link_matrix = self._association_matrix[frame_key][future_frame_key]
                self._association_matrix[frame_key][future_frame_key] = self._optimise_link_matrix(link_matrix,
                                                                                                   frame_key,
                                                                                                   future_frame_key)
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
                            reduction_cost_row = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][
                                col_index]
                            reduction_cost_col = self._cost_matrix[frame_key][future_frame_key][row_index][
                                col_index_with_link]
                            introduction_row_col = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][
                                col_index_with_link]
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
                            reduction_cost_row = self._cost_matrix[frame_key][future_frame_key][row_index_with_link][
                                col_index]
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
                            reduction_cost_col = self._cost_matrix[frame_key][future_frame_key][row_index][
                                col_index_with_link]
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
            if trajectory._position_exists_in_trajectory(particle_position):
                return True
        return False

    def _points_are_linked(self, point, future_point):
        if point['frame_index'] == future_point['frame_index']:
            return False
        nr_of_frames_between_points = self._calculate_number_of_frames_between_particle_positions(point, future_point)
        if nr_of_frames_between_points <= self.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:

            time_key = str(point['frame_index'])
            r_key = str(nr_of_frames_between_points)

            link_matrix = self._association_matrix[time_key][r_key]

            points_in_same_frame_as_point = self._get_particle_positions_in_frame(point['frame_index'])
            points_in_same_frame_as_future_point = self._get_particle_positions_in_frame(future_point['frame_index'])

            index_of_point = \
                np.where(points_in_same_frame_as_point['integer_position'] == point['integer_position'])[0][0]
            index_of_future_point = \
                np.where(points_in_same_frame_as_future_point['integer_position'] == future_point['integer_position'])[0][0]
            return int(link_matrix[index_of_point + 1][index_of_future_point + 1]) == 1
        else:
            return False

    @staticmethod
    def _calculate_center_of_mass(y):
        x = np.arange(0, y.shape[0])
        return np.sum(x * y) / np.sum(y)

    @staticmethod
    def _calculate_number_of_frames_between_particle_positions(p1, p2):
        return int(p2['frame_index'] - p1['frame_index'])

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
    def normalise_intensity(frames):
        """
        frames: np.array
            Normalises the intensity of the frames according to :math:`I_n = (I-I_{min})/(I_{max}-I_{min})`, where :math:`I` is
            the intensity of the frames, :math:`I_{min}`, :math:`I_{max}` are the global intensity minima and maxima of
            the frames.

        Returns
        -------
            np.array
                The normalised intensity.
        """
        frames = frames - np.amin(frames)
        return frames / np.amax(frames)

    @staticmethod
    def _find_index_of_nearest(array, value):
        return (np.abs(np.array(array) - value)).argmin()
