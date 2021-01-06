import numpy as np
import matplotlib.pyplot as plt
from particle_tracker_one_d.trajectory.trajectory import Trajectory
from ..frames.frames import Frames
from scipy.signal import find_peaks


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
    particle_detection_threshold
    maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles
    maximum_distance_a_particle_can_travel_between_frames
    particle_positions
    """

    def __init__(self, frames, time, automatic_update=True):
        ParticleTracker._validate_class_arguments(frames, time, automatic_update)
        self._Frames = Frames(frames=frames, time=time, automatic_update=automatic_update)
        self._automatic_update = automatic_update
        self._integration_radius_of_intensity_peaks = 1
        self._particle_detection_threshold = 1
        self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 1
        self._maximum_distance_a_particle_can_travel_between_frames = 1
        self._particle_positions = [np.array([])] * self.frames.shape[0]
        self._trajectories = []
        self._cost_matrix = []
        self._association_matrix = []
        self._trajectory_links = []
        self._cost_coefficients = np.array([1, 1, 1, 0], dtype=np.float32)

    @property
    def frames(self):
        """
        np.array:
            The frames which the particle tracker tries to find trajectories in. If the property boxcar_width!=0 it will return the smoothed frames.
        """
        return self._Frames.frames

    @property
    def boxcar_width(self):
        """
        int:
            Number of values used in the boxcar averaging of the frames.
        """
        return self._Frames.boxcar_width

    @boxcar_width.setter
    def boxcar_width(self, width):
        if not width == self._Frames.boxcar_width:
            self._Frames.boxcar_width = width
            if self._automatic_update:
                self._find_particle_positions()
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
    def integration_radius_of_intensity_peaks(self, radius):
        if type(radius) is not int:
            raise TypeError('Attribute integration_radius_of_intensity_peaks should be of type int')
        if not -1 < radius <= self.frames.shape[1] / 2:
            raise ValueError(
                'Attribute integration_radius_of_intensity_peaks should be a positive integer less or equal the half of the number of pixels in each frame.')

        if not radius == self._integration_radius_of_intensity_peaks:
            self._integration_radius_of_intensity_peaks = radius
            if self._automatic_update:
                self._find_particle_positions()
                self._update_association_matrix()
                self._update_trajectories()

    @property
    def particle_detection_threshold(self):
        """
        float:
            Defines the threshold value for finding intensity peaks. Local maximas below this threshold will not be
            considered as particles. Should be a value between 0 and 1.
        """
        return self._particle_detection_threshold

    @particle_detection_threshold.setter
    def particle_detection_threshold(self, threshold):
        if not (type(threshold) == int or type(threshold) == float):
            raise TypeError('Attribute particle_detection_threshold should be a numerical value between 0 and 1.')
        if not 0 <= threshold <= 1:
            raise ValueError('Attribute particle_detection_threshold should be a value between 0 and 1.')
        if not threshold == self._particle_detection_threshold:
            self._particle_detection_threshold = threshold
            if self._automatic_update:
                self._find_particle_positions()
                self._update_association_matrix()
                self._update_trajectories()

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
        if type(number_of_frames) is not int:
            raise TypeError(
                'Attribute maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles should be an integer.')
        if not 0 <= number_of_frames < self.frames.shape[0]:
            raise ValueError(
                'Attribute maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles should be larger or equal to 0 and smaller than the number of frames.')
        if not number_of_frames == self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:
            self._maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = number_of_frames
            if self._automatic_update:
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
        if not (type(distance) == int or type(distance) == float):
            raise TypeError(
                'Attribute maximum_distance_a_particle_can_travel_between_frames should be a numerical value.')
        if not 0 < distance < self.frames.shape[1]:
            raise ValueError(
                'Attribute maximum_distance_a_particle_can_travel_between_frames should be larger than 0 and smaller than the number of pixels in each frames.')
        if not distance == self._maximum_distance_a_particle_can_travel_between_frames:
            self._maximum_distance_a_particle_can_travel_between_frames = distance
            if self._automatic_update:
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
        list:
            List with numpy arrays containing all particle positions.
        """
        return self._particle_positions

    @property
    def time(self):
        """
        np.array:
            The time for each frame.
        """
        return self._Frames.time

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
        return self.frames[index]

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
        ax.imshow(self.frames, **kwargs)
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

    def plot_frame(self, frame_index, ax=None, **kwargs):
        """
        frame_index: index
            The index of the frame you want to plot.
        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        intensity = self.frames[frame_index]
        if ax is None:
            ax = plt.axes()
        ax.plot(intensity, **kwargs)
        return ax

    def plot_all_particles(self, ax=None, **kwargs):
        """
        ax: matplotlib axes instance
            The axes which you want the particle detections to be plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.scatter method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        for frame_index, positions in enumerate(self._particle_positions):
            for position in positions:
                ax.plot(position, frame_index, **kwargs)
        return ax

    def plot_moments(self, ax=None, **kwargs):
        """
        ax: matplotlib axes instance
            The axes which you want the moments to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.scatter method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        zeroth_order_moments = [m for moments in self._zeroth_order_moments for m in moments]
        second_order_moments = [m for moments in self._second_order_moments for m in moments]
        ax.scatter(zeroth_order_moments, second_order_moments, **kwargs)
        ax.set_xlabel('$m_0$')
        ax.set_ylabel('$m_2$')
        return ax

    def change_cost_coefficients(self, a=1, b=1, c=1, d=1):
        """
        Change the coefficients of the cost function :math:`c(p_1,p_2) = a\\cdot (x_{p_1} - x_{p_2})^2 + b \\cdot (m_0(p_1)-m_0(p_2))^2 + b \\cdot (m_2(p_1)-m_2(p_2))^2) + d \\cdot (t_{p_1}-t_{p_2})^2`

        a: float

        b: float

        c: float

        d: float
        """
        new_cost_coefficients = np.array([a, b, c, d], dtype=np.float32)
        if np.array_equal(new_cost_coefficients, np.array([0, 0, 0, 0])):
            raise ValueError('All cost coefficients can\'t be zero')
        if not np.array_equal(new_cost_coefficients, self._cost_coefficients):
            self._cost_coefficients = new_cost_coefficients
            if self._automatic_update:
                self._update_association_matrix()
                self._update_trajectories()

    def _build_trajectory(self, trajectory):
        last_position = trajectory[-1]
        if last_position[0] == len(self._association_matrix) - 1:
            return trajectory
        link_matrix = self._association_matrix[last_position[0]][0]
        link_index = link_matrix[last_position[1] + 1, :].nonzero()[0][0]
        if link_index > 0:
            trajectory.append((last_position[0] + 1, link_index - 1))
            return self._build_trajectory(trajectory)
        else:
            return trajectory

    def _find_initial_trajectories(self):
        trajectories = []
        particle_has_been_used = [np.zeros(positions.shape, dtype=bool) for positions in self._particle_positions]
        for frame_index, positions in enumerate(self._particle_positions):
            for position_index, position in enumerate(positions):
                if not particle_has_been_used[frame_index][position_index]:
                    trajectory = self._build_trajectory([(frame_index, position_index)])
                    for p in trajectory:
                        particle_has_been_used[p[0]][p[1]] = True
                    trajectories.append(trajectory)
        return trajectories

    def _find_all_associations(self, position):
        associations = []
        for index, link_matrix in enumerate(self._association_matrix[position[0]]):
            link_index = link_matrix[position[1] + 1, :].nonzero()[0][0]
            cost = self._cost_matrix[position[0]][index][position[1] + 1, link_index]
            if link_index > 0:
                associations.append((position[0] + index + 1, link_index - 1, cost))
        return associations

    def __connect_broken_trajectories(self, trajectories):
        for index, trajectory in enumerate(trajectories):
            if trajectory[-1][0] < self.frames.shape[0] - 1:
                possible_associations = self._find_all_associations(trajectory[-1])
                possible_associations.sort(key=lambda x: x[2])
                if len(possible_associations) > 0:
                    trajectories[index].append(possible_associations)

        return trajectories

    def _initialise_trajectory_links(self):
        for t, _ in enumerate(self._association_matrix[:-1]):
            for i, _ in enumerate(self._association_matrix[t][0]):
                for j, is_associated in enumerate(self._association_matrix[t][0][i]):
                    if is_associated and (i > 0):
                        if t == 0:
                            self._trajectory_links[t][0][i, j] = 1
                        else:
                            self._trajectory_links[t][0][i, j] = 1 + np.sum(self._trajectory_links[t - 1][0][:, i])

    def _is_connected_to_dummy_particle(self, frame_index, particle_index):
        return self._association_matrix[frame_index][0][particle_index + 1][0]

    def _find_cheapest_valid_link(self, frame_index, particle_index):
        t = frame_index
        i = particle_index + 1
        costs = []
        for r, link_matrix in enumerate(self._association_matrix[t]):
            j = link_matrix[i].nonzero()[0][0]
            costs.append((r, j, self._cost_matrix[t][r][i][j]))
        costs.sort(key=lambda x: x[2])
        for c in costs:
            if (
                    t + c[0] + 1 < len(self._trajectory_links) - 1 and
                    (np.sum(self._trajectory_links[t + c[0] + 1][0][c[1]]) == 1) and
                    c[2] < np.inf
            ):
                return t, c[0], i, c[1]
            elif c[2] < np.inf and (t + c[0] + 1 == len(self._trajectory_links) - 1):
                is_already_linked = False
                for link_matrices in self._trajectory_links[t:-1]:
                    if link_matrices[-1][:, c[1]].any():
                        is_already_linked = True
                if not is_already_linked:
                    return t, c[0], i, c[1]

        return t, 0, i, 0

    def _connect_broken_trajectories(self):
        for frame_index, positions in enumerate(self._particle_positions[:-1]):
            for particle_index, _ in enumerate(positions):
                if self._is_connected_to_dummy_particle(frame_index, particle_index):
                    t, r, i, j = self._find_cheapest_valid_link(frame_index, particle_index)
                    if j > 0:
                        self._trajectory_links[t][r][i][j] = 1
                        if t + r + 1 < len(self._trajectory_links) - 1:
                            j_prim = self._trajectory_links[t + r + 1][0][j].nonzero()[0][0]
                            self._trajectory_links[t + r + 1][0][j][j_prim] += 1

    def _update_trajectories(self):
        self._initialise_trajectory_links()
        self._connect_broken_trajectories()
        self._build_trajectories()

    def _build_trajectories(self):
        self._trajectories = []
        for t, link_matrices in enumerate(self._trajectory_links[:-1]):
            for i, links in enumerate(link_matrices[0]):
                for j, link in enumerate(links):
                    if link == 1:
                        self._trajectories.append(
                            self._build_trajectory_from_particle(t, i)
                        )

    def _create_trajectory(self, indices):
        last_position = indices[-1]
        if last_position[0] == len(self._trajectory_links) - 1:
            return indices
        for r, link_matrix in enumerate(self._trajectory_links[last_position[0]]):
            j = link_matrix[last_position[1]].nonzero()[0]
            if j.size > 0 and (j[0] > 0):
                indices.append((last_position[0] + r + 1, j[0]))
                return self._create_trajectory(indices)
        return indices

    def _build_trajectory_from_particle(self, t, i):
        index_trajectory = self._create_trajectory([(t, i)])
        trajectory = Trajectory()
        particle_positions = np.empty((len(index_trajectory),),
                                      dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32),
                                             ('zeroth_order_moment', np.float32),
                                             ('second_order_moment', np.float32)])
        for index, indices in enumerate(index_trajectory):
            particle_positions[index]['frame_index'] = indices[0]
            particle_positions[index]['time'] = self.time[indices[0]]
            particle_positions[index]['position'] = self._particle_positions[indices[0]][indices[1] - 1]
            particle_positions[index]['zeroth_order_moment'] = self._zeroth_order_moments[indices[0]][indices[1] - 1]
            particle_positions[index]['second_order_moment'] = self._second_order_moments[indices[0]][indices[1] - 1]

        trajectory.particle_positions = particle_positions
        return trajectory

    def _create_trajectory_from_particle(self, indexes):
        for future_frame_index, link_matrix in enumerate(self._association_matrix[indexes[-1][0]]):
            future_particle_index = np.where(link_matrix[indexes[-1][1] + 1])[0][0]
            if future_particle_index != 0:
                indexes.append(
                    np.array([future_frame_index + indexes[-1][0] + 1, future_particle_index - 1], dtype=np.int16))
                return self._create_trajectory_from_particle(indexes)
        return indexes

    def _update_association_matrix(self):
        self._calculate_particle_moments()
        self._initialise_association_and_cost_matrices()
        self._calculate_cost_matrices()
        self._create_initial_links_in_association_matrix()
        self._optimise_association_matrix()

    def _calculate_particle_moments(self):
        self._zeroth_order_moments = [None] * self.frames.shape[0]
        self._second_order_moments = [None] * self.frames.shape[0]
        for frame_index, positions in enumerate(self._particle_positions):
            self._zeroth_order_moments[frame_index] = np.array(
                [self._calculate_zeroth_order_intensity_moment(position, frame_index) for position in positions],
                dtype=np.float64)
            self._second_order_moments[frame_index] = np.array(
                [self._calculate_second_order_intensity_moment(position, frame_index) for position in positions],
                dtype=np.float64)

    def _find_particle_positions(self):
        self._find_initial_particle_positions()
        self._refine_particle_positions()

    def _find_initial_particle_positions(self):
        self._particle_positions = [None] * self.frames.shape[0]
        for index, frame in enumerate(self.frames):
            self._particle_positions[index] = self._find_local_maximas_larger_than_threshold(frame)

    def _refine_particle_positions(self):
        if self._integration_radius_of_intensity_peaks == 0:
            return
        for frame_index, positions in enumerate(self._particle_positions):
            for index, position in enumerate(positions):
                if position == 0 or position + 1 == self._Frames.frames.shape[1]:
                    continue
                elif position < self.integration_radius_of_intensity_peaks:
                    integration_radius = position
                elif position > self._Frames.frames.shape[1] - self._integration_radius_of_intensity_peaks - 1:
                    integration_radius = self._Frames.frames.shape[1] - position
                else:
                    integration_radius = self.integration_radius_of_intensity_peaks
                intensity = self.frames[frame_index][
                            int(position - integration_radius):int(position + integration_radius + 1)]
                intensity = intensity - np.min(intensity)
                self._particle_positions[frame_index][index] = position + self._calculate_center_of_mass(
                    intensity) - integration_radius

    def _calculate_second_order_intensity_moment(self, position, frame_index):
        position = int(round(position))
        if self._integration_radius_of_intensity_peaks == 0:
            return 0
        elif position == 0:
            if self._integration_radius_of_intensity_peaks == 1:
                return 2 * self.frames[frame_index, 1] / self._calculate_zeroth_order_intensity_moment(position,
                                                                                                       frame_index)
            else:
                second_order_index_array = np.arange(0, self._integration_radius_of_intensity_peaks + 1) ** 2
                return (
                               2 * np.dot(self.frames[frame_index, :self.integration_radius_of_intensity_peaks + 1],
                                          second_order_index_array)
                       ) / self._calculate_zeroth_order_intensity_moment(position, frame_index)
        elif position == self._Frames.frames.shape[1] - 1:
            if self._integration_radius_of_intensity_peaks == 1:
                return 2 * self.frames[frame_index, -2] / self._calculate_zeroth_order_intensity_moment(position,
                                                                                                        frame_index)
            else:
                second_order_index_array = np.arange(-self._integration_radius_of_intensity_peaks, 0) ** 2
                return 2 * np.dot(self.frames[frame_index, -self._integration_radius_of_intensity_peaks - 1:-1],
                                  second_order_index_array) / self._calculate_zeroth_order_intensity_moment(position,
                                                                                                            frame_index)
        elif position < self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - position
            if w == 1:
                second_order_index_array = np.arange(-position, position + 1) ** 2
                return (
                        (
                                np.dot(self.frames[frame_index, :2 * position + 1], second_order_index_array) +
                                2 * self._integration_radius_of_intensity_peaks ** 2 * self.frames[
                                    frame_index, position + self._integration_radius_of_intensity_peaks]
                        ) / self._calculate_zeroth_order_intensity_moment(position, frame_index)
                )
            else:
                second_order_index_array = np.arange(-position, position + 1) ** 2
                second_order_index_array_big = np.arange(position + 1,
                                                         position + self._integration_radius_of_intensity_peaks) ** 2
                return (
                               np.dot(second_order_index_array, self.frames[frame_index, :2 * position + 1]) +
                               2 * np.dot(second_order_index_array_big,
                                          self.frames[frame_index,
                                          2 * position + 1:2 * position + self._integration_radius_of_intensity_peaks])
                       ) / self._calculate_zeroth_order_intensity_moment(position, frame_index)
        elif position > self._Frames.frames.shape[1] - 1 - self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - (self._Frames.frames.shape[1] - position - 1)
            if w == 1:
                second_order_index_array = np.arange(-(self._Frames.frames.shape[1] - 1 - position),
                                                     self._Frames.frames.shape[1] - position) ** 2
                return (
                               np.dot(second_order_index_array,
                                      self.frames[frame_index, 2 * position - self._Frames.frames.shape[1] + 1:])
                               + 2 * self._integration_radius_of_intensity_peaks ** 2 * self.frames[
                                   frame_index, position - self._integration_radius_of_intensity_peaks]
                       ) / self._calculate_zeroth_order_intensity_moment(position, frame_index)
            else:
                second_order_index_array = np.arange(-(self._Frames.frames.shape[1] - 1 - position),
                                                     self._Frames.frames.shape[1] - position) ** 2
                second_order_index_array_big = np.arange(- self._integration_radius_of_intensity_peaks,
                                                         -(self._Frames.frames.shape[1] - position) + 1) ** 2
                return (
                               np.dot(second_order_index_array,
                                      self.frames[frame_index, -2 * (self._Frames.frames.shape[1] - position) + 1:]) +
                               2 * np.dot(second_order_index_array_big,
                                          self.frames[frame_index,
                                          position - self._integration_radius_of_intensity_peaks:-2 * (
                                                  self._Frames.frames.shape[1] - position) + 1])
                       ) / self._calculate_zeroth_order_intensity_moment(position, frame_index)

        else:
            w = self._integration_radius_of_intensity_peaks
            second_order_index_array = np.arange(-w, w + 1) ** 2
            return np.dot(self.frames[frame_index, position - w:position + w + 1],
                          second_order_index_array) / self._calculate_zeroth_order_intensity_moment(position,
                                                                                                    frame_index)

    def _calculate_zeroth_order_intensity_moment(self, position, frame_index):
        position = int(round(position))
        if self._integration_radius_of_intensity_peaks == 0:
            return self.frames[frame_index, position]
        elif position == 0:
            if self._integration_radius_of_intensity_peaks == 1:
                return self.frames[frame_index, position] + 2 * self.frames[frame_index, position + 1]
            else:
                return (
                        self.frames[frame_index, position] +
                        2 * np.sum(
                    self.frames[frame_index, position + 1:position + 1 + self._integration_radius_of_intensity_peaks])
                )
        elif position == self._Frames.frames.shape[1] - 1:
            if self._integration_radius_of_intensity_peaks == 1:
                return self.frames[frame_index, position] + 2 * self.frames[frame_index, position - 1]
            else:
                return (
                        self.frames[frame_index, position] +
                        2 * np.sum(
                    self.frames[frame_index, position - self._integration_radius_of_intensity_peaks:position])
                )
        elif position < self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - position
            if w == 1:
                return np.sum(self.frames[frame_index, :2 * position + 1]) + 2 * self.frames[
                    frame_index, position + w + 1]
            else:
                return (
                        np.sum(self.frames[frame_index, :2 * position + 1]) +
                        2 * np.sum(self.frames[frame_index, 2 * position + 1:2 * position + w + 1])
                )
        elif position > self._Frames.frames.shape[1] - 1 - self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - (self._Frames.frames.shape[1] - position - 1)
            if w == 1:
                return (
                        np.sum(self.frames[frame_index, -2 * (self._Frames.frames.shape[1] - position) + 1:]) +
                        2 * self.frames[frame_index, -2 * (self._Frames.frames.shape[1] - position)]
                )
            else:
                return (
                        np.sum(self.frames[frame_index, -2 * (self._Frames.frames.shape[1] - position) + 1:]) +
                        2 * np.sum(
                    self.frames[frame_index, position - self._integration_radius_of_intensity_peaks:position - 1])
                )
        else:
            w = self._integration_radius_of_intensity_peaks
            return np.sum(self.frames[frame_index, position - w:position + w + 1])

    def _initialise_association_and_cost_matrices(self):
        r = self.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles
        number_of_frames = self._Frames.frames.shape[0]

        self._association_matrix = [[] for _ in range(number_of_frames)]
        self._cost_matrix = [[] for _ in range(number_of_frames)]
        self._trajectory_links = [[] for _ in range(number_of_frames)]
        for frame_index in range(0, number_of_frames):
            for future_frame_index in range(frame_index + 1, frame_index + r + 2):
                if future_frame_index < number_of_frames:
                    self._association_matrix[frame_index].append(
                        np.zeros(
                            (len(self._particle_positions[frame_index]) + 1,
                             len(self._particle_positions[future_frame_index]) + 1), dtype=bool)
                    )
                    self._cost_matrix[frame_index].append(
                        np.zeros(
                            (len(self._particle_positions[frame_index]) + 1,
                             len(self._particle_positions[future_frame_index]) + 1), dtype=np.float32)
                    )
                    self._trajectory_links[frame_index].append(
                        np.zeros(
                            (len(self._particle_positions[frame_index]) + 1,
                             len(self._particle_positions[future_frame_index]) + 1), dtype=np.int32)
                    )

    def _create_initial_links_in_association_matrix(self):
        for t, _ in enumerate(self._cost_matrix):
            for r, _ in enumerate(self._cost_matrix[t]):
                used_js = []
                for i, _ in enumerate(self._cost_matrix[t][r]):
                    js = np.argsort(self._cost_matrix[t][r][i])
                    for j in js:
                        if j not in used_js and (self._cost_matrix[t][r][i][j] != np.inf):
                            used_js.append(j)
                            self._association_matrix[t][r][i][j] = 1
                            break
                for i, row in enumerate(self._association_matrix[t][r]):
                    if not row.any():
                        self._association_matrix[t][r][i][0] = True
                for i, row in enumerate(self._association_matrix[t][r].T):
                    if not row.any():
                        self._association_matrix[t][r][0, i] = True

    def _calculate_cost_matrices(self):
        for frame_index, _ in enumerate(self._cost_matrix):
            for future_frame_index, _ in enumerate(self._cost_matrix[frame_index]):
                for particle_index, _ in enumerate(self._cost_matrix[frame_index][future_frame_index]):
                    for future_particle_index, _ in enumerate(
                            self._cost_matrix[frame_index][future_frame_index][particle_index]):
                        if particle_index == 0 and future_particle_index == 0:
                            self._cost_matrix[frame_index][future_frame_index][particle_index][
                                future_particle_index] = 0
                        elif particle_index == 0 or future_particle_index == 0:
                            self._cost_matrix[frame_index][future_frame_index][particle_index][
                                future_particle_index] = self._calculate_cost_for_association_with_dummy_particle(
                                frame_index, frame_index + future_frame_index + 1)
                        else:
                            self._cost_matrix[frame_index][future_frame_index][particle_index][
                                future_particle_index] = self._calculate_linking_cost(
                                frame_index,
                                particle_index - 1,
                                frame_index + future_frame_index + 1,
                                future_particle_index - 1
                            )

    def _calculate_linking_cost(self, frame_index, particle_index, future_frame_index, future_particle_index):
        cost = (
                self._cost_coefficients[0] * (
                self.particle_positions[frame_index][particle_index] - self.particle_positions[future_frame_index][
            future_particle_index]) ** 2 +
                self._cost_coefficients[1] * (
                        self._zeroth_order_moments[frame_index][particle_index] -
                        self._zeroth_order_moments[future_frame_index][future_particle_index]) ** 2 +
                self._cost_coefficients[2] * (self._second_order_moments[frame_index][particle_index] -
                                              self._second_order_moments[future_frame_index][
                                                  future_particle_index]) ** 2
                + self._cost_coefficients[3] * (future_frame_index - frame_index)**2
        )
        if cost > self._calculate_cost_for_association_with_dummy_particle(frame_index, future_frame_index):
            return np.inf
        return cost

    def _calculate_cost_for_association_with_dummy_particle(self, frame_index, future_frame_index):
        return self._cost_coefficients[0] * (self.maximum_distance_a_particle_can_travel_between_frames * (future_frame_index - frame_index)) ** 2

    def _optimise_link_matrix(self, links, costs):
        lowest_cost = 1
        i_lowest = 0
        j_lowest = 0
        k_lowest = 0
        l_lowest = 0
        test_cost = 1
        for i in range(links.shape[0]):
            for j in range(links.shape[1]):
                if (links[i, j] == 0) and (costs[i, j] != np.inf):
                    k = links[:, j].nonzero()[0][0]
                    l = links[i, :].nonzero()[0][0]
                    if (
                            (i > 0) and
                            (j > 0) and
                            (costs[i, l] != np.inf) and
                            (costs[k, j] != np.inf) and
                            (costs[k, l] != np.inf)
                    ):
                        test_cost = costs[i, j] - costs[i, l] - costs[k, j] + costs[k, l]
                    elif (
                            (j > 0) and
                            (k > 0) and
                            (l == 0) and
                            (costs[k, j] != np.inf)
                    ):
                        test_cost = costs[0, j] - costs[k, j] + costs[k, 0]
                    elif (
                            (i > 0) and
                            (l > 0) and
                            (k == 0) and
                            (costs[i, l] != np.inf)
                    ):
                        test_cost = costs[i, 0] - costs[i, l] + costs[0, l]

                    if test_cost < lowest_cost:
                        lowest_cost = test_cost
                        i_lowest = i
                        j_lowest = j
                        k_lowest = k
                        l_lowest = l
        if lowest_cost < 0:
            links[i_lowest, j_lowest] = 1
            links[k_lowest, l_lowest] = 1
            if i_lowest > 0:
                links[i_lowest, l_lowest] = 0
            if j_lowest > 0:
                links[k_lowest, j_lowest] = 0
            return self._optimise_link_matrix(links, costs)
        return links

    def _optimise_association_matrix(self):
        for frame_index in range(len(self._association_matrix)):
            for future_frame_index in range(len(self._association_matrix[frame_index])):
                self._association_matrix[frame_index][future_frame_index] = self._optimise_link_matrix(
                    self._association_matrix[frame_index][future_frame_index],
                    self._cost_matrix[frame_index][future_frame_index]
                )

    def _find_local_maximas_larger_than_threshold(self, y):
        local_maximas, _ = find_peaks(y, height=self.particle_detection_threshold,
                                      distance=2 * self.integration_radius_of_intensity_peaks)
        return local_maximas.astype(np.float32)

    @staticmethod
    def _calculate_center_of_mass(y):
        x = np.arange(0, y.shape[0])
        return np.sum(x * y) / np.sum(y)

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

    @staticmethod
    def _validate_class_arguments(frames, time, automatic_update):
        ParticleTracker._test_if_frames_have_correct_format(frames)
        ParticleTracker._test_if_time_has_correct_format(time)
        ParticleTracker._test_if_time_and_frames_has_same_length(time, frames)
        ParticleTracker._test_if_automatic_update_has_correct_format(automatic_update)

    @staticmethod
    def _test_if_frames_have_correct_format(frames):
        if type(frames) is not np.ndarray:
            raise TypeError('Class argument frames not of type np.ndarray')
        if not (len(frames.shape) == 2 and frames.shape[0] > 1 and frames.shape[1] > 2):
            raise ValueError(
                'Class argument frames need to be of shape (nFrames,nPixels) with nFrames > 1 and nPixels >2')
        return True

    @staticmethod
    def _test_if_time_has_correct_format(time):
        if type(time) is not np.ndarray:
            raise TypeError('Class argument frames not of type np.ndarray')
        if not (len(time.shape) == 1 and time.shape[0] > 1):
            raise ValueError('Class argument time need to be of shape (nFrames,) with nFrames > 1.')
        if not all(np.diff(time) > 0):
            raise ValueError('Class argument time not increasing monotonically.')
        return True

    @staticmethod
    def _test_if_time_and_frames_has_same_length(time, frames):
        if not time.shape[0] == frames.shape[0]:
            raise ValueError('Class arguments time and frames does not of equal length.')
        return True

    @staticmethod
    def _test_if_automatic_update_has_correct_format(automatic_update):
        if not type(automatic_update) == bool:
            raise ValueError('Class argument automatic_update must be True or False.')
        return True
