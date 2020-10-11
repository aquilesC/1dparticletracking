import numpy as np
import matplotlib.pyplot as plt


class Trajectory:
    """
    Object that describes a trajectory. With functions for checking if the trajectory describes real diffusion,
    convenient plotting and calculations of diffusion coefficients.

    Parameters
    ----------
    pixel_width: float
        Defines the length one pixel corresponds to. This value will be used when calculating diffusion
        coefficients. Default is 1.

    Attributes
    ----------
    pixel_width
    particle_positions: np.array
        Numpy array with all particle positions in the trajectory on the form `np.array((nParticles,), dtype=[('frame_index', np.int16),
        ('time', np.float32),('position', np.int16),('zeroth_order_moment', np.float32),('second_order_moment', np.float32)])`
    """

    def __init__(self, pixel_width=1):
        self.particle_positions = np.empty((0,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                        ('second_order_moment', np.float32)])
        self.pixel_width = pixel_width

    def __add__(self, other):
        new_trajectory = Trajectory(pixel_width=self.pixel_width)
        if self.pixel_width != other.pixel_width:
            raise ValueError('Pixel width must be equal when adding trajectories together.')
        elif self.particle_positions.shape[0] == 0 and other._particle_positions.shape[0] == 0:
            return new_trajectory
        elif self.particle_positions.shape == (0,):
            new_trajectory.particle_positions = other._particle_positions
        elif other._particle_positions.shape == (0,):
            new_trajectory.particle_positions = self.particle_positions
        elif other._particle_positions[0]['frame_index'] == self.particle_positions[0]['frame_index']:
            raise ValueError('Both trajectories cant start at same frame index.')
        elif other._particle_positions.shape[0] == 1 and self.particle_positions.shape[0] == 1:
            if other._particle_positions['frame_index'][0] < self.particle_positions['frame_index'][0]:
                new_trajectory.particle_positions = np.append(other._particle_positions, self.particle_positions)
            else:
                new_trajectory.particle_positions = np.append(self.particle_positions, other._particle_positions)
        elif other._particle_positions['frame_index'][0] < self.particle_positions['frame_index'][0]:
            index = np.where(
                other._particle_positions['frame_index'] < self.particle_positions[0]['frame_index']
            )
            new_trajectory.particle_positions = np.append(other._particle_positions[index], self.particle_positions)
        elif self.particle_positions['frame_index'][0] < other._particle_positions['frame_index'][0]:
            index = np.where(
                self.particle_positions['frame_index'] < other._particle_positions[0]['frame_index']
            )
            new_trajectory.particle_positions = np.append(self.particle_positions[index], other._particle_positions)

        return new_trajectory

    @property
    def velocities(self):
        """
        np.array:
            The velocities the particle moves at.
        """
        if self.length > 1:
            return self._calculate_particle_velocities()
        return 0

    @property
    def density(self):
        """
        float:
            How dense the trajectory is in time. Returns self.length/(self.particle_positions['frame_index'][-1]-self.particle_positions['frame_index'][0]).
        """
        if self.length == 0 or self.length == 1:
            return 1
        return self.length / (1 + self.particle_positions['frame_index'][-1] - self.particle_positions['frame_index'][0])

    @property
    def length(self):
        """
        int:
            The length of the trajectory. Returns self.particle_postions.shape[0]
        """
        return self.particle_positions.shape[0]

    def overlaps_with(self, trajectory):
        """
        Check if the trajectories overlaps

        trajectory: Trajectory to compare with. If both trajectories has any identical elements will return true otherwise false.

        Returns
        -------
            bool
        """
        if self.length == 0 or trajectory.length == 0:
            return False
        for p in trajectory.particle_positions:
            for p2 in self.particle_positions:
                if (p['frame_index'] == p2['frame_index']) and (p['position'] == p2['position']):
                    return True
        return False

    def split(self, trajectory):
        """
        If two trajectories overlaps, this function will split them into three or more non overlapping trajectories.

        trajectory:

        Returns
        -------
            list
                Returns a list with the new trajectories
        """
        new_trajectories = []
        while (self.length > 0) or (trajectory.length > 0):
            if (self.length == 0) and (trajectory.length > 0):
                new_trajectory = Trajectory(pixel_width=trajectory.pixel_width)
                new_trajectory.particle_positions = trajectory.particle_positions
                trajectory.particle_positions = np.array([])
                new_trajectories.append(new_trajectory)
            elif (trajectory.length == 0) and (self.length > 0):
                new_trajectory = Trajectory(pixel_width=self.pixel_width)
                new_trajectory.particle_positions = self.particle_positions
                self.particle_positions = np.array([])
                new_trajectories.append(new_trajectory)
            elif self._first_position_is_same(self, trajectory):
                self.particle_positions, trajectory.particle_positions, overlapping_trajectory = self._extract_first_overlapping_trajectory(self, trajectory)
                new_trajectories.append(overlapping_trajectory)
            else:
                self.particle_positions, trajectory.particle_positions, not_overlapping_trajectory_1, not_overlapping_trajectory_2 = self._extract_first_non_overlapping_trajectories(
                    self, trajectory)
                if not_overlapping_trajectory_1 is not None:
                    new_trajectories.append(not_overlapping_trajectory_1)
                if not_overlapping_trajectory_2 is not None:
                    new_trajectories.append(not_overlapping_trajectory_2)
        return new_trajectories

    def plot_trajectory(self, x='frame_index', y='position', ax=None, **kwargs):
        """
        Plots the trajectory using the frame index and the particle position in pixels.

        x: str
            'frame_index', 'time', 'position' (default), 'zeroth_order_moment', 'second_order_moment' choose the x-axis value
        y: str
            'frame_index' (default), 'time', 'position', 'zeroth_order_moment', 'second_order_moment' choose the y-axis value
        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of a matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        ax.plot(self.particle_positions[x], self.particle_positions[y], **kwargs)
        return ax

    def plot_velocity_auto_correlation(self, ax=None, **kwargs):
        """
        Plots the particle velocity auto correlation function which can be used for examining if the trajectory
        describes free diffusion.

        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of a matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        ax.acorr(self.velocities, **kwargs)
        return ax

    def calculate_mean_square_displacement_function(self):
        """
        Calculate the average squared displacements for different time steps.

        Returns
        -------
            time: np.array
                The time corresponding to the mean squared displacements.

            msd: np.array
                The mean squared displacements of the trajectory.
        """

        mean_square_displacements = np.zeros((self.particle_positions['frame_index'][-1] - self.particle_positions['frame_index'][0] + 1,),
                                             dtype=[('msd', np.float32), ('nr_of_values', np.int16)])
        times = np.arange(0, self.particle_positions['frame_index'][-1] - self.particle_positions['frame_index'][0] + 1, dtype=np.float32) * self._calculate_time_step()

        for first_index, first_position in enumerate(self.particle_positions[:-1]):
            for second_index, second_position in enumerate(self.particle_positions[first_index + 1:]):
                index_difference = second_position['frame_index'] - first_position['frame_index']
                mean_square_displacements['msd'][index_difference] += ((second_position['position'] - first_position['position']) * self.pixel_width) ** 2
                mean_square_displacements['nr_of_values'][index_difference] += 1

        for index, msd in enumerate(mean_square_displacements):
            if mean_square_displacements['nr_of_values'][index] != 0:
                mean_square_displacements['msd'][index] = msd['msd'] / mean_square_displacements['nr_of_values'][index].astype(np.float32)

        non_zeros_indices = np.nonzero(mean_square_displacements['nr_of_values'])
        return times[non_zeros_indices], mean_square_displacements['msd'][non_zeros_indices]

    def _append_position(self, particle_position):
        self.particle_positions = np.append(self.particle_positions, particle_position, axis=0)

    def _position_exists_in_trajectory(self, particle_position):
        for p in self.particle_positions:
            if np.array_equal(p, particle_position):
                return True

    def _calculate_particle_velocities(self):
        time_steps = np.diff(self.particle_positions['time'])
        position_steps = np.diff(self.particle_positions['position'] * self.pixel_width)
        return position_steps / time_steps

    @staticmethod
    def _remove_non_unique_values(array):
        return np.unique(array)

    @staticmethod
    def _sort_values_low_to_high(array):
        return np.sort(array)

    def calculate_diffusion_coefficient_from_mean_square_displacement_function(self, fit_range=None):
        """
        Fits a straight line to the mean square displacement function and calculates the diffusion coefficient from the
        gradient of the line. The mean squared displacement of the particle position is proportional to :math:`2Dt`
        where :math:`D` is the diffusion coefficient and :math:`t` is the time.

        fit_range: list, None (default)
            Define the range of the fit, the data for the fit will be `time[fit_range[0]:fit_range[1]`` and `mean_squared_displacement[fit_range[0]:fit_range[1]]`.

        Returns
        -------
            diffusion_coefficient: float
            error: float
        """
        time, mean_square_displacement = self.calculate_mean_square_displacement_function()
        if fit_range is None:
            polynomial_coefficients, error_estimate = self._fit_straight_line_to_data(time, mean_square_displacement)
        else:
            polynomial_coefficients, error_estimate = self._fit_straight_line_to_data(time[fit_range[0]:fit_range[1]], mean_square_displacement[fit_range[0]:fit_range[1]])
        return polynomial_coefficients[0] / 2, error_estimate[0] / 2

    @staticmethod
    def _fit_straight_line_to_data(x, y):
        polynomial_coefficients, covariance_matrix = np.polyfit(x, y, 1, cov=True)
        error_estimate = [np.sqrt(covariance_matrix[0, 0]), np.sqrt(covariance_matrix[1, 1])]
        return polynomial_coefficients, error_estimate

    def calculate_diffusion_coefficient_using_covariance_based_estimator(self, R=None):
        """
        Unbiased estimator of the diffusion coefficient. More info at `https://www.nature.com/articles/nmeth.2904`.
        If the motion blur coefficient is entered a variance estimate is also calculated.

        R: float, motion blur coefficient

        Returns
        -------
            diffusion_coefficient: float
            variance_estimate: float
        """
        squared_displacements = []
        covariance_term = []

        for index, first_position in enumerate(self.particle_positions[:-2]):
            second_position = self.particle_positions[index + 1]
            third_position = self.particle_positions[index + 2]
            if first_position['frame_index'] - second_position['frame_index'] == -1 and first_position['frame_index'] - third_position['frame_index'] == -2:
                squared_displacements.append((self.pixel_width * (second_position['position'] - first_position['position'])) ** 2)
                covariance_term.append((second_position['position'] - first_position['position']) * (
                        third_position['position'] - second_position['position']) * self.pixel_width ** 2)

        time_step = self._calculate_time_step()
        number_of_points_used = len(squared_displacements)
        diffusion_coefficient = np.mean(squared_displacements) / (2 * time_step) + np.mean(covariance_term) / time_step

        if R is not None:
            localisation_error = R * np.mean(squared_displacements) + (2 * R - 1) * np.mean(covariance_term)
            epsilon = localisation_error ** 2 / (diffusion_coefficient * time_step) - 2 * R
            variance_estimate = diffusion_coefficient ** 2 * ((6 + 4 * epsilon + 2 * epsilon ** 2) / number_of_points_used + 4 * (1 + epsilon) ** 2 / (number_of_points_used ** 2))
            return diffusion_coefficient, variance_estimate

        return diffusion_coefficient

    def _calculate_time_step(self):
        return (self.particle_positions['time'][1] - self.particle_positions['time'][0]) / (self.particle_positions['frame_index'][1] - self.particle_positions['frame_index'][0])

    @staticmethod
    def _first_position_is_same(t1, t2):
        return t1.particle_positions[0] == t2.particle_positions[0]

    @staticmethod
    def _extract_first_overlapping_trajectory(t1, t2):
        n = 0
        length_of_shortest_trajectory = min(t1.length, t2.length)
        while (n < length_of_shortest_trajectory) and (t1.particle_positions[n] == t2.particle_positions[n]):
            n += 1
        overlapping_trajectory = Trajectory(pixel_width=t1.pixel_width)
        overlapping_trajectory.particle_positions = t1.particle_positions[:n]
        return t1.particle_positions[n:], t2.particle_positions[n:], overlapping_trajectory

    @staticmethod
    def _extract_first_non_overlapping_trajectories(t1, t2):
        n1 = 0
        n2 = 0
        while (n1 < t1.length) and (n2 < t2.length) and (t1.particle_positions[n1] != t2.particle_positions[n2]):
            if t1.particle_positions[n1]['frame_index'] < t2.particle_positions[n2]['frame_index']:
                n1 += 1
            elif t1.particle_positions[n1]['frame_index'] > t2.particle_positions[n2]['frame_index']:
                n2 += 1
            else:
                n1 += 1
                n2 += 1
        if n1 > 0:
            first_part_of_t1 = Trajectory(pixel_width=t1.pixel_width)
            first_part_of_t1.particle_positions = t1.particle_positions[:n1]
        else:
            first_part_of_t1 = None
        if n2 > 0:
            first_part_of_t2 = Trajectory(pixel_width=t2.pixel_width)
            first_part_of_t2.particle_positions = t2.particle_positions[:n1]
        else:
            first_part_of_t2 = None
        return t1.particle_positions[n1:], t2.particle_positions[n2:], first_part_of_t1, first_part_of_t2
