import numpy as np


class Hindrance:

    def __init__(self):
        self._channel_x_dimension = None
        self._channel_y_dimension = None
        self._molecule_radius = None

    @property
    def channel_x_dimension(self):
        return self._channel_x_dimension

    @channel_x_dimension.setter
    def channel_x_dimension(self, dimension):
        self._channel_x_dimension = dimension

    @property
    def channel_y_dimension(self):
        return self._channel_y_dimension

    @channel_y_dimension.setter
    def channel_y_dimension(self, dimension):
        self._channel_y_dimension = dimension

    @property
    def molecule_radius(self):
        return self._molecule_radius

    @molecule_radius.setter
    def molecule_radius(self, radius):
        self._molecule_radius = radius

    def _calculate_equilibrium_partition_coefficient(self):
        if self._channel_x_dimension and self._channel_y_dimension and self._molecule_radius:
            return (1 - self._molecule_radius / np.sqrt(self._channel_x_dimension * self._channel_y_dimension))**2
        else:
            raise ValueError('Channel dimensions or molecule radius has not been set.')

    def calculate_hindrance_factor(self):
        equilibrium_partition_coefficient = self._calculate_equilibrium_partition_coefficient()
        ratio_molecule_size_and_dimension = self._molecule_radius / np.sqrt(self._channel_x_dimension * self._channel_y_dimension)
        H = 1 + 9 / 8 * ratio_molecule_size_and_dimension * np.log(ratio_molecule_size_and_dimension) - 1.56034 * ratio_molecule_size_and_dimension + \
            0.528155 * ratio_molecule_size_and_dimension ** 2 + 1.91521 * ratio_molecule_size_and_dimension ** 3 - \
            2.81903 * ratio_molecule_size_and_dimension ** 4 + 0.270788 * ratio_molecule_size_and_dimension ** 5 + \
            1.10115 * ratio_molecule_size_and_dimension ** 6 - 0.435933 * ratio_molecule_size_and_dimension ** 7
        return H / equilibrium_partition_coefficient
