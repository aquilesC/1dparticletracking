How does it work?
=================

I will here give a basic explanation of how the algorithm works. For deeper information I refer you to
`Sbalzarini, Ivo F., and Petros Koumoutsakos., Journal of structural biology 151.2 (2005): 182-195.`__
Note also that some parts differ from the algorithm in the paper and I will hence point out these in this description.

__ https://www.sciencedirect.com/science/article/pii/S1047847705001267


Finding particle positions
--------------------------

To find particle positions the tracker first finds local intensity maximas with intensities higher than the attribute `feature_point_threshold`. These positions
are assumed to be particle observations. To get closer to the actual particle position a refinement of these positions is done by finding the center of mass in the range
`integration_radius_of_intensity_peaks` around the initial position. If two local maximas are found within the distance of two times the `integration_radius_of_intensity_peaks`,
the one with lowest first intensity moment is discarded. This discrimination is not in the paper but was necessary in the case of very noisy data.

Linking feature points
----------------------

When the particle positions are found an association matrix is created with a corresponding cost matrix. The association matrix contains information about
links between particle positions and has the form

.. math::

    G^{t}_{r} = g_{ij} = \begin{cases} 1 & \text{if there is a link between particle }i \text{ in frame } t \text{ and particle j in frame } t+r \\ 0 & \text{else} \end{cases}.

The first row :math: `g_{0j}` and the first column :math: `g_{i0}` represents dummy particles and allows for particles to disappear (getting out of focus or similar) in a few frames
and still being able to be used in trajectories when it reappears.
The corresponding cost matrix has the same shape as the association matrix but instead contains values descibing the cost to link particle :math: `i` in frame :math: `t` with particle
:math: `j` in frame :math: `t+r`.

Cost function
-------------

The cost function is a function of two particle positions and their associated intensity and describes how costly it is to link two particles into a trajectory.
The default cost function is

Optimisation
------------


Analysing the trajectories
--------------------------

Velocity autocorrelation
------------------------

Diffusion coefficient
---------------------
The software comes with two methods of determining the diffusion coefficient, either by fitting a straight line to the mean squared displacement function or by a covariance
based estimator. For more information about determining diffusion coefficients, see
