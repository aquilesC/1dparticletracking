How does it work?
=================

I will here give a basic explanation of how the algorithm works. For deeper information I refer you to
`Sbalzarini, Ivo F., and Petros Koumoutsakos., Journal of structural biology 151.2 (2005): 182-195`__

__ https://www.sciencedirect.com/science/article/pii/S1047847705001267


Finding particle positions
--------------------------

To find particle positions the tracker first finds local intensity maximas with
intensities higher than the attribute `feature_point_threshold`. These positions
are assumed to be particle observations. To get closer to the actual particle position
a refinement of these positions is done by finding the center of mass in the range
`integration_radius_of_intensity_peaks` around the initial position.

Linking feature points
----------------------


Cost function
-------------

Optimisation
------------


Analysing the trajectories
--------------------------

Velocity autocorrelation
------------------------

Diffusion coefficient
---------------------
The software comes with two methods of determining the diffusion coefficient, either by fitting a straight line to
the mean squared displacement function or by a covariance based estimator. For more information about determining
diffusion coefficients, see



The particle tracker finds feature points in the graphs