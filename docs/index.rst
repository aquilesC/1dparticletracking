.. particle-tracker-one-d documentation master file, created by
   sphinx-quickstart on Thu Oct 17 15:34:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to particle-tracker-one-d's documentation!
==================================================

This software was developed to track particles in Kymographs, eg. intensity graphs. It is based on the article
`Sbalzarini, Ivo F., and Petros Koumoutsakos., Journal of structural biology 151.2 (2005): 182-195`__
to track feature points but with the exception, that in this implementation the only noise filtering is the boxcar
averaging and the frames are one dimensional. The feature points that are found are linked together by minimising a cost function and results in one or
more trajectories.

From the trajectories one can easily plot the velocity auto correlation function and calculate the diffusion
coefficient from either the mean squared displacement function or by a covariance based estimator.

__ https://www.sciencedirect.com/science/article/pii/S1047847705001267

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/Installation
   usage/Quickstart
   usage/HowDoesItWork
   usage/ParticleTracker
   usage/ShortestPathFinder
   usage/Trajectory


Indices and tables
==================

* :ref:`search`
