SUGAR Geometry-Based Data Generation
-------------------------------------------------------

SUGAR is a tool for generating high dimensional data that follows a low dimensional manifold. SUGAR (Synthesis Using Geometrically Aligned Random-walks) uses a diffusion process to learn a manifold geometry from the data. Then, it generates new points evenly along the manifold by pulling randomly generated points into its intrinsic structure using a diffusion kernel. SUGAR equalizes the density along the manifold by selectively generating points in sparse areas of the manifold.

[Ofir Lindenbaum, Jay S. Stanley III, Guy Wolf, Smita Krishnaswamy **Geometry-Based Data Generation**. 2018. *Arxiv*](https://arxiv.org/abs/1802.04927)


SUGAR has been implemented in Python3 and Matlab.


### Getting started

#### MATLAB installation
1. The MATLAB version of PHATE can be accessed using:

        $ git clone git://github.com/stanleyjs/SUGAR.git
        $ cd SUGAR/Matlab

2. Add the SUGAR/Matlab directory to your MATLAB path and run any of our `test` scripts to get a feel for SUGAR.
