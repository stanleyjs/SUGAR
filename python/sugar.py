"""
Synthesis using Geometrically Aligned Random Walks (SUGAR)

Geometry-Based Data Generation https://arxiv.org/abs/1802.04927
    Ofir Lindenbaum, Jay S. Stanley III, Guy Wolf, Smita Krishnaswamy
    Advances in Neural Information Processing Systems 31 (NIPS 2018)
"""

import numpy as np
import graphtools as gt
import numbers
import types
import functools
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy import sparse
from scipy.spatial.distances import pdist, squareform
from sklearn.neighbors import NearestNeighbors

import warnings
import tasklogger
from .utils import *


class SUGAR(BaseEstimator):
    """SUGAR operator which performs data generation on input.
    Geometry based data generation via Synthesis Using Geometrically
    Aligned Random Walks(SUGAR) augments high dimensional data by
    generating points along a low dimensional diffusion embedding
    as described Lindenbaum and Stanley et al, 2018.

    Parameters
    ----------
    noise_cov : string, numeric, or callable, optional, default = 'knn'
        Bandwidth of Gaussian noise, i.e. radius of hypersphere for generating
            noise covariance
        If string, choose from ['std','knn','minmax','median']
            'std', use standard deviation of the distances
            'knn', use the distance to the `noise_k` nearest neighbor
            'minmax', use min-max on the distance matrix
            'median', use the median of the distances
        If callable, function f accepts as input a distance matrix
             and returns a number or vector of length [n_samples,]
        If numeric, use the covariance of neighbors within a
            hypersphere of radius `noise_cov`.
    noise_k : positive int, optional, default = 5
        Neighborhood size for k-nn bandwidth noise.
        Only applicable when `noise_cov = 'knn'`.
    degree_sigma : string, numeric, or callable, optional, default = 'std'
        Kernel bandwidth for degree estimation
        See `noise_cov` for details
    degree_k : positive int, optional, default = 5
        Neighborhood size for adaptive bandwidth degree kernel.
        Only applicable when `degree_sigma = 'knn'`.
    degree_a : positive float, optional, default = 2
        Alpha-kernel decay parameter for degree computation.
        2 is Gaussian kernel.
    degree_fac : positive float, optional, default = 1
        Rescale degree kernel bandwidth
    M : non-negative int, optional, default = 0
        Number of points to generate.  Can affect strength of density
        equalization.
        if M>0 and equalize: density equalization will be
             scaled by M.  M < N will negatively impact density
             equalization, M << N is not recommended and M <<< N may fail.
        if M == 0 and equalize: density equalization will not be scaled.
        if M>0  and not equalize: approximately M points will be
             generated according to a constant difference of the
             max density.
        if M == 0 and not equalize: approximately N points will be
             generated.
    equalize : bool, optional, default = False
        Density equalization.  See `M` for details.
    mgc_magic : non-negative float, optional, default = 1
        Time steps of MGC diffusion to perform.
        mgc_magic = 0 disables diffusion.
    mgc_sigma : string, numeric, or callable, optional, default = 'knn'
        Bandwidth for MGC diffusion kernel.
        See `noise_cov` for details.
    mgc_k : positive int, optional, default = 5
        Neighborhood size for adaptive bandwidth MGC kernel.
        Only applicable when `mgc_sigma = 'knn'`.
    mgc_a : positive float, optional, default = 2
        Alpha-kernel decay parameter for MGC kernel. 2 is Gaussian kernel.
    mgc_fac : positive float, optional, default = 1
        Rescale mgc kernel bandwidth
    magic_rescale : bool, optional, default = True
        Percentile-rescale new points after final diffusion.
    distance_metric : string, optional, default = 'euclidean' 
        recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for kernels.
    compute_partial : bool, optional, default = True
        Use knn to compute a truncated distance matrix.  
        Recommended for improved performance
    verbose : bool, optional, default = True
        Enable/disable warnings.
    low_memory : bool, optional, default = True
        Not currently supported.
        Save graphs in memory for reuse.
    Attributes
    ----------

    X : array-like, shape=[n_samples, n_components]
        Input points
    Y : array-like, shape = [n_generated, n_components],
        Output new points
    Y_random : array-like, shape = [n_generated, n_components],
        None if `low_memory`.
        Random points before applying MGC.  If mgc_magic = 0, then
        Y_random = Y.
    sparsity_idx : array-like, shape = [n_components,] or [n_keep,],
                    default=None
        Column indices of input dimensions to estimate sparsity.
        If None estimate sparsity from complete input column set
        If shape = [n_dims,] `sparsity_idx` will be cast to boolean for logical
            indexing of input columns
        if shape = [n_keep,], elements must be unique nonnegative integers
            strictly less than n_samples
    sparsity : array-like , shape = [n_samples,]
        Sparsity estimate on the data
    X_labels : array-like, shape = [n_samples,], default = None
        Classifier labels to generate data.
    Y_labels : array-like, shape = [n_generated,], default = None
        Classifier labels for generated data.
    covs : array_like, shape = [n_samples, n_features, n_features],
        None if `low_memory`
        Local covariances used to generate new points
    deg_kernel : array_like, shape = [n_samples, n_samples],
        None if `low_memory`
        Kernel matrix used to estimate degree distribution
    mgc_kernel : array-like, shape = [Y.shape[0], Y.shape[0]]
        None if `low_memory`
        MGC kernel used to diffuse new points through old points.

    """

    def __init__(self, noise_cov='knn', noise_k=5,
                 degree_sigma=5, degree_k=5,
                 degree_a=2, degree_fac=1, M=False, equalize=False,
                 mgc_magic=1, mgc_sigma='knn', mgc_a=2, mgc_k=5,
                 mgc_fac=1, magic_rescale=1, distance_metric='euclidean',
                 verbose=True, low_memory=False):
        # set parameters
        self.noise_cov = noise_cov
        self.noise_k = noise_k
        self.degree_sigma = degree_sigma
        self.degree_k = degree_k
        self.degree_a = degree_a
        self.degree_fac = degree_fac
        self.M = M
        self.equalize = equalize
        self.mgc_magic = mgc_magic
        self.mgc_sigma = mgc_sigma
        self.mgc_k = mgc_k
        self.mgc_a = mgc_a
        self.mgc_fac = mgc_fac
        self.magic_rescale = magic_rescale
        self.distance_metric = distance_metric
        self.verbose = verbose
        self.low_memory = low_memory
        # set outputs
        self.X = None
        self._Xdists = None
        self.Y = None
        self.Y_random = None
        self.sparsity_idx = None
        self.sparsity = None
        self.X_labels = None
        self.Y_labels = None
        self.covs = None
        self.deg_kernel = None
        self.mgc_kernel = None
        self._check_params()

    def _check_params(self):
        check_positive(noise_k=self.noise_k, degree_k=self.degree_k,
                       degree_a=self.degree_a, degree_fac=self.degree_fac,
                       mgc_k=self.mgc_k, mgc_a=self.mgc_a,
                       mgc_fac=self.mgc_fac)
        check_int(noise_k=self.noise_k, degree_k=self.degre_k,
                  mgc_k=self.mgc_k, M=self.M)
        check_nonnegative(M=self.M, mgc_magic=self.mgc_magic)
        check_in(['std', 'knn', 'minmax', 'median',
                  numbers.Number, types.FunctionType,
                  types.BuiltinFunctionType, functools.partial],
                 self.noise_cov, self.degree_sigma, self. mgc_sigma)
        check_in(['euclidean', 'precomputed',
                  'cosine', 'correlation', 'cityblock',
                  'l1', 'l2', 'manhattan', 'braycurtis', 'canberra',
                  'chebyshev', 'dice', 'hamming', 'jaccard',
                  'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                  'rogerstanimoto', 'russellrao', 'seuclidean',
                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule',
                  'precomputed_affinity', 'precomputed_distance'],
                 knn_dist=self.distance_metric)
        if not isinstance(self.equalize, bool) and \
           self.equalize not in [0, 1]:
            warnings.warn("equalize expected bool, got {}. "
                          "Casting to boolean. ".format(type(self.equalize)))
        self.equalize = int(bool(self.equalize))

        if not isinstance(self.magic_rescale, bool) and \
           self.magic_rescale not in [0, 1]:
            warnings.warn("equalize expected bool, got {}. "
                          "Casting to boolean."
                          " ".format(type(self.magic_rescale)))
        self.magic_rescale = int(bool(self.magic_rescale))

        if not isinstance(self.verbose, bool) and \
           self.verbose not in [0, 1]:
            warnings.warn("verbose expected bool, got {}. "
                          "Casting to boolean. ".format(type(self.verbose)))
        self.verbose = int(bool(self.verbose))
        tasklogger.set_level(self.verbose)

        if not isinstance(self.low_memory, bool) and \
           self.low_memory not in [0, 1]:
            warnings.warn("verbose expected bool, got {}. "
                          "Casting to boolean. ".format(type(self.low_memory)))
        self.low_memory = int(bool(self.low_memory))

    @property
    def covs(self):
        if self.X is not None:
            if self._covs is None:
                tasklogger.log_start("covariance estimation")
                tasklogger.log_info("Computing covariance tensor...")
            else:
                pass
            return self._covs
        else:
            raise NotFittedError("This SUGAR instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments before "
                                 "using this method.")

    def fit(self, X, precomputed=None):

        self.X = X
        if self.distance_metric in ['precomputed',
                                    'precomputed_distance'] \
           and precomputed is not None:
            self._Xdists = precomputed

