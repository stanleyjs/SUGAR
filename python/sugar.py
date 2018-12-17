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
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

import warnings
import tasklogger
from utils import *


def log_start(message):
    tasklogger.log_start(message, logger="SUGAR")


def log_complete(message):
    tasklogger.log_complete(message, logger="SUGAR")


class SUGAR(BaseEstimator):
    """SUGAR operator which performs data generation on input.
    Geometry based data generation via Synthesis Using Geometrically
    Aligned Random Walks(SUGAR) augments high dimensional data by
    generating points along a low dimensional diffusion embedding
    as described Lindenbaum and Stanley et al, 2018.

    Parameters
    ----------
    noise_cov : string, numeric, or callable, optional, default = None
        Bandwidth of Gaussian noise, i.e. radius of hypersphere for generating
            noise covariance
        If string, choose from ['std','knn','minmax','median']
            'std', use standard deviation of the distances
            None/'knn', use the distance to the `noise_k` nearest neighbor
            'minmax', use min-max on the distance matrix
            'median', use the median of the distances
        If callable, function f accepts as input a distance matrix
             and returns a number or vector of length [n_samples,]
        If numeric, use the covariance of neighbors within a
            hypersphere of radius `noise_cov`.
        If None, defaults to knn.
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
    generation_estimate : distibution of points generated around each original point.
    sparsity_idx : array-like, shape = [n_components,] or [n_keep,],
                    default=None
        Column indices of input dimensions to estimate sparsity.
        If None estimate sparsity from complete input column set
        If shape = [n_dims,] `sparsity_idx` will be cast to boolean for logical
            indexing of input columns
        if shape = [n_keep,], elements must be unique nonnegative integers
            strictly less than n_dim
    sparsity_estimate : array-like , shape = [n_samples,]
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

    def __init__(self, noise_cov=None, noise_k=5,
                 degree_sigma='std', degree_k=5,
                 degree_a=2, degree_fac=1, M=False, equalize=False,
                 mgc_magic=1, mgc_sigma=None, mgc_a=2, mgc_k=5,
                 mgc_fac=1, magic_rescale=1, distance_metric='euclidean',
                 verbose=True, low_memory=False):
        # set parameters
        self.noise_cov = string_lower(noise_cov)
        self.noise_k = noise_k
        self.degree_sigma = string_lower(degree_sigma)
        self.degree_k = degree_k
        self.degree_a = degree_a
        self.degree_fac = degree_fac
        self.M = M
        self.equalize = equalize
        self.mgc_magic = mgc_magic
        self.mgc_sigma = string_lower(mgc_sigma)
        self.mgc_k = mgc_k
        self.mgc_a = mgc_a
        self.mgc_fac = mgc_fac
        self.magic_rescale = magic_rescale
        self.distance_metric = distance_metric
        self.verbose = verbose
        self.low_memory = low_memory
        # set outputs
        self._reset_model()
        self._check_params()
        self._convert_sigmas()

    def _reset_model(self):
        self._X = None
        self._N = None
        self._Xdists = None
        self._Xg = None
        self._gen_est = None
        self._covs = None
        self._Y = None
        self._Y_random = None
        self.sparsity_idx = None
        self.sparsity_estimate = None
        self.X_labels = None
        self.Y_labels = None
        self.deg_kernel = None
        self.mgc_kernel = None
        self._Xdists = None
        self._covs = None

    def _check_params(self):
        check_positive(noise_k=self.noise_k, degree_k=self.degree_k,
                       degree_a=self.degree_a, degree_fac=self.degree_fac,
                       mgc_k=self.mgc_k, mgc_a=self.mgc_a,
                       mgc_fac=self.mgc_fac)
        check_int(noise_k=self.noise_k, degree_k=self.degree_k,
                  mgc_k=self.mgc_k, M=self.M)
        check_nonnegative(M=self.M, mgc_magic=self.mgc_magic)
        check_in(['std', 'knn', 'minmax', 'median', None,
                  numbers.Number, types.FunctionType,
                  types.BuiltinFunctionType, functools.partial],
                 noise_cov=self.noise_cov,
                 degree_sigma=self.degree_sigma,
                 mgc_sigma=self. mgc_sigma)
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
        logger = tasklogger.set_level(self.verbose, logger="SUGAR")
        logger.min_runtime = -1
        if not isinstance(self.low_memory, bool) and \
           self.low_memory not in [0, 1]:
            warnings.warn("verbose expected bool, got {}. "
                          "Casting to boolean. ".format(type(self.low_memory)))
        self.low_memory = int(bool(self.low_memory))

    def _convert_sigmas(self):
        """ Convert string-valued bandwidth arguments to corresponding
        anonymous functions defined on matrices."""

        func_dict = {'knn': None,
                     'std': lambda x: np.std(x),
                     'minmax': lambda x: np.min(np.max(x, axis=1)),
                     'median': lambda x: np.median(x)}
        if isinstance(self.noise_cov, str):
            self.noise_cov = func_dict[self.noise_cov]
        if isinstance(self.degree_sigma, str):
            self.degree_sigma = func_dict[self.degree_sigma]
        if isinstance(self.mgc_sigma, str):
            self.mgc_sigma = func_dict[self.mgc_sigma]

    @property
    def N(self):
        if self._N is None:
            self._N = self.X.shape[0]
        return self._N

    @property
    def covs(self):
        if self._covs is None:
            log_start("covariance estimation")
            self._covs = []
            if self.noise_cov is None:
                noise_nbrs = np.argpartition(
                    self.Xdists,
                    self.noise_k + 1, axis=1)[
                    :, :self.noise_k]
                noise_nbrs = {i: p for i, p in enumerate(noise_nbrs)}
            else:
                if callable(self.noise_cov):
                    noise_bw = self.noise_cov(self.Xdists)
                else:
                    noise_bw = self.noise_cov
                noise_tmp = np.where(self.Xdists <= noise_bw)
                noise_nbrs = {}
                for key, value in zip(*noise_tmp):
                    if key in noise_nbrs:
                        noise_nbrs[key] = np.append(noise_nbrs[key], value)
                    else:
                        noise_nbrs[key] = value
            for ix in range(0, self._N):
                self._covs.append(np.cov(self.X[noise_nbrs[ix], :].T))
            log_complete("covariance estimation")
        else:
            pass
        return self._covs

    @property
    def Xdists(self):
        if self._Xdists is None:
            log_start("distance matrix")
            self._Xdists = squareform(pdist(self.X,
                                            metric=self.distance_metric))
            log_complete("distance matrix")
        return self._Xdists

    @property
    def Xg(self):
        if self._Xg is None:
            log_start("sparsity kernel")
            if self.sparsity_idx is not None:
                self._Xg = gt.Graph(self.X[:, self.sparsity_idx],
                                    bandwidth=self.degree_sigma,
                                    bandwidth_fac=self.degree_fac,
                                    decay=self.degree_a,
                                    knn=self.degree_k)
            else:
                self._Xg = gt.Graph(self.Xdists, bandwidth=self.degree_sigma,
                                    knn=self.degree_k, decay=self.degree_a,
                                    bandwidth_fac=self.degree_fac,
                                    precomputed='distance')
            log_complete("sparsity kernel")

        return self._Xg

    def compute_sparsity(self, X=None, sparsity_idx=None):
        log_start("sparsity estimate")

        if X is not None:
            if sparsity_idx is not None:
                X = X[:, sparsity_idx]

            g = gt.Graph(X,
                         bandwidth=self.degree_sigma,
                         bandwidth_fac=self.degree_fac,
                         decay=self.degree_a,
                         knn=self.degree_k)
            s = 1 / g.K.sum(axis=1)
            del g
        else:
            s = 1 / self.Xg.K.sum(axis=1)
            if self.low_memory:
                del self._Xg
                self._Xg = None
        log_complete("sparsity estimate")
        return s

    def estimate_generation(self, precomputed=False):
        """estimate_generation: compute the amount of points to generate for every x_i.

        Returns
        -------
        np.array(shape=(n_samples,))
            Estimated amount of points to generate around each original point.
        """
        log_start("generation estimate")
        self.sparsity_estimate = self.compute_sparsity()
        log_complete("generation estimate")
        return self._gen_est

    @property
    def X(self):
        if self._X is None:
            raise NotFittedError("This SUGAR instance is not fitted yet. "
                                 "Call `fit` with appropriate arguments "
                                 "before using this method.")
        else:
            return self._X

    @property
    def generation_estimate(self):
        if self._gen_est is None:
            self.estimate_generation()
        return self._gen_est

    @property
    def Y_random(self):
        if self._Y_random is None:
            self.generate_points()
        return self._Y_random

    def generate_points(self, X=None, gen_est=None):
        if X is None:
            X = self.X
        if gen_est is None:
            gen_est = self.generation_estimate
        self._Y_random = np.ndarray((np.sum(gen_est), X.shape[1]))
        cur_idx = 0
        for ix, ell in enumerate(gen_est):
            self._Y_random[cur_idx:cur_idx + ell, :] = \
                np.random.multivariate_normal(self.X[ix, :],
                                              self.covs[ix],
                                              ell)
            cur_idx += ell
        return self.Y_random

    def fit(self, X, sparsity_idx=None, precomputed=None, refit=True):
        """Fit the SUGAR estimator to the data

        Parameters
        ----------
        X : np.ndarray(shape=(nsamples, nfeatures))
        sparsity_idx : array-like, shape = [n_components,] or [n_keep,],
                    default=None
            Column indices of input dimensions to estimate sparsity.
            If None estimate sparsity from complete input column set
            If shape = [n_dims,] `sparsity_idx` will be cast to boolean
                for logical indexing of input columns
            If shape = [n_keep,], elements must be unique nonnegative integers
                strictly less than n_dim
        precomputed : None, optional
            Precomputed distance matrix to compute degree 
            and covariance estimates.
        """
        if refit:
            self._reset_model()
        self._X = X

        self.sparsity_idx = sparsity_idx
        if self.distance_metric in ['precomputed',
                                    'precomputed_distance'] \
           and precomputed is not None:
            self._Xdists = precomputed
            if self.sparsity_idx is not None:
                warnings.warn("Precomputed distance matrix overrides"
                              "sparsity_idx.")
            self.sparsity_idx = None

        self.estimate_generation()
