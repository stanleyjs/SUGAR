"""
Synthesis using Geometrically Aligned Random Walks (SUGAR)

Geometry-Based Data Generation https://arxiv.org/abs/1802.04927
    Ofir Lindenbaum, Jay S. Stanley III, Guy Wolf, Smita Krishnaswamy
    Advances in Neural Information Processing Systems 31 (NIPS 2018)
"""

import numpy as np
import graphtools as gt
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy import sparse
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
    noise_k : int, optional, default = 5
        Neighborhood size for k-nn bandwidth noise.
        Only applicable when `noise_cov = 'knn'`.
    sparsity_idx : array-like, shape = [n_dims,] or [n_keep,] optional,
                    default=None
        Column indices of input dimensions to estimate sparsity.
        If None estimate sparsity from complete input column set
        If shape = [n_dims,] `sparsity_idx` will be cast to boolean for logical
            indexing of input columns
        if shape = [n_keep], elements must be unique nonnegative integers
            strictly less than n_samples
    degree_sigma : 'std'
        Diffusion kernel bandwidth.
    degree_k : TYPE
        Description
    degree_a : TYPE
        Description
    degree_fac : TYPE
        Description
    M : TYPE
        Description
    equalize : TYPE
        Description
    mgc_magic : TYPE
        Description
    mgc_sigma : TYPE
        Description
    mgc_a : TYPE
        Description
    mgc_fac : TYPE
        Description
    magic_rescale : TYPE
        Description
    verbose : TYPE
        Description

    Attributes
    ----------
    labels : array_like, shape = [n_samples,], optional, default = None
        Classifier labels to generate data. 
    """

    def __init__(self, noise_cov='knn', noise_k=5,
                 sparsity_idx=None, degree_sigma=5, degree_k=5,
                 degree_a=2, degree_fac=1, M=False, equalize=False,
                 mgc_magic=1, mgc_sigma='knn', mgc_a=2, mgc_k=5,
                 mgc_fac=1, magic_rescale=1, verbose=False):
        # set parameters
        self.noise_cov = noise_cov
        self.noise_k = noise_k
        self.sparsity_idx = sparsity_idx
        self.degree_sigma = degree_sigma
        self.degree_k = degree_k
        self.degree_a = degree_a
        self.degree_fac = degree_fac
        self.M = M
        self.equalize = equalize
        self.mgc_magic = mgc_magic
        self.mgc_sigma = mgc_sigma
        self.mgc_a = mgc_a
        self.mgc_fac = mgc_fac
        self.magic_rescale = magic_rescale
        self.verbose = int(verbose)

    def _check_params(self):
