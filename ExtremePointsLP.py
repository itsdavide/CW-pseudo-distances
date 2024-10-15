#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D. Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes the extreme points of the set of joint belief
functions with given marginals, with a vertex enumeration algorithm.
Extreme points are found referring to the polytope given by A*x <= b, where we
add the non-negativity constraint: -x <= 0.
"""

import numpy as np
from pypoman import compute_polytope_vertices

def ExtremePointsLP(NORM, P, Q):
    # Non-negativity constraints
    nm = len(P) * len(Q)
    A = np.diag(-np.ones(nm))
    b = np.zeros(nm)
    
    # Normalization contraint
    A = np.vstack((A, -np.ones(nm)))
    A = np.vstack((A, np.ones(nm)))
    b = np.append(b, -NORM)
    b = np.append(b, NORM)
    
    # P marginal constraints
    for i in range(len(P)):
        new = np.zeros(nm)
        for j in range(len(Q)):
            new[len(Q)*i + j] = 1
        A = np.vstack((A, -new))
        A = np.vstack((A, new))
    for i in range(len(P)):
        b = np.append(b, -P[i])
        b = np.append(b, P[i])
    
    # Q marginal constraints
    for i in range(len(Q)):
        new = np.zeros(nm)
        for j in range(len(P)):
            new[i + len(Q)*j] = 1
        A = np.vstack((A, -new))
        A = np.vstack((A, new))
    for i in range(len(Q)):
        b = np.append(b, -Q[i])
        b = np.append(b, Q[i])
        
        
    vertices = compute_polytope_vertices(A, b)
    return vertices