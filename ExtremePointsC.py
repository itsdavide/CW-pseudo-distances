#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D.Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes the extreme points of the set of joint belief
functions with given marginals by means of extreme copulas (minimum and 
Lukasiewicz) by varying the permutations of the marginal underlying spaces.
The code shows that some extreme points cannot be obtained in this way.
"""

import numpy as np
from itertools import permutations
import itertools


def T_M(x, y):
    return np.minimum(x, y)

def T_L(x, y, NORM):
    return np.maximum(x + y - NORM, 0)

def ExtremePointsC(NORM, P, Q):
    # Extract index sets
    I = np.array(range(len(P)))
    J = np.array(range(len(Q)))

    Perm_I = [list(p) for p in permutations(I)]
    Perm_J = [list(p) for p in permutations(J)]

    vertices = []

    for sigma in Perm_I:
        for pi in Perm_J:
            J_M = np.zeros((len(P), len(Q)))
            J_L = np.zeros((len(P), len(Q)))
            C_P = 0
            for i in range(len(P)):
                C_P_prec = C_P
                C_P += P[sigma[i]]
                C_Q = 0
                for j in range(len(Q)):
                    C_Q_prec = C_Q
                    C_Q += Q[pi[j]] 
                    J_M[sigma[i], pi[j]] = T_M(C_P, C_Q) - T_M(C_P_prec, C_Q) - T_M(C_P, C_Q_prec) + T_M(C_P_prec, C_Q_prec)
                    J_L[sigma[i], pi[j]] = T_L(C_P, C_Q, NORM) - T_L(C_P_prec, C_Q, NORM) - T_L(C_P, C_Q_prec, NORM) + T_L(C_P_prec, C_Q_prec, NORM)
            vertices.append(list(J_M.flatten()))
            vertices.append(list(J_L.flatten()))
            
    vertices.sort()
    vertices = list(vertices for vertices,_ in itertools.groupby(vertices))
    return np.array(vertices)