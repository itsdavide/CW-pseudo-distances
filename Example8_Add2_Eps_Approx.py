#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D.Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes the pessimistic (alpha = 1) and optimistic 
(alpha = 0) Choquet-Wasserstein approximation of a 2-additive belief function
with an epsilon-contamination model, using an entropic regularization with 
parameter lamb. The ground metric is assumed to be the discrete metric.
In this case the order p in [1,+infinity) is irrelevant.
"""

import numpy as np

c_min = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    ])

c_max = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    ])

alpha = 0

c = alpha * c_min + (1 - alpha) * c_max

w = np.array([3, 1, 2])
m_mu = w / w.sum()

print('m_mu:', m_mu, 'with sum = ', sum(m_mu))

def prox_G1(theta):
    gamma = np.zeros_like(theta)
    (m, n) = theta.shape
    for i in range(m):
        for j in range(n):
            gamma[i, j] = m_mu[i] * theta[i, j] / sum(theta[i, :])
    return gamma


Lambdas = np.arange(0.2, 0, -0.01)

for lamb in Lambdas:
    m_gamma0 = np.exp(- c / lamb)
    
    z = np.ones(c.shape)

    print ('lambda:', round(lamb, 4))
    m_gamma_old = m_gamma0
    for n in range(2000):
        m_gamma = prox_G1(m_gamma_old * z)
        z = z * (m_gamma_old / m_gamma)
        if (np.sum(np.abs(m_gamma - m_gamma_old)) < 0.00000001):
            break
        m_gamma_old = m_gamma
    
    print('m_gamma:\n', np.round(m_gamma, 6), 'with sum =', round(sum(sum(m_gamma)), 4), '\n')
    print('m_mu:', np.round(np.sum(m_gamma, axis=1), 6), 'with sum = ', round(np.sum(m_gamma), 4))
    print('m_nu:', np.round(np.sum(m_gamma, axis=0), 6), 'with sum = ', round(np.sum(m_gamma), 4))
    (m, n) = c.shape
    d = 0
    for i in range(m):
        for j in range(n):
            d += c[i, j] * m_gamma[i ,j]
    print('d = ', round(d, 4), '\n')
    
    
    
    