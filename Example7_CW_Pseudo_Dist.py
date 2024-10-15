#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D. Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes the entropic regularized Choquet-Wasserstein 
pseudo-distances between two belief functions on the same metric space
with absolute value ground metric, for different values of the regularization
parameter. The computation is done with a modification of Dykstra's algortithm.
"""

import numpy as np

tolerance = 10**(-8)

print('TOLERANCE:', tolerance)

# Order of the Choquet-Wasserstein pseudo-distances
p = 1

x = np.arange(1, 5, 1)
y = x

print('Space X:', x)

# Pessimism index (alpha = 1: pessimistic; alpha = 0: optimistic)
alpha = 1

c = np.zeros((len(x) + 1, len(y) + 1))
for i in range(len(x)):
    for j in range(len(y)):
        c[i, j] = abs(x[i] - y[j])**p
for i in range(len(x)):
    c[i, len(y)] = alpha * c[i, 0:len(y)].min() + (1 - alpha) * c[i, 0:len(y)].max()
for j in range(len(y)):
    c[len(x), j] = alpha * c[0:len(x),j].min() + (1 - alpha) * c[0:len(x),j].max()
c[len(x),len(y)] = alpha * c.min() + (1 - alpha) * c.max()
print('c:')
print(c)

w1 = np.array([2, 2, 4, 1, 1])
m_mu = w1 / w1.sum()
supp_m_mu = (m_mu > 0)

w2 = np.array([4, 2, 2, 1, 1])
m_nu = w2 / w2.sum()
supp_m_nu = (m_nu > 0)

(m, n) = c.shape
supp_m_gamma = np.full((m, n), True)
for i in range(m):
    for j in range(n):
        supp_m_gamma[i, j] = supp_m_mu[i] and supp_m_nu[j]

print('Given marginals:')
print('m_mu:', np.round(m_mu, 6), 'with sum = ', round(np.sum(m_mu), 4))
print('m_nu:', np.round(m_nu, 6), 'with sum = ', round(np.sum(m_nu), 4))

def prox_G1(theta):
    m_gamma = np.zeros_like(theta)
    (m, n) = theta.shape
    for i in range(m):
        for j in range(n):
            if supp_m_mu[i] and supp_m_nu[j]:
                m_gamma[i, j] = m_mu[i] * theta[i, j] / sum(theta[i, :])
    return m_gamma

def prox_G2(theta):
    m_gamma = np.zeros_like(theta)
    (m, n) = theta.shape
    for i in range(m):
        for j in range(n):
            if supp_m_mu[i] and supp_m_nu[j]:
                m_gamma[i, j] = m_nu[j] * theta[i, j] / sum(theta[:, j])
    return m_gamma


Lambdas = np.arange(0.2, 0, -0.05)

for lamb in Lambdas:
    print('Trying lambda:', np.round(lamb, 4))
    m_gamma0 = np.zeros(c.shape)
    m_gamma0[supp_m_gamma] = np.exp(- c[supp_m_gamma] / lamb)

    z1 = np.ones(c.shape)
    z2 = np.ones(c.shape)

    m_gamma_old = m_gamma0
    for n in range(100000):
        m_gamma = prox_G1(m_gamma_old * z1)
        z1[supp_m_gamma] = z1[supp_m_gamma] * (m_gamma_old[supp_m_gamma] / m_gamma[supp_m_gamma])
        m_gamma_old = m_gamma
        m_gamma = prox_G2(m_gamma_old * z2)
        z2[supp_m_gamma] = z2[supp_m_gamma] * (m_gamma_old[supp_m_gamma] / m_gamma[supp_m_gamma])
        if (np.sum(np.abs(m_gamma - m_gamma_old)) < tolerance):
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
        
    d = d**(1/p)
    print('d = ', round(d, 6), '\n')
    
print('Given marginals:')
print('m_mu:', np.round(m_mu, 6), 'with sum = ', round(np.sum(m_mu), 4))
print('m_nu:', np.round(m_nu, 6), 'with sum = ', round(np.sum(m_nu), 4))

    
    
    