#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D. Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes the order-preserving possibility distribution 
minimizing the pessimistic (when alpha = 1) or optimistic (when alpha = 0) 
Choquet-Wasserstein pseudo-distance with respect to the probability
distribution m_mu, using an entropic regularization with parameter lamb.
The ground metric is assumed to be the absolute value metric.
The code produces images diplaying the initial probability distribution and
the approximated order-preserving possibility distributions.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def CW_minimal_possibility(m_mu, alpha, Sigma, lamb, p):

    min_d = np.infty
    
    min_sigma = None
    
    min_m_nu = None
    
    for sigma in Sigma:
        print('Checking:', sigma)
        c = np.zeros((len(X),len(X)))
        (m, n) = c.shape
        for i in range(m):
            for j in range(n):
                max_c = -np.infty
                min_c = np.infty
                for x in F_X[i]:
                    for y in F_Y[j]:
                        temp_c = (np.abs(X[x] - X[sigma[y]]))**p
                        if temp_c > max_c:
                            max_c = temp_c
                        if temp_c < min_c:
                            min_c = temp_c
                c[i, j] = alpha * min_c + (1 - alpha) * max_c
        print(c)
                        
    
        print('m_mu:', m_mu, 'with sum = ', sum(m_mu))
        
        def prox_G1(theta):
            m_gamma = np.zeros_like(theta)
            (m, n) = theta.shape
            for i in range(m):
                for j in range(n):
                    m_gamma[i, j] = m_mu[i] * theta[i, j] / sum(theta[i, :])
            return m_gamma
        
        
        m_gamma0 = np.exp(- c / lamb)
        
        z = np.ones(c.shape)
    
        print ('lamb:', round(lamb, 4))
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
        
        # Keeps track
        d = d**(1 / p)
        distances.append(d)
        nus.append(np.round(np.sum(m_gamma, axis=0), 6))
        sigma_list.append(list(sigma))
        
        if d <= min_d:
            min_d = d
            min_sigma = sigma
            min_m_nu = np.round(np.sum(m_gamma, axis=0), 6)
            
    print('*** MIN DISTANCE ***')
    print('min_d:', round(min_d, 4))
    print('min_sigma:', min_sigma)
    print('min_m_nu:', min_m_nu)
    
    # Extract the possibilitu distributin
    nu_bar = np.zeros_like(min_m_nu)
    for i in range(n):
        for j in range(i, n):
            nu_bar[min_sigma[i]] += min_m_nu[min_sigma[j]]
    
    return d, np.round(nu_bar,6)



# Order of the Choquet-Wasserstein pseudo-distances
p = 1

# Range of the random variables
n = 31
X = np.arange(n)


# Build the sets of focal elements
F_X = []
F_Y = []

for i in range(n):
    F_X.append({i})
    s = []
    for j in range(i + 1):
        s.append(j)
    F_Y.append(set(s))
    
print('F_X:', F_X)
print('F_Y:', F_Y)


###############################################################################
# Probability distribution
# (Bin(30, 0.25)
q = 0.25
m_mu = stats.binom.pmf(X, n=n, p=q)
distances = []
nus = []
sigma_list = []
sigma = np.argsort(m_mu)[::-1]
Sigma = [sigma] 

fig, ax = plt.subplots(1, 1)
plt.title('Probability distribution of $\mu$ ($X \sim$ Bin(30, ' + str(q) + '))')
ax.plot(X, m_mu, 'ro', ms=5, mec='r')
ax.vlines(X, 0, m_mu, colors='r', lw=4)
plt.savefig('mu_' + str(q) +'.png', dpi=300)
plt.show()
    
d, nu_bar = CW_minimal_possibility(m_mu, 1, Sigma, 0.05, 1)   
print('PESS d =', d)
print('nu_bar:', np.round(nu_bar,4))
fig, ax = plt.subplots(1, 1)
plt.title(r'$d_{\mathcal{CW}}^' + str(p) + r'$-minimal possibility distribution of $\overline{\nu}$')
ax.plot(X, nu_bar, 'go', ms=5, mec='green')
ax.vlines(X, 0, nu_bar, colors='green', lw=4)
plt.savefig('PESS_nu_' + str(q) +'.png', dpi=300)
plt.show()

d, nu_bar = CW_minimal_possibility(m_mu, 0, Sigma, 0.05, 1)   
print('PESS d =', d)
print('nu_bar:', np.round(nu_bar,4))
fig, ax = plt.subplots(1, 1)
plt.title(r'$\overline{d}_{\mathcal{CW}}^' + str(p) + r'$-minimal possibility distribution of $\overline{\nu}$')
ax.plot(X, nu_bar, 'bo', ms=5, mec='blue')
ax.vlines(X, 0, nu_bar, colors='blue', lw=4)
plt.savefig('OPT_nu_' + str(q) +'.png', dpi=300)
plt.show()

###############################################################################
# Probability distribution
# (Bin(30, 0.75)
q = 0.75
m_mu = stats.binom.pmf(X, n=n, p=q)
distances = []
nus = []
sigma_list = []
sigma = np.argsort(m_mu)[::-1]
Sigma = [sigma] 

fig, ax = plt.subplots(1, 1)
plt.title('Probability distribution of $\mu$ ($X \sim$ Bin(30, ' + str(q) + '))')
ax.plot(X, m_mu, 'ro', ms=5, mec='r')
ax.vlines(X, 0, m_mu, colors='r', lw=4)
plt.savefig('mu_' + str(q) +'.png', dpi=300)
plt.show()
    
d, nu_bar = CW_minimal_possibility(m_mu, 1, Sigma, 0.05, 1)   
print('PESS d =', d)
print('nu_bar:', np.round(nu_bar,4))
fig, ax = plt.subplots(1, 1)
plt.title(r'$d_{\mathcal{CW}}^' + str(p) + r'$-minimal possibility distribution of $\overline{\nu}$')
ax.plot(X, nu_bar, 'go', ms=5, mec='green')
ax.vlines(X, 0, nu_bar, colors='green', lw=4)
plt.savefig('PESS_nu_' + str(q) +'.png', dpi=300)
plt.show()

d, nu_bar = CW_minimal_possibility(m_mu, 0, Sigma, 0.05, 1)   
print('PESS d =', d)
print('nu_bar:', np.round(nu_bar,4))
fig, ax = plt.subplots(1, 1)
plt.title(r'$\overline{d}_{\mathcal{CW}}^' + str(p) + r'$-minimal possibility distribution of $\overline{\nu}$')
ax.plot(X, nu_bar, 'bo', ms=5, mec='blue')
ax.vlines(X, 0, nu_bar, colors='blue', lw=4)
plt.savefig('OPT_nu_' + str(q) +'.png', dpi=300)
plt.show()

###############################################################################
# Probability distribution
# (1/2) * (Bin(30, 0.25) + Bin(30, 0.75))
m_mu = 0.5 * stats.binom.pmf(X, n=n, p=0.25) + 0.5 * stats.binom.pmf(X, n=n, p=0.75)
distances = []
nus = []
sigma_list = []
sigma = np.argsort(m_mu)[::-1]
Sigma = [sigma] 

fig, ax = plt.subplots(1, 1)
plt.title(r'Probability distribution of $\mu$ ($X \sim \frac{1}{2} (\mathrm{Bin}(30,0.25) + \mathrm{Bin}(30,0.75))$)')
ax.plot(X, m_mu, 'ro', ms=5, mec='r')
ax.vlines(X, 0, m_mu, colors='r', lw=4)
plt.savefig('mu_mix.png', dpi=300)
plt.show()
    
d, nu_bar = CW_minimal_possibility(m_mu, 1, Sigma, 0.05, 1)   
print('PESS d =', d)
print('nu_bar:', np.round(nu_bar,4))
fig, ax = plt.subplots(1, 1)
plt.title(r'$d_{\mathcal{CW}}^' + str(p) + r'$-minimal possibility distribution of $\overline{\nu}$')
ax.plot(X, nu_bar, 'go', ms=5, mec='green')
ax.vlines(X, 0, nu_bar, colors='green', lw=4)
plt.savefig('PESS_nu_mix.png', dpi=300)
plt.show()

d, nu_bar = CW_minimal_possibility(m_mu, 0, Sigma, 0.05, 1)   
print('PESS d =', d)
print('nu_bar:', np.round(nu_bar,4))
fig, ax = plt.subplots(1, 1)
plt.title(r'$\overline{d}_{\mathcal{CW}}^' + str(p) + r'$-minimal possibility distribution of $\overline{\nu}$')
ax.plot(X, nu_bar, 'bo', ms=5, mec='blue')
ax.vlines(X, 0, nu_bar, colors='blue', lw=4)
plt.savefig('OPT_nu_mix.png', dpi=300)
plt.show()


