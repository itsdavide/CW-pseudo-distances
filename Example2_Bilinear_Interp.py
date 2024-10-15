#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D. Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes two (non-extremal) copulas giving rise to a fixed 
joint Mobius inverse via bilinear interpolation. The graph and the contour 
lines of each copula are saved in images.
"""

import numpy as np
import matplotlib.pyplot as plt


NORM = 20

M_mu = np.array([0, 10, 12, 16, 20]) / NORM

M_nu = np.array([0, 8, 12, 20]) / NORM


q = 1
C_q = np.array([
    [0, 0, 0, 0],
    [0, 4, 6, 10],
    [0, 4, 8, 12],
    [0, 8, 12, 16],
    [0, 8, 12, 20]
    ]) / NORM

"""
q = 2
C_q = np.array([
    [0, 0, 0, 0],
    [0, 4, 6, 10],
    [0, 4, 8, 12],
    [0, 4, 8, 20],
    [0, 8, 12, 20]
    ]) / NORM
"""

def alpha(x):
    lower_x = max(M_mu[M_mu <= x])
    upper_x = min(M_mu[M_mu >= x])
    if lower_x < upper_x:
        return (x - lower_x) / (upper_x - lower_x)
    return 1

def beta(y):
    lower_y = max(M_nu[M_nu <= y])
    upper_y = min(M_nu[M_nu >= y])
    if lower_y < upper_y:
        return (y - lower_y) / (upper_y - lower_y)
    return 1

def C(x,y):
    i_lower_x = np.where(M_mu == max(M_mu[M_mu <= x]))[0][0]
    i_upper_x = np.where(M_mu == min(M_mu[M_mu >= x]))[0][0]
    j_lower_y = np.where(M_nu == max(M_nu[M_nu <= y]))[0][0]
    j_upper_y = np.where(M_nu == min(M_nu[M_nu >= y]))[0][0]
    
    val = (1 - alpha(x)) * (1 - beta(y)) * C_q[i_lower_x, j_lower_y] + (1 - alpha(x)) * beta(y) * C_q[i_lower_x, j_upper_y] + alpha(x) * (1 - beta(y)) * C_q[i_upper_x, j_lower_y] + alpha(x) * beta(y) * C_q[i_upper_x, j_upper_y]
    return val
    

# Vectorize the function
vT = np.vectorize(C)
    

# Make data.
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)


X, Y = np.meshgrid(x, y)
zs = np.array(vT(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)


levels = np.arange(0, 1, 0.02)


# Surface plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 250)

ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap='viridis', edgecolors='k', lw=0.6)

plt.title(r'Graph of $\mathsf{C}_' + str(q) + '$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\mathsf{C}_' + str(q) + '(x,y)$', rotation=90)

ax.view_init(30, 250)
plt.savefig('C' + str(q) + '.png', dpi=300)


# Contour lines plot
plt.clf()
fig = plt.figure(figsize=(5,5))
plt.title(r'Contour lines of $\mathsf{C}_' + str(q) + '$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.contour(X, Y, Z, levels)
plt.savefig('C' + str(q) + '-cl.png', dpi=300)



    