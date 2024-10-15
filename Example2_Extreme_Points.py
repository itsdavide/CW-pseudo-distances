#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference code for the paper:
    
S. Lorenzini, D.Petturiti, B. Vantaggi.
Choquet-Wasserstein pseudo-distances via optimal transport under partially 
specified marginal probabilities. 2024.

CONTENT: The code computes the extreme points of the set of joint belief
functions with given marginals, both with a vertex enumeration algorithm and
by means of extreme copulas (minimum and Lukasiewicz) by varying the
permutations of the marginal underlying spaces.
"""

from ExtremePointsLP import ExtremePointsLP
from ExtremePointsC import ExtremePointsC
import numpy as np
from sys import exit


# PROBLEM
NORM = 20
m_mu = np.array([10, 2, 4, 4])
m_nu = np.array([8, 4, 8])

print('Check const.:', m_mu.sum() == NORM, 'and', m_nu.sum() == NORM)
if not m_mu.sum() == NORM or not m_nu.sum() == NORM:
    exit(1)


print('\nExtreme points with Linear Programming (LP):')
verticesLP = ExtremePointsLP(NORM, m_mu, m_nu)
verticesLP = np.array(verticesLP)
verticesLP = np.abs(np.round(verticesLP, 0))
verticesLP = verticesLP.tolist()

number = 0
for v in verticesLP:
    number += 1
    print ("Extreme point n.: ", number)
    for i in range(len(v)):
        print(v[i], end="  ")
    print()
    
    

print('\nExtreme points with extreme Copulas (C):')
verticesC = ExtremePointsC(NORM, m_mu, m_nu)
verticesC = np.abs(np.round(verticesC, 0))
verticesC = verticesC.tolist()

number = 0
for v in verticesC:
    number += 1
    print ("Extreme point n.: ", number)
    for i in range(len(v)):
        print(v[i], end="  ")
    print()



# Check extra extreme points
i = 0
tot = 0
print('\n\nExtra extreme points')
for v in verticesLP:
    i += 1
    if not v in verticesC:
        tot += 1
        print('Not present extreme point: ', i)
        print(v)
        
print('\n# extreme points LP:', len(verticesLP))   
print('\n# extreme points C:', len(verticesC))      
print('\n# of missing extreme points:', tot, '\n')

