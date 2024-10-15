# CW-pseudo-distances

Reference code for the paper:
    
S. Lorenzini, D.Petturiti, B. Vantaggi.
_Choquet-Wasserstein pseudo-distances via optimal transport under partially specified marginal probabilities_. 2024.

# Requirements
The code requires the *pypoman* library available at: https://pypi.org/project/pypoman/.

# File inventory
_Example2_Bilinear_Interp.py_: The code computes two (non-extremal) caopulas giving rise to a fixed 
joint Mobius inverse via bilinear interpolations. The graph and the contour 
lines of each copula are saved in images.

_Example2_Extreme_Points.py_: The code computes the extreme points of the set of joint belief
functions with given marginals, both with a vertex enumeration algorithm and
by means of extreme copulas (minimum and Lukasiewicz) by varying the
permutations of the marginal underlying spaces.

_Example7_CW_Pseudo_Dist.py_: The code computes the entropic regularized Choquet-Wasserstein 
pseudo-distances between two belief functions on the same metric space
with ground absolute value metric, for different values of the regularization
parameter. The computation is done with a modification of Dykstra's algortithm.

_Example8_Add2_Eps_Approx.py_: The code computes the pessimistic (alpha = 1) and optimistic 
(alpha = 0) Choquet-Wasserstein approximation of a 2-additive belief function
with an epsilon-contamination model, using an entropic regularization with 
parameter lamb. The ground metric is assumed to be the discrete metric.
In this case the order p in [1,+infinity) is irrelevant.

_Example9_Prob_Poss_Approx.py_: The code computes the order-preserving possibility distribution 
minimizing the pessimistic (when alpha = 1) or optimistic (when alpha = 0) 
Choquet-Wasserstein pseudo-distance with respect to the probability
distribution m_mu, using an entropic regularization with parameter lamb.
The ground metric is assumed to be the absolute value metric.
The code produces images diplaying the initial probability distribution and
the approximated order-preserving possibility distributions.

_ExtremePointsC.py_: The code computes the extreme points of the set of joint belief
functions with given marginals by means of extreme copulas (minimum and 
Lukasiewicz) by varying the permutations of the marginal underlying spaces.
The code shows that some extreme points cannot be obtained in this way.

_ExtremePointsLP.py_: The code computes the extreme points of the set of joint belief
functions with given marginals, with a vertex enumeration algorithm.
Extreme points are found referring to the polytope given by A*x <= b, where we
add the non-negativity constraint: -x <= 0.

