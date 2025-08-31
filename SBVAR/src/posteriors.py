import numpy as np 
from identification import *


def posterior_B(theta1 , theta2 , Y , X , M , Phi0):
    """
    Draws a sample from the posterior of the reduced‐form coefficient matrix B 
    under a Normal–Minnesota prior and Gaussian likelihood.

    What it does:
    1. Reconstructs the structural matrix A₀ from `theta1`.
    2. Builds D = diag(sigma_i²) using `theta2` (vector of error variances).
    3. Computes the reduced‐form covariance Ω = A₀⁻¹ · D · (A₀⁻¹)'.
    4. Forms the posterior covariance of B: (M⁻¹ + X'X)⁻¹.
    5. Computes the posterior mean: 
       M_si · (M⁻¹·Phi0 + X'·Y).
    6. Draws one sample from the MatrixNormal distribution:
       B ∼ MN(mean=B_s, rowcov=M_si, colcov=Ω).

    Parameters
    ----------  
    theta1:  Vector encoding structural parameters for A₀.
    theta2:  Diagonal variances (σ₁²,…,σₙ²) for the reduced‐form errors.
    Y: T×k matrix of endogenous observations.
    M:  Prior row‐covariance matrix for B.
    Phi0: Prior mean matrix for B.

    """
    A = A0_Mtx(theta1)
    D = np.diag(theta2)
    A0i = inv(A)
    Omega = np.matrix(A0i @ D @ A0i.T)
    Omega = Omega.getH()
    Omega = np.asarray(Omega)

    # posterior covariance of B
    M_si = np.matrix(inv(inv(M)+X.T@X))
    M_si = M_si.getH()
    M_si = np.asarray(M_si)
    
    # posterior media of B
    B_s = M_si @ (inv(M) @ Phi0 + X.T @ Y)

    # sampling from the normal matrix distribution
    return matrix_normal.rvs(B_s , M_si , Omega)


# Function of the posterior of Sigma 
def Posterior_D(theta1 , Y , X , M , phi0 , S0 , omega , S2 , n , T):
    """
    Samples the diagonal error‐variance vector (σ₁²,…,σₙ²) from its conditional
    Inverse‐Gamma posterior, given structural parameters and data.

    What it does:
    1. Reconstructs A₀ from `theta1`.
    2. Scales the pilot covariance S0 by (κ−1) and transforms via A₀:
       Sz0 = (κ−1)*A₀·S0·A₀'.
    3. Sets the posterior shape κ₁ = (κ + T)/2.
    4. For each i, computes the scale ts_i = diag(Sz0)[i] + 0.5*(diag(Sz0)[i] + diag(Sz0)[i])
       and draws σᵢ² ∼ Inverse‐Gamma(κ₁, scale=ts_i).
    """
    
    kappa = 2
    A = A0_Mtx(theta1)
    Sz0 = np.matrix((kappa-1) * A @ S0 @ A.T)
    Sz0 = Sz0.getH()
    Sz0 = np.asarray(Sz0)
    d = np.zeros(n)
    kappa1 = (kappa + T) / 2
    t = np.diag(Sz0) 
    for i in range(n):
        ts = (t[i] + 0.5*(t[i]+t[i]))
        d[i] = invgamma.rvs(kappa1 ,scale=ts)
    return d