import numpy as np 
from identification import *
from posteriors import *
import time


def decorador_2(func):
    def wrapped(*args):
        time1 = time.time()
        rta = func(*args)
        time2 = time.time()
        print('La funcion demoro' , time2-time1)
        return rta 
    return wrapped 

setup_var_minnesota_globals(YY, m=3, lambda0=0.1, lambda1=0.1, lambda3=10.0, kappa=2.0)


# Hastings Metropolis Algorithm
def Metropolis_Hastings(theta , Vmo , rng=None):
    """
    Performs one Metropolis–Hastings update for structural parameters θ.

    What it does:
    1. Proposes a new candidate θ' by drawing from a multivariate normal centered at θ with covariance Vmo.
    2. Computes the acceptance ratio r = exp[–qA(θ') + qA(θ)], where qA evaluates the (negative) log-posterior objective.
    3. Draws u ∼ Uniform(0,1). If u < min(1, r), accepts the proposal (θ ← θ'); otherwise, retains the current θ.
    4. Returns the updated θ.

    Parameters
    ----------
    theta : ndarray, shape (d,)
        Current parameter vector (free entries of A₀ and C).
    Vmo : ndarray, shape (d, d)
        Proposal covariance matrix for the random-walk multivariate normal.

    Returns
    -------
    ndarray, shape (d,)
        The next draw of θ: either the new candidate (if accepted) or the original (if rejected).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- candidate ---
    cand = rng.multivariate_normal(theta, Vmo)

    # --- 2. rate in log (overflow) ---
    log_r = -qA(cand) + qA(theta)        # qA = –log-posterior
    accept = False
    if np.log(rng.random()) < min(0.0, log_r):   # acceptance if log(u) < log_r
        theta, accept = cand, True

    return theta, accept 

# Generate samples of the posterior.
@decorador_2
def MCMC_Metropolis_Hastings(R, theta, Y, X, M, phi0, S0, omega, S2, Vmo,  b=5000):
    """
    Runs a hybrid Gibbs–Metropolis–Hastings sampler with adaptive proposal scaling
    to draw from the joint posterior of structural parameters (A₀, C), error variances Σ,
    and reduced-form coefficients B.

    What it does:
    1. Allocates storage for:
       - theta_s1: structural draws of shape (nalpha, R)
       - theta_s2: variance draws of shape (n, R)
       - theta_s3: B draws of shape (R, k, n)
    2. Initializes theta1 = theta as the starting state.
    3. Repeats for i in 0…R-1:
       a. **Metropolis–Hastings** step: propose and accept/reject a new theta1 using Vmo.
       b. **Gibbs** draw for Σ via Posterior_D.
       c. **Gibbs** draw for B via posterior_B.
       d. Store each draw in the arrays.
       e. During the first `warmup` iterations, adapt Vmo every `adapt_block` steps
          to target an acceptance rate near 25%.
    4. After R iterations, discards the first `b` samples as burn-in and returns the rest.

    Parameters
    ----------
    R      : int
        Total number of MCMC iterations.
    Vmo    : ndarray, shape (d, d)
        Current proposal covariance matrix (must be SPD).
    b      : int, optional
        Number of initial samples to discard as burn-in (default=5000).

    Returns
    -------
    theta_s1[:, b:], theta_s2[:, b:], theta_s3[b:, :, :]
        Posterior samples for (A₀,C), Σ, and B after burn-in.
    """
    # initialize storage arrays
    theta_s1 = np.zeros((nalpha, R))
    theta_s2 = np.zeros((n, R))
    theta_s3 = np.zeros((R, k, n))

    # adaptive tuning variables
    accepted_last100 = 0        # count of acceptances in the last adapt_block
    adapt_block      = 100      # adjust Vmo every this many iterations
    warmup           = b        # number of iterations to adapt Vmo
    rng              = np.random.default_rng()

    theta1 = theta
    for i in range(R):
        # Metropolis–Hastings update for structural parameters
        theta1, accepted = Metropolis_Hastings(theta1, Vmo, rng)
        accepted_last100 += int(accepted)
        # Gibbs draw for error variances Σ
        theta2 = Posterior_D(theta1, Y, X, M, phi0, S0, omega, S2)
        # Gibbs draw for reduced-form coefficients B
        theta3 = posterior_B(theta1, theta2, Y, X, M, phi0)

        theta_s1[:, i] = theta1
        theta_s2[:, i] = theta2
        theta_s3[i, :, :] = theta3
        
        # adapt proposal covariance during warmup
        if i < warmup and (i + 1) % adapt_block == 0:
            rate = accepted_last100 / adapt_block
            if rate < 0.15:
                Vmo *= 0.5    # decrease step size if acceptance is too low
            elif rate > 0.35:
                Vmo *= 2.0    # increase step size if acceptance is too high
            accepted_last100 = 0
            print(f"Iter: {i+1} , accept-rate: {rate:.2f}")

    # discard burn-in samples and return
    return theta_s1[:, b:], theta_s2[:, b:], theta_s3[b:, :, :]
