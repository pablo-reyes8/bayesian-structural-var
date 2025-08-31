import numpy as np
from scipy.special import gammaln , multigammaln ,gamma
from scipy.stats import t ,norm , matrix_normal , invgamma , beta
from numpy.linalg import inv


def setup_var_minnesota_globals(
    YY: np.ndarray,
    m: int = 3,
    lambda0: float = 0.1,
    lambda1: float = 0.1,
    lambda3: float = 10.0,
    kappa: float = 2.0,):


    """
    Construye X, Y, estimadores ML (PHI, OMEGA) y la Minnesota prior (M, phi0, S0)
    para un VAR con identificación agnóstica. 
    Deja las variables clave en el ambiente global (globals()) y retorna un dict.

    Parámetros
    ----------
    YY : (T_full, n) matriz con las series (sin media ni dummies aquí)
    m  : rezagos del VAR
    lambda0, lambda1, lambda3 : hiperparámetros Minnesota
    kappa : hiperparámetro extra (guardado como parte del estado)

    Variables dejadas en el ambiente global
    ---------------------------------------
    n, m, k, Y, T, X
    e, S0
    eta, v1, v2, v3, M, phi0
    PHI, OMEGA, S2
    lambda0, lambda1, lambda3, kappa
    """
    import numpy as np
from numpy.linalg import inv

def setup_var_minnesota_globals(
    YY: np.ndarray,
    m: int = 3,
    lambda0: float = 0.1,
    lambda1: float = 0.1,
    lambda3: float = 10.0,
    kappa: float = 2.0,
):
    """
    Prepara X, Y, estimadores ML (PHI, OMEGA) y la Minnesota prior (M, phi0, S0)
    para un VAR con m rezagos. No retorna nada: deja las variables en el ambiente
    global del cuaderno (globals()).
    """
    if YY.ndim != 2:
        raise ValueError("YY debe ser 2D (T x n).")
    if m < 1:
        raise ValueError("m (rezagos) debe ser >= 1.")
    if YY.shape[0] <= m:
        raise ValueError("T_full debe ser > m.")

    n = YY.shape[1]
    k = n * m + 1  # parámetros por ecuación (m rezagos de n vars + intercepto)

    # Construcción de Y y X
    Y = YY[m:, :]
    T = Y.shape[0]
    X = np.ones((T, k), dtype=float)
    for i in range(m):
        X[:, (i * n):(i + 1) * n] = YY[m - i - 1:-i - 1, :]

    # Residuos univariados por variable i (solo rezagos propios) para S0
    e = np.zeros((T, n), dtype=float)
    for i in range(n):
        Xi = X[:, i::n][:, :m]      # m rezagos de la variable i
        Yi = Y[:, i]
        beta_i = np.linalg.solve(Xi.T @ Xi, Xi.T @ Yi)
        e[:, i] = Yi - Xi @ beta_i

    S00 = np.zeros(n, dtype=float)
    for i in range(n):
        S00[i] = (e[:, i].T @ e[:, i]) / T
    S0 = np.diag(S00)

    # Minnesota prior
    eta = 0.75 * np.hstack((np.eye(n), np.zeros((n, n * (m - 1) + 1))))
    v1 = np.array([1.0 / (i ** (2 * lambda1)) for i in range(1, m + 1)], dtype=float)
    # Precisión por variable (usar recip de la diag de S0 porque S0 es diagonal)
    v2 = np.diag(1.0 / np.diag(S0))
    v3 = (lambda0 ** 2) * np.append(np.kron(v1, v2), lambda3 ** 2)
    M = np.diag(v3)
    phi0 = eta.T  # (k x n)

    # Estimadores ML
    XtX = X.T @ X
    XtX_inv = inv(XtX)
    PHI = XtX_inv @ (X.T @ Y)               # (k x n)
    resid = Y - X @ PHI
    OMEGA = resid.T @ resid                 # (n x n)

    # Distancia prior
    S2 = (PHI - phi0).T @ inv(M + XtX_inv) @ (PHI - phi0)

    # Exportar todo a globals()
    globals().update({
        "n": n, "m": m, "k": k, "Y": Y, "T": T, "X": X,
        "e": e, "S0": S0,
        "eta": eta, "v1": v1, "v2": v2, "v3": v3, "M": M, "phi0": phi0,
        "PHI": PHI, "OMEGA": OMEGA, "S2": S2,
        "lambda0": lambda0, "lambda1": lambda1, "lambda3": lambda3, "kappa": kappa
    })



setup_var_minnesota_globals(YY, m=3, lambda0=0.1, lambda1=0.1, lambda3=10.0, kappa=2.0)


def A0_Mtx(phi1):
    x1 ,x2,x3,x4= phi1[0], phi1[1] ,phi1[2] ,phi1[3]
    x5 ,x6,x7,x8= phi1[4], phi1[5] ,phi1[6] ,phi1[7]
    x9 ,x10,x12= phi1[8] ,phi1[9] ,phi1[10]
    x13 ,x14,x15,x16= phi1[11], phi1[12] ,phi1[13] ,phi1[14]

    A = np.array([[1 , 0 ,0 ,0, 0,0] , [-x1 , 1, -x2 , 0 , -x3,0] , [-x4, -x5, 1 ,0 ,-x6,0] , [0, 0, -x7 ,1 ,0,-x8],
                  [-x9 ,-x10,0,-x12 ,1 , -x13] , [0 ,-x14,0,-x15 ,-x16 , 1]])
    return A


# Helper function to check truncation intervals
def logsubexp(a, b):
    if a == b:
        return -np.inf  
    elif b > a:
        raise ValueError("The argument 'a' must be greater than 'b' for the result to be real.")
    else:
        return a + np.log1p(-np.exp(b - a))
    

def logpdf_truncated_t(loc, scale, df, a, b, αs):
    """
    Computes the log‐density of a Student’s t distribution truncated to [a, b] at the point αs.

    What it does:
    1. Checks if αs lies outside the truncation interval [a, b]; if so, returns –inf.
    2. Evaluates the log‐PDF of a t(df, loc, scale) at αs.
    3. Computes log‐CDF at the bounds a and b.
    4. Uses log‐subtraction to form the normalizing constant log(F(b)−F(a)).
    5. Subtracts that from the log‐PDF to get the truncated log‐density.

    Returns
    -------
    float
        Log‐density of the truncated t at αs, or –inf if αs is outside [a, b].

    """

    # Check truncation interval compatibility with sign of αs
    if a >= 0 and b > 0:
        if αs < 0:
            return -np.inf
    if a < 0 and b <= 0:
        if αs > 0:
            return -np.inf

    dist = t(df, loc=loc, scale=scale)
    logpdf_value = dist.logpdf(αs)

    log_cdf_a = dist.logcdf(a)
    log_cdf_b = dist.logcdf(b)
    try:
        normalization_factor = logsubexp(log_cdf_b, log_cdf_a)
    except ValueError:
        return -np.inf

    logpdf_truncated = logpdf_value - normalization_factor

    return logpdf_truncated


##### Asymmetric t-distribution for sing restrictions on the IRFs #####

def T_Asimetric(mu_h, sigma_h, v_h, lambda_h, h):
    """
    Computes the log‐density of an asymmetric Student’s t distribution,
    used to impose sign restrictions on impulse‐response functions.

    What it does:
    1. Standardizes the input: x = (h - mu_h) / sigma_h.
    2. Evaluates the symmetric t log‐PDF component:
         - term1: scale adjustment.
         - term2 & term3: normalization constants via gamma functions.
         - term4: heavy‐tail penalty.
    3. Computes an asymmetry factor via the Gaussian CDF:
         norm.logcdf((h * lambda_h) / sigma_h).
    4. Adds both logs to form the final asymmetric log‐density.

    Returns
    -------
    float or ndarray
        Asymmetric t log‐density at each h, combining the symmetric t log‐PDF
        and a skewing factor from the normal CDF.
    """

    x = (h - mu_h) / sigma_h

    term1 = -np.log(sigma_h)
    term2 = gammaln((v_h + 1) / 2) - gammaln(v_h / 2)
    term3 = -0.5 * np.log(v_h * np.pi)
    term4 = -((v_h + 1) / 2) * np.log1p((x ** 2) / v_h)  

    log_phiv = term1 + term2 + term3 + term4
    log_cdf = norm.logcdf((h * lambda_h) / sigma_h)

    log_ph = log_phiv + log_cdf

    return log_ph

def qA(theta1 , kappa, T, OMEGA , S2 , S0 ):
    """
    Computes a positive objective proportional to the negative log-posterior for
    structural parameters under agnostic identification with sign restrictions.

    What it does:
    1. Applies a truncated‐t log‐prior to each entry in `theta1`.
    2. Reconstructs A₀ and inverts it to get H = A₀⁻¹.
    3. Extracts five impulse‐response entries (H[1,0], H[2,0], H[5,0], H[3,0], H[2,0])
       corresponding to policy‐shock effects on TES yields, exchange rate, unemployment,
       inflation, and policy rate.
    4. Evaluates asymmetric‐t densities on those IRF values to impose sign restrictions.
    5. Builds transformed covariance matrices S0ar, Sar, S2ar via A₀·S·A₀'.
    6. Computes a log‐determinant penalty on Sar and a scale penalty combining diagonals
       of S0ar, Sar, and S2ar.
    7. Assembles the final objective = –(sum of log‐priors + log‐IRF penalties)
       – (T/2)*log|Sar| + scale penalty. Returns 1e5 if non‐positive.

    Parameters
    ----------
    theta1 : array-like
        Vector of free parameters defining the contemporaneous matrix A₀.

    Returns
    -------
    float
        A large positive value representing the penalized negative log‐posterior.
    """
    # symmetric truncated‐t log‐priors on all theta entries
    logpdf_values = sum(
        logpdf_truncated_t(0, 8, 3, -np.inf, np.inf, th)
        for th in theta1
    )

    # reconstruct A₀ and its inverse H
    A = A0_Mtx(theta1)
    H = np.linalg.inv(A)

    # 3) extract IRF entries for sign restrictions
    restr1 = H[1, 0]  # TES increase
    restr2 = H[2, 0]  # exchange rate increase
    restr3 = H[5, 0]  # unemployment decrease
    restr4 = H[3, 0]  # inflation increase
    restr5 = H[4, 0]  # policy rate increase

    # asymmetric‐t densities on those IRFs. These sign restrictions come from the IS-LM-BP model simulating an increase in the interest rate of the foreign country 
    tasimetric_values = [
        T_Asimetric(5, 1, 3, 1, restr1),
        T_Asimetric(5, 1, 3, 1, restr2),
        T_Asimetric(-0.245, 1.0, 5, -3, restr3),
        T_Asimetric(5, 1, 3, 1, restr4),
        T_Asimetric(5, 1, 3, 1, restr5),
    ]

    LogpA = logpdf_values + sum(tasimetric_values)

    # build covariance transforms and penalties
    S0ar = (kappa - 1) * A @ S0 @ A.T
    Sar  = A @ OMEGA @ A.T
    S2ar = A @ S2 @ A.T

    t   = np.diag(S0ar)
    t1  = np.diag(Sar)
    t2  = np.diag(S2ar)
    ts  = t + 0.5 * (t1 + t2)

    # log‐determinant penalty
    Determinante = np.log(np.linalg.det(Sar))

    # scale penalty
    log_ts = (kappa + 0.5 * T) * np.log(ts) - kappa * np.log(t)
    final_log_ts = np.sum(log_ts)

    # assemble objective
    Objetivo = -LogpA - (T / 2) * Determinante + final_log_ts

    return 1e5 if Objetivo <= 0 else Objetivo



