import numpy as np 
from identification import *
from mcmc import *
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch


setup_var_minnesota_globals(YY, m=3, lambda0=0.1, lambda1=0.1, lambda3=10.0, kappa=2.0)

R1 = 80000 # Number of the posterior samples for the IRFS
HOR = 26 # Forecast period 

def bigAS(Posterior_A0 , R1):
    """
    Constructs an array of structural A₀ matrices from posterior draws.

    What it does:
    1. Takes `R1` posterior draws of the flattened A₀ parameters.
    2. For each draw, calls `A0_Mtx(...)` to reconstruct the n×n contemporaneous
       matrix A₀.
    3. Stacks these matrices into a 3-D array of shape (R1, n, n), ready for IRF computation.
    Returns
    -------
    As : ndarray, shape (R1, n, n)
        Stack of reconstructed A₀ matrices, one per posterior draw.
    """
    As = np.zeros((R1,n,n))
    for i in range(0,R1):
        As[i:,:,] = A0_Mtx(Posterior_A0[:,i])
    return As

def bigPhsi(Posterior_B , R1):
    """
    Builds companion‐form transition matrices from posterior draws of reduced‐form coefficients.

    What it does:
    1. Takes `R1` draws of the reduced‐form coefficient matrix B (shape R1×k×n).
    2. For each draw:
       a. Transposes B to get Xi of shape (n×m, n) if m = k/n.
       b. Constructs the companion‐matrix top block by dropping the last n columns of Xi.
       c. Builds the lower block as [I_{n(m−1)} | 0_{n(m−1)×n}] to shift lags.
       d. Vertically stacks these two blocks into an (n·m × n·m) matrix.
    3. Stacks all R1 companion matrices into an array of shape (R1, n·m, n·m),
       ready for IRF recursion.
    Returns
    -------
    phis : ndarray, shape (R1, n*m, n*m)
        Array of companion matrices, one per posterior draw.
    """
    phis = np.zeros((R1 , n*m , n*m))
    for i in range(0,R1):
        Xi= Posterior_B[i,:,:].T
        horizontal = np.hstack((np.eye(n*(m-1)) , np.zeros((n*(m-1),n))))
        phis[i:,:,] = np.vstack((Xi[:, :-1] , horizontal))
    return phis

@decorador_2
def IRF(Posterior_A0, Posterior_B, Horiz):
    """
    Compute impulse‐response functions (IRFs) for each posterior draw of a VAR.

    What it does:
    1. Constructs the companion‐form transition matrices (`phiss`) from `Posterior_B`.
    2. Reconstructs and inverts each structural matrix A₀ from `Posterior_A0` to get initial impact `Boi`.
    3. Uses a selector matrix `J` to extract the first block of the companion form.
    4. Recursively propagates the shock through `Horiz` periods:
       - At h=0, IRF is simply A₀⁻¹.
       - For h>0, IRF[h] = J · (companion^h) · J.T · A₀⁻¹.
    5. Stacks all IRFs into an array of shape (R1, n, n*(Horiz+1)), where R1 is the number
       of posterior draws.

    Returns
    -------
    GIR : ndarray, shape (R1, n, n*(Horiz+1))
        Array of impulse‐response functions. For each draw i:
          GIR[i, :, 0:n]        = A₀⁻¹ (impact at h=0),
          GIR[i, :, h*n:(h+1)*n] = IRF at horizon h (for h=1…Horiz).
    """
    
    J = np.hstack((np.eye(n), np.zeros((n, n * (m - 1)))))
    As = bigAS(Posterior_A0, R1)
    Boi = np.zeros((n, n))
    phiss = bigPhsi(Posterior_B, R1)
    GIR = np.zeros((R1, n, n * (Horiz + 1))) 
    
    for i in range(R1):
        bigAis = phiss[i, :, :]  # Companion matrix for draw i
        Boi = inv(As[i, :, :]) # Initial impact (h=0) 
        GIR[i, 0:n ,0:n] = Boi 

        # Recursive propagation for h = 1 … Horiz
        for k in range(Horiz):
            GIR[i, :, (k + 1) * n: (k + 2) * n] = J @ bigAis @ J.T @ Boi
            bigAis = bigAis @ phiss[i, :, :]  # advance the companion power
    
    return GIR



def FEVD(IR , n , s):
    """
    Compute the Forecast Error Variance Decomposition (FEVD) from IRF matrices.

    Given the impulse response function (IR) array for a single Monte Carlo draw,
    this function calculates, for each horizon up to s-1, the proportion of the
    h-step ahead forecast error variance of each variable that is attributable
    to shocks in each variable.

    Returns
    -------
    Ws : ndarray, shape (n, n*s)
        FEVD proportions. For each horizon j (0 ≤ j < s), the block
        Ws[:, j*n:(j+1)*n] contains an n×n matrix where entry (i, k)
        is the fraction of variable i’s forecast error variance at
        horizon j explained by a unit shock to variable k.
    """

    Ms = np.zeros((n , n*s))
    Fs = np.zeros((n , n*s))
    Ws = np.zeros((n , n*s))

    Mss = (IR[0:n , 0:n]) * (IR[0:n , 0:n])
    Fss = (IR[0:n , 0:n]) @ (IR[0:n , 0:n]).T * np.eye(n)

    for j in range(1 , s-1):
        Mss = Mss + (IR[0:n , j*n:(j+1)*n]) * IR[0:n , j*n:(j+1)*n]
        Ms[:,j*n:(j+1)*n] = Mss
        Fss = Fss + (IR[0:n,j * n:(j+1) * n]) @ (IR[0:n,j * n:(j+1) * n]).T * np.eye(n)
        Fs[:,j * n:(j + 1) * n] = Fss
        Ws[:,j * n:(j + 1) * n] = np.linalg.solve(Fss, Mss)
    return Ws



def plot_irfs_from_mcif(MCIF: np.ndarray,
    HOR: int,var_names=None,
    shock_idx: int = 0,title: str = "Respuesta a shock",
    suptitle: str | None = None,
    quantiles=(0.05, 0.16, 0.50, 0.84, 0.95),ncols: int = 3,
    start_at_h1: bool = True,show_legend: bool = True):

    """
    Grafica IRFs con bandas de credibilidad a partir de simulaciones (MCIF).

    Parámetros
    ----------
    MCIF : array, shape (S, n, n*HOR)
        S = número de simulaciones Monte Carlo.
        n = número de variables.
        El tercer eje apila los n impulsos por cada horizonte: [h=0,...,HOR-1],
        con bloque de tamaño n por horizonte, i.e. índice = j + h*n (j=0..n-1).
    HOR : int
        Número de horizontes (incluyendo h=0 si corresponde).
    var_names : list[str] | None
        Nombres de variables para títulos. Si None, se usan genéricos v0..vn-1.
    shock_idx : int
        Índice de la variable que recibe el shock (fila en MCIF[:, i, ...]).
    title : str
        Título principal del conjunto (usado si suptitle es None).
    suptitle : str | None
        Título superior para fig.suptitle. Si None, usa `title`.
    quantiles : tuple[float,...]
        Cuantiles a calcular. Por defecto (0.05, 0.16, 0.50, 0.84, 0.95).
    ncols : int
        Número de columnas en el grid de subplots.
    start_at_h1 : bool
        Si True, el eje x comienza en 1 y se oculta h=0 (útil si h=0 es normalizado).
        Si False, plotea desde h=0 con x = [0, 1, ..., HOR-1].
    show_legend : bool
        Muestra la leyenda en el primer subplot.

    Notas
    -----
    - Asume que las IRF para el horizonte h y variable j se encuentran en la
      columna `j + h*n` del último eje de MCIF.
    """
    
    if MCIF.ndim != 3:
        raise ValueError("MCIF debe ser 3D con shape (S, n, n*HOR).")
    S, n, last = MCIF.shape
    if last != n * HOR:
        raise ValueError(f"El último eje de MCIF debe ser n*HOR (= {n*HOR}), recibido {last}.")
    if var_names is None:
        var_names = [f"y{j}" for j in range(n)]
    if len(var_names) != n:
        raise ValueError("len(var_names) debe coincidir con n.")

    # Prealocar cuantiles
    q_dict = {q: np.zeros((HOR, n)) for q in quantiles}


    # Calcular cuantiles por variable y horizonte desde MCIF
    # Índice correcto: j + h*n (h inicia en 0)
    for j in range(n):
        for h in range(HOR):
            col = j + h * n
            series = MCIF[:, shock_idx, col]  
            for q in quantiles:
                q_dict[q][h, j] = np.quantile(series, q)

    # Extraer bandas típicas (usando disponibles en quantiles)
    q05 = q_dict[min(quantiles, key=lambda x: abs(x-0.05))] if len(quantiles) else None
    q16 = q_dict[min(quantiles, key=lambda x: abs(x-0.16))] if len(quantiles) else None
    q50 = q_dict[min(quantiles, key=lambda x: abs(x-0.50))] if len(quantiles) else None
    q84 = q_dict[min(quantiles, key=lambda x: abs(x-0.84))] if len(quantiles) else None
    q95 = q_dict[min(quantiles, key=lambda x: abs(x-0.95))] if len(quantiles) else None

    # Eje x y rango 
    if start_at_h1:
        x = np.arange(1, HOR)    
        sl = slice(1, None)       
    else:
        x = np.arange(0, HOR)    
        sl = slice(None)

    # Layout
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    if suptitle is None:
        suptitle = title
    fig.suptitle(suptitle, fontsize=17, fontweight='bold', y=0.98)

    # Bucle de gráficos por variable
    for j in range(n):
        row, col = divmod(j, ncols)
        ax = axes[row, col]

        # Curva central
        y = q50[sl, j] if q50 is not None else None

        # Bandas 90–10 y 84–16, si existen
        low90 = q05[sl, j] if q05 is not None else None
        up90  = q95[sl, j] if q95 is not None else None
        low84 = q16[sl, j] if q16 is not None else None
        up84  = q84[sl, j] if q84 is not None else None

        # Sombreado (80% y 68% aprox)
        if (low90 is not None) and (up90 is not None):
            ax.fill_between(x, low90, up90, color='lightgray', label='80 % intervalo')
        if (low84 is not None) and (up84 is not None):
            ax.fill_between(x, low84, up84, color='gray', label='68 % intervalo')

        if y is not None:
            ax.plot(x, y, color='black', linewidth=2)

        ax.set_title(var_names[j], fontsize=11, fontweight='bold')
        ax.axhline(0.0, lw=0.8, color='k', alpha=0.5)

        if show_legend and (j == 0):
            ax.legend(loc="upper right")

    for k in range(n, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    for c in range(min(ncols, axes.shape[1])):
        axes[-1, c].set_xlabel("Horizonte")

    plt.tight_layout()
    plt.show()


def plot_fevd_stack(ws: np.ndarray,
    var_names=None,shock_names=None,
    drop_h0: bool = True,      
    drop_last: bool = True,    ncols: int = 3,
    cmap_name: str = "tab20",suptitle: str = "FEVD – Descomposición de Varianza por Shock"):

    """
    Dibuja FEVD como áreas apiladas (stackplot) por variable y horizonte.
    
    Parámetros
    ----------
    ws : np.ndarray
        FEVD con shape (n, n*s) o (n, s, n).
        - n: número de variables y shocks.
        - s: número de horizontes.
        Convención típica: para (n, n*s), cada fila i concatena [h=0..s-1] con bloques n.
    var_names : list[str] | None
        Nombres de las variables (longitud n).
    shock_names : list[str] | None
        Nombres de los shocks (longitud n).
    drop_h0 : bool
        Si True, elimina el horizonte 0 al graficar (suele ser trivial/normalizado).
    drop_last : bool
        Si True, elimina el último horizonte (útil si el arreglo incluye un cierre).
    ncols : int
        Número de columnas del grid de subplots.
    cmap_name : str
        Nombre del colormap para los shocks.
    suptitle : str
        Título superior de la figura.
    """

    if ws.ndim == 2:
        n = ws.shape[0]
        s = ws.shape[1] // n
        if ws.shape[1] != n * s:
            raise ValueError("Para ws 2D, se espera shape (n, n*s).")
        
        fevd = ws.reshape(n, s, n)
    elif ws.ndim == 3:
        n, s, n2 = ws.shape
        if n != n2:
            raise ValueError("Para ws 3D, se espera shape (n, s, n).")
        fevd = ws
    else:
        raise ValueError("ws debe tener 2 o 3 dimensiones.")

    # Nombres
    if var_names is None:
        var_names = [f"y{j}" for j in range(n)]
    if shock_names is None:
        shock_names = [f"ε{j}" for j in range(n)]
    if len(var_names) != n or len(shock_names) != n:
        raise ValueError("len(var_names) y len(shock_names) deben ser n.")

    # Recorte de horizontes
    horiz_full = np.arange(s)           
    h_start = 1 if drop_h0 else 0
    h_end = -1 if drop_last and s > 1 else s
    horiz = horiz_full[h_start:h_end]     
    fevd_cut = fevd[:, h_start:h_end, :]  

    # Layout
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(int(i * (cmap.N / max(n, 1)))) for i in range(n)]

    # Graficar cada variable i
    for i in range(n):
        ax = axes[i]
        data = fevd_cut[i].T     # (n_shocks, s_used)
        ax.stackplot(horiz, *data, colors=colors, linewidth=0)
        ax.set_title(var_names[i], fontsize=12, weight='semibold')
        ax.set_xlim(horiz[0], horiz[-1])
        ax.set_ylim(0, 1)
        ax.axhline(1.0, lw=0.6, color="k", alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Participación FEVD")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Horizonte")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [Patch(facecolor=colors[k]) for k in range(n)]
    fig.legend(legend_handles, shock_names, loc='lower center', ncol=min(n, 6),
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(suptitle, fontsize=16, weight='bold', y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.90)
    plt.show()

