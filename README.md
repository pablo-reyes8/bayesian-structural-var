# Bayesian Structural VAR (BSVAR) Analysis Notebook


![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/bayesian-structural-var)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/bayesian-structural-var)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/bayesian-structural-var)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/bayesian-structural-var)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/bayesian-structural-var?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/bayesian-structural-var?style=social)


A self-contained Jupyter notebook that implements a **Bayesian Structural VAR with agnostic identification** to isolate pure U.S. Fed policy shocks and trace their impact on Colombian macro-financial variables:

- **Variables:** Unemployment, Inflation, Central Bank rate, Exchange rate (TRM), 5-year TES yields  
- **Identification:** Soft priors (truncated-t and asymmetric-t) on structural contemporaneous matrix and shock loadings  
- **Sampling:**  
  - Gibbs draws for reduced-form coefficients (Matrix-Normal) and error variances (Inverse-Gamma)  
  - Metropolis–Hastings for non-conjugate structural blocks, with adaptive scaling during warm-up  

## Contents

1. **Data Preparation**  
   - Load series in levels, logs and HP-filtered cycles  
   - Build lagged regressors and intercept terms  

2. **Model Specification**  
   - Structural form:  
     $$A_0\,y_t = a_0 + \sum_{i=1}^p A_i\,y_{t-i} + C\,\varepsilon_t$$  
   - Reduced form with Minnesota-style conjugate priors for \(B\) and \(\Sigma\)  
   - Agnostic identification via soft-information priors

3. **MCMC Estimation**  
   - Metropolis–Hastings updates for $(A_0,C)$  
   - Gibbs sampling for $B$ and $\Sigma$  
   - Adaptive tuning of proposal covariance to target 20–30% acceptance  

4. **Diagnostics**  
   - Trace plots and histograms with posterior mean markers  
   - Acceptance-rate feedback and autocorrelation checks  

5. **Impulse-Response Analysis**  
   - Construct companion matrices from draws  
   - Compute IRFs for each posterior sample  
   - Plot median response with 68% & 80% credible bands  

6. **FEVD**  
   - Calculate forecast-error variance shares by shock and horizon  
   - Stack-area charts showing evolving contributions  

7. **Utility Functions**  
   - Nearest SPD projection for Hessian-based proposals  
   - Truncated and skewed-t log-densities for soft restrictions  
   - Companion-matrix and IRF builder routines  

---

### How to Use

1. Clone this notebook.  
2. Install required Python packages.  
3. Open in Jupyter and run all cells from top to bottom.  
4. Modify hyperparameters (lag order, prior strengths, horizons) at the top.  
5. Examine the resulting IRFs, FEVDs and convergence diagnostics to draw economic insights.

## Dependencies

```bash
pip install pandas numpy scipy numdifftools matplotlib seaborn
```

References: 

**J. Jacobo, Una introducción a los métodos de máxima entropía y de inferencia bayesiana en econometría**

## Contributing

Contributions are welcome! Please open issues or submit pull requests at  
https://github.com/pablo-reyes8

## License

This project is licensed under the Apache License 2.0.  
