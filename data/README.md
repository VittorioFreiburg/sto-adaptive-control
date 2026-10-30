# `data/` — precomputed datasets

Every file is written by exactly one MATLAB script (run from the repository root via
`reproduce_all('full')`, or individually) and is read by the figure renderers, so that all
paper figures can be regenerated in under a minute without repeating the Monte Carlo runs
(`reproduce_all('quick')`). All simulations use fixed seeds (`rng(42)`); see Section 8.5 of
the paper for the full protocol.

| File | Written by | Used for | Contents (MATLAB variable names) |
| :--- | :--- | :--- | :--- |
| `data_part1_nominal.mat` | `matlab/persistent_excitation_regularized_sto.m` | Fig. 2, Fig. 5 | Parameter-error trajectories `E_pe` (3 × 40 000; rows = regularization λ ∈ {0, 1e-6, 1e-3}) for persistent excitation σ = 0.18 and `E_low` (σ = 0.10); final particle ensembles `mu_pe`, `mu_low` (N = 10⁵); empirical Gram matrices `H_pe`, `H_low`; final parameters `theta_final_pe`; reference parameters `theta_star`, `lambdas`, `x_ref`, `N_steps`. |
| `data_fast_theorem74.mat` | `matlab/fast_theorem74_convergence_sto.m` | Fig. 3, supplementary diagnostic | δ-sweep error curves `E_reg_sweep`, `E_opt_sweep` (δ ∈ `deltas`), regularized equilibrium `theta_star_lambda` vs. Prop. 7.7 prediction `theta_pred_prop77` and optimum `theta_opt`, Gram spectrum (`H_star`, `lambda_min_H`, `lambda_max_H`), measured decay slopes `slopes_emp`, `slope_ratios`, unregularized/ablation runs (`E_unreg_d05`, `E_opt_lam5`), long single run (`theta_hist_fast`, `E_reg_fast`, `E_opt_fast`). |
| `data_validate_theoretical.mat` | `matlab/validate_theoretical_sto.m` | Fig. 6, Fig. 7 | Two-phase deployment: offline / deployed / online-STO ensembles (`mu_off`, `mu_deploy`, `X_opt`, and gain-matched `mu_off_loose`, `mu_deploy_loose`), action-error and parameter-error histories (`Action_Err`, `E_T_opt`), parameters (`theta_off`, `theta_opt_final`, `theta_star`, `theta_star_lambda`), δ and λ sweeps (`E_T_deltas`, `E_T_unreg_d05`, `E_T_reg_opt_d05`, `E_T_reg_star_d05`). |
| `data_bifurcation_sweep.mat` | `matlab/bifurcation_stability_loss_sto.m` | Fig. 8, Fig. 9 | Bifurcation scatter data (`bifurcation_data_delta`, `bifurcation_data_state`, δ ∈ [1.50, 2.25]), density histories for the damped (`P_den_stable`, δ = `delta_stable`) and dispersed (`P_den_chaos`, δ = `delta_chaos`) regimes, candidate list (`delta_candidates`, `P_den_cand`). |
| `data_fig4_linear_response.mat` | `matlab/fig_linear_response_contribution.m` | Fig. 4 | Ulam-operator (M = 1 200) and N = 10⁵ particle relaxation of the misspecified quadratic compensator: `err_ulam_odd`, `err_part_odd`, `err_ulam_even`, rates `rate_H_odd`, `rate_J_odd`, `rate_H_ev`, `rate_J_ev`, stationary density `mu_star_lr` and linear response `dmu_dtheta1` on `x_grid`. |

Load in Python with `scipy.io.loadmat`; load in MATLAB with `load`.
