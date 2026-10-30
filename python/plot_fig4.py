"""
==============================================================================
PLOT_FIG4.PY - High-Resolution Publication Vector Renderer (Figure 4)

Manuscript:
  "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
   A Barycentric State-Parameter Formulation" (Section 8.2.1, Figure 4)

Authors:
  Stefano Galatolo (Universita di Pisa, stefano.galatolo@unipi.it)
  Vittorio Lippi (Calejo Hybrid Intelligence AB, vittorio.lippi@calejo.ai)

Scientific Questions Visualized:
  - Panel (a): Odd-Mode Relaxation: Frozen Hessian H_* vs Full Jacobian H_* + C_* R_* B_*
  - Panel (b): Slowest Even-Mode Relaxation: lambda_min(H_even) vs lambda_min(H_even + C_* R_* B_*)
  - Panel (c): Physical Mechanism: Stationary Density mu_{theta_*}(x) and Linear Response R_* B_* e_2

Input / Output (relative to the repository root):
  - data/data_fig4_linear_response.mat   (input; recomputed in Python if absent)
  - figures/fig_linear_response_contribution.pdf   (output)
==============================================================================
"""

import os
import time
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

# Repository folders (independent of the current working directory)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
FIG_DIR = REPO_ROOT / 'figures'
SUPP_DIR = FIG_DIR / 'supplementary'
FIG_DIR.mkdir(exist_ok=True)
SUPP_DIR.mkdir(exist_ok=True)

def reflect(x):
    return np.abs(np.mod(x - 2.0, 8.0) - 4.0) - 2.0

def build_ulam_operator(F_x, x_grid, dx, sigma):
    inv_s2 = 1.0 / (sigma**2)
    norm_c = dx / (np.sqrt(2.0 * np.pi) * sigma)
    P = np.zeros((len(x_grid), len(x_grid)))
    K_deriv = np.zeros((len(x_grid), len(x_grid)))
    x_col = x_grid[:, None]
    F_row = F_x[None, :]
    for m in [-1, 0, 1]:
        z1 = (x_col + 8.0 * m) - F_row
        z2 = (4.0 - x_col + 8.0 * m) - F_row
        q1 = norm_c * np.exp(-0.5 * (z1**2) * inv_s2)
        q2 = norm_c * np.exp(-0.5 * (z2**2) * inv_s2)
        P += q1 + q2
        K_deriv += (z1 * inv_s2) * q1 + (z2 * inv_s2) * q2
    P /= np.sum(P, axis=0, keepdims=True)
    return P, K_deriv

def stationary_distribution(P):
    M = P.shape[0]
    A = np.eye(M) - P
    A[-1, :] = 1.0
    b = np.zeros(M)
    b[-1] = 1.0
    mu = np.linalg.solve(A, b)
    mu = np.maximum(mu, 0.0)
    mu /= np.sum(mu)
    return mu

def compute_reduced_jacobian_ulam(theta, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl):
    p = len(theta)
    M = len(x_grid)
    F_x = A_cl * x_grid + d_grid - (theta @ Phi_grid)
    P_star, K_deriv = build_ulam_operator(F_x, x_grid, dx, sigma)
    mu_star = stationary_distribution(P_star)
    res = (theta @ Phi_grid) - d_grid
    H_star = (Phi_grid * mu_star) @ Phi_grid.T
    B_star = np.zeros((M, p))
    for j in range(p):
        dF_j = -Phi_grid[j, :]
        col = (K_deriv * dF_j[None, :]) @ mu_star
        col -= np.sum(col) * mu_star
        B_star[:, j] = col
    R_B = np.linalg.solve(np.eye(M) - P_star + np.outer(mu_star, np.ones(M)), B_star)
    for j in range(p):
        R_B[:, j] -= np.sum(R_B[:, j]) * mu_star
    C_star = Phi_grid * res[None, :]
    CRB_star = C_star @ R_B
    J_star = H_star + lam * np.eye(p) + CRB_star
    dmu_dtheta1 = R_B[:, min(p-1, 1)]
    return J_star, H_star, CRB_star, mu_star, dmu_dtheta1

def solve_reduced_equilibrium_ulam(lam, theta_init, Phi_grid, d_grid, x_grid, dx, sigma, A_cl):
    theta_eq = theta_init.copy()
    for _ in range(35):
        J, _, _, _, _ = compute_reduced_jacobian_ulam(theta_eq, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl)
        res = (theta_eq @ Phi_grid) - d_grid
        F_x = A_cl * x_grid + d_grid - (theta_eq @ Phi_grid)
        P_star, _ = build_ulam_operator(F_x, x_grid, dx, sigma)
        mu_star = stationary_distribution(P_star)
        G_val = Phi_grid @ (res * mu_star) + lam * theta_eq
        if np.linalg.norm(G_val) < 1e-13:
            break
        theta_eq -= np.linalg.solve(J, G_val)
    return theta_eq

def sample_from_grid_pdf(x_grid, mu_weights, N):
    cdf = np.cumsum(mu_weights)
    cdf /= cdf[-1]
    u = np.random.rand(N)
    bin_idx = np.searchsorted(cdf, u)
    bin_idx = np.clip(bin_idx, 0, len(x_grid) - 1)
    dx = x_grid[1] - x_grid[0]
    return x_grid[bin_idx] + (np.random.rand(N) - 0.5) * dx

def generate_figure4():
    mat_file = str(DATA_DIR / 'data_fig4_linear_response.mat')
    if os.path.exists(mat_file):
        print(f"Loading precomputed data from {mat_file}...")
        data = sio.loadmat(mat_file)
        steps_vec = data['steps_vec'].flatten()
        rate_H_odd = float(np.squeeze(data['rate_H_odd']))
        rate_J_odd = float(np.squeeze(data['rate_J_odd']))
        dtheta_odd_0 = data['dtheta_odd_0'].flatten()
        err_ulam_odd = data['err_ulam_odd'].flatten()
        err_part_odd = data['err_part_odd'].flatten()
        v_slow_even = data['v_slow_even'].flatten()
        rate_H_ev = float(np.squeeze(data['rate_H_ev']))
        rate_J_ev = float(np.squeeze(data['rate_J_ev']))
        err_ulam_even = data['err_ulam_even'].flatten()
        x_grid = data['x_grid'].flatten()
        dx = float(np.squeeze(data['dx']))
        mu_star_lr = data['mu_star_lr'].flatten()
        dmu_dtheta1 = data['dmu_dtheta1'].flatten()
        delta_lr = float(np.squeeze(data['delta_lr']))
        N_steps_lr = int(np.squeeze(data['N_steps_lr']))
    else:
        print("Precomputed data not found. Computing Ulam transfer operator and trajectories in Python...")
        t0 = time.time()
        A_cl = 0.80
        B_true = 0.15
        M_grid = 1200
        edges = np.linspace(-2.0, 2.0, M_grid + 1)
        dx = edges[1] - edges[0]
        x_grid = 0.5 * (edges[:-1] + edges[1:])
        d_grid = B_true * (x_grid**3)
        sigma_lr = 0.24
        Phi_grid_lr = np.vstack([np.ones(M_grid), x_grid, x_grid**2])

        theta_star_lr = solve_reduced_equilibrium_ulam(
            0.0, np.array([0.0, 0.08, 0.0]), Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl
        )
        J_lr, H_lr, _, mu_star_lr, dmu_dtheta1 = compute_reduced_jacobian_ulam(
            theta_star_lr, 0.0, Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl
        )

        delta_lr = 0.05
        N_steps_lr = 800
        N_ens = 100000
        dtheta_odd_0 = np.array([0.0, 0.12, 0.0])

        theta_u_odd = theta_star_lr + dtheta_odd_0
        mu_u_odd = mu_star_lr.copy()
        err_ulam_odd = np.zeros(N_steps_lr)

        np.random.seed(42)
        X_lr_odd = sample_from_grid_pdf(x_grid, mu_star_lr, N_ens)
        theta_p_odd = theta_star_lr + dtheta_odd_0
        err_part_odd = np.zeros(N_steps_lr)

        print("  Running odd-mode simulation (800 steps)...")
        for n in range(N_steps_lr):
            err_ulam_odd[n] = np.linalg.norm(theta_u_odd - theta_star_lr)
            F_uo = A_cl * x_grid + d_grid - (theta_u_odd @ Phi_grid_lr)
            res_uo = (theta_u_odd @ Phi_grid_lr) - d_grid
            G_uo = Phi_grid_lr @ (res_uo * mu_u_odd)
            P_step, _ = build_ulam_operator(F_uo, x_grid, dx, sigma_lr)
            mu_u_odd = P_step @ mu_u_odd
            theta_u_odd -= delta_lr * G_uo

            err_part_odd[n] = np.linalg.norm(theta_p_odd - theta_star_lr)
            noise_lr = sigma_lr * np.random.randn(N_ens)
            phi_lr = np.vstack([np.ones(N_ens), X_lr_odd, X_lr_odd**2])
            u_lr = theta_p_odd @ phi_lr
            d_lr = B_true * (X_lr_odd**3)
            G_p_lr = (phi_lr @ (u_lr - d_lr)) / N_ens
            X_lr_odd = reflect(A_cl * X_lr_odd + d_lr - u_lr + noise_lr)
            theta_p_odd -= delta_lr * G_p_lr

        print("  Running even-mode simulation (800 steps)...")
        even_idx = [0, 2]
        J_even = J_lr[np.ix_(even_idx, even_idx)]
        evals_even, evecs_even = np.linalg.eig(J_even)
        idx_min_ev = np.argmin(evals_even)
        v_min = evecs_even[:, idx_min_ev].real
        v_slow_even = np.zeros(3)
        v_slow_even[even_idx] = 0.08 * v_min / np.linalg.norm(v_min)

        theta_u_even = theta_star_lr + v_slow_even
        mu_u_even = mu_star_lr.copy()
        err_ulam_even = np.zeros(N_steps_lr)
        for n in range(N_steps_lr):
            err_ulam_even[n] = np.linalg.norm(theta_u_even - theta_star_lr)
            F_ue = A_cl * x_grid + d_grid - (theta_u_even @ Phi_grid_lr)
            res_ue = (theta_u_even @ Phi_grid_lr) - d_grid
            G_ue = Phi_grid_lr @ (res_ue * mu_u_even)
            P_step, _ = build_ulam_operator(F_ue, x_grid, dx, sigma_lr)
            mu_u_even = P_step @ mu_u_even
            theta_u_even -= delta_lr * G_ue

        steps_vec = np.arange(N_steps_lr)
        rate_H_odd = float(H_lr[1, 1])
        rate_J_odd = float(J_lr[1, 1])
        H_even = H_lr[np.ix_(even_idx, even_idx)]
        rate_H_ev = float(np.min(np.linalg.eigvals(H_even).real))
        rate_J_ev = float(np.min(evals_even.real))

        # Save to MAT file for future instant runs
        sio.savemat(mat_file, {
            'steps_vec': steps_vec, 'rate_H_odd': rate_H_odd, 'rate_J_odd': rate_J_odd,
            'dtheta_odd_0': dtheta_odd_0, 'err_ulam_odd': err_ulam_odd, 'err_part_odd': err_part_odd,
            'v_slow_even': v_slow_even, 'rate_H_ev': rate_H_ev, 'rate_J_ev': rate_J_ev,
            'err_ulam_even': err_ulam_even, 'x_grid': x_grid, 'dx': dx, 'mu_star_lr': mu_star_lr,
            'dmu_dtheta1': dmu_dtheta1, 'delta_lr': delta_lr, 'N_steps_lr': N_steps_lr
        })
        print(f"  Computation complete ({time.time()-t0:.1f}s), saved to {mat_file}.")

    # =========================================================================
    # PLOTTING
    # =========================================================================
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'mathtext.fontset': 'cm'
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=300)

    # Panel (a): Odd Mode Relaxation
    ax1.semilogy(steps_vec, np.linalg.norm(dtheta_odd_0) * np.exp(-delta_lr * rate_H_odd * steps_vec),
                 'r--', linewidth=2.0, label=rf'Frozen Hessian $H_*$ ($\lambda = {rate_H_odd:.3f}$)')
    ax1.semilogy(steps_vec, np.linalg.norm(dtheta_odd_0) * np.exp(-delta_lr * rate_J_odd * steps_vec),
                 'k-', linewidth=2.2, label=rf'Full $D\overline{{G}}(\theta_*) = H_* + C_*R_*B_*$ ($\lambda = {rate_J_odd:.3f}$)')
    ax1.semilogy(steps_vec, err_ulam_odd, 'b-.', linewidth=1.8,
                 label=r'Deterministic discretized STO orbit $\Phi_\delta(\mu_n,\theta_n)$')
    ax1.semilogy(steps_vec, err_part_odd, color=[0.0, 0.65, 0.3], linestyle=':', linewidth=1.8,
                 label=r'Particle Ensemble ($N = 10^5$)')
    ax1.set_ylim([1e-4, 2e-1])
    ax1.set_xlim([0, N_steps_lr])
    ax1.set_xlabel(r'Adaptation Steps $n$ [-]')
    ax1.set_ylabel(r'Parameter Error $\|\theta_n - \theta_*\|$ [-]')
    ax1.set_title(r'(a) Linear Response Speedup: Odd Mode ($\psi_2(x) = x$)')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # Panel (b): Slowest Even-Mode Relaxation
    ax2.semilogy(steps_vec, np.linalg.norm(v_slow_even) * np.exp(-delta_lr * rate_H_ev * steps_vec),
                 'r--', linewidth=2.0, label=rf'Frozen Hessian $\lambda_{{\min}}(H_{{\rm even}}) = {rate_H_ev:.3f}$')
    ax2.semilogy(steps_vec, np.linalg.norm(v_slow_even) * np.exp(-delta_lr * rate_J_ev * steps_vec),
                 'k-', linewidth=2.2, label=rf'Full $\lambda_{{\min}}(H_{{\rm even}}+C_*R_*B_*) = {rate_J_ev:.3f}$')
    ax2.semilogy(steps_vec, err_ulam_even, 'b-.', linewidth=1.8,
                 label=r'Deterministic discretized STO orbit $\Phi_\delta(\mu_n,\theta_n)$')
    ax2.set_ylim([2e-3, 1.2e-1])
    ax2.set_xlim([0, N_steps_lr])
    ax2.set_xlabel(r'Adaptation Steps $n$ [-]')
    ax2.set_ylabel(r'Parameter Error $\|\theta_n - \theta_*\|$ [-]')
    ax2.set_title(r'(b) Linear Response Speedup: Even Mode ($\psi_{1,3} = 1, x^2$)')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    # Panel (c): Stationary Density & Linear Response Sensitivity
    l1 = ax3.plot(x_grid, mu_star_lr / dx, 'b-', linewidth=1.8, label=r'Stationary Density $\mu_{\theta_*}(x)$')
    ax3.set_xlabel(r'Physical State $x$ [-]')
    ax3.set_ylabel(r'Probability Density $\mu_{\theta_*}(x)$ [$x^{-1}$]', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_xlim([-2, 2])
    ax3.grid(True, which='both', linestyle=':', alpha=0.6)

    ax3_right = ax3.twinx()
    l2 = ax3_right.plot(x_grid, dmu_dtheta1 / dx, 'r-', linewidth=1.8, label=r'Linear Response $D_{\theta_1}\mu_{\theta_*}(x)$')
    ax3_right.axhline(0, color='k', linestyle=':', linewidth=1.0)
    ax3_right.set_ylabel(r'Sensitivity $[R_* B_* e_2](x)$ [$x^{-1}$]', color='r')
    ax3_right.tick_params(axis='y', labelcolor='r')

    # Panel (c) title using raw string and mathtext
    ax3.set_title(r"(c) Stationary Density & Linear Response $R_*B_*$")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    output_pdf = str(FIG_DIR / 'fig_linear_response_contribution.pdf')
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully generated and saved {output_pdf}!")

if __name__ == '__main__':
    generate_figure4()
