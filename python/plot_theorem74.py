"""
==============================================================================
PLOT_THEOREM74.PY - High-Resolution Publication Vector Renderer (Figure 3)

Manuscript:
  "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
   A Barycentric State-Parameter Formulation" (Section 8.1 & 8.2, Figure 3)

Authors:
  Stefano Galatolo (Universita di Pisa, stefano.galatolo@unipi.it)
  Vittorio Lippi (Calejo Hybrid Intelligence AB, vittorio.lippi@calejo.ai)

Publication & License:
  - Target Venue:     Journal of Nonlinear Science (submitted, 2026)
  - License:          MIT License (c) 2026 Stefano Galatolo and Vittorio Lippi

Subject Classification & Keywords:
  AMS MSC (2020): 37C30, 93C40, 37M25, 34E13, 62L20
  Keywords: Self-Consistent Transfer Operator (STO), Time-Scale Separation,
            Exponential Relaxation Rate, Proposition 7.7 Equilibrium Shift,
            Slow-Time Collapse tau = delta * n, Reflected Noisy Cubic Control System.

Scientific Questions Visualized:
  - Figure 3 Left: How does the exponential convergence rate toward theta_*^lambda
    scale with the learning rate delta across {0.005, 0.02, 0.05, 0.08, 2.02}?
  - Figure 3 Right: How does the O(lambda) equilibrium bias relative to theta_opt
    compare with exponential convergence toward the shifted root theta_*^lambda?
  - Diagnostic Figure: Do trajectories across different learning rates collapse
    onto a single universal curve exp(-(lambda_min(H_*) + lambda) * tau) when
    plotted against rescaled slow time tau = delta * n?

Input Data Dependencies (relative to the repository root):
  - data/data_fast_theorem74.mat

Output Artifacts:
  - figures/fig_part1_timescale_and_reg.pdf
  - figures/supplementary/fig_diagnostic_loglog_vs_semilogy.pdf
==============================================================================
"""

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

def generate_plots():
    data = sio.loadmat(str(DATA_DIR / 'data_fast_theorem74.mat'))
    deltas = data['deltas'].flatten()
    E_reg_sweep = data['E_reg_sweep']
    E_opt_sweep = data['E_opt_sweep']
    E_unreg_d05 = data['E_unreg_d05'].flatten()
    theta_star_lambda = data['theta_star_lambda'].flatten()
    theta_opt = data['theta_opt'].flatten()
    lambda_min_H = float(np.squeeze(data['lambda_min_H']))
    lambda_val = 1e-3
    rate_theory = lambda_min_H + lambda_val
    N_steps = E_reg_sweep.shape[1]
    steps = np.arange(1, N_steps + 1)

    colors = [
        '#0072BD', # delta = 0.005 (blue)
        '#D95319', # delta = 0.020 (orange)
        '#7E2F8E', # delta = 0.050 (purple)
        '#77AC30', # delta = 0.080 (green)
        '#A2142F'  # delta = 2.020 (dark red)
    ]

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.autolayout': True
    })

    # ==========================================
    # FIGURE 3: fig_part1_timescale_and_reg.pdf
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=300)

    # Left panel: Exponential Convergence to theta_*^lambda across delta
    for i, d in enumerate(deltas):
        label_str = rf'$\delta = {d:.3f}$'
        if d > 1.0:
            label_str += ' (Divergent, $\\delta > \\delta_c$)'
        ax1.semilogy(steps, E_reg_sweep[i, :], label=label_str, color=colors[i], linewidth=1.8)

    ax1.set_ylim([1e-4, 1e1])
    ax1.set_xlim([1, N_steps])
    ax1.set_xlabel('Adaptation Steps ($n$)')
    ax1.set_ylabel(r'$\|\theta_n - \theta_*^\lambda\|$')
    ax1.set_title(r'Local exponential convergence to $\theta^\lambda_*$ ($\lambda=10^{-3}$, $\sigma=0.18$)')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # Right panel: Direct comparison unregularized vs regularized
    ax2.semilogy(steps, E_unreg_d05, label=r'Unregularized to $\theta_{\rm opt}$ ($\lambda = 0$)',
                 color='#D95319', linestyle='-', linewidth=2.0)
    ax2.semilogy(steps, E_opt_sweep[2, :], label=r'Regularized to $\theta_{\rm opt}$ ($\lambda = 10^{-3}$, Prop. 7.7 Bias)',
                 color='#EDB120', linestyle='--', linewidth=1.8)
    if 'E_opt_lam5' in data:
        E_opt_lam5 = data['E_opt_lam5'].flatten()
        ax2.semilogy(steps, E_opt_lam5, label=r'Ablation to $\theta_{\rm opt}$ ($\lambda = 5\times 10^{-3}$, Prop. 7.7)',
                     color='#77AC30', linestyle=':', linewidth=1.8)
    ax2.semilogy(steps, E_reg_sweep[2, :], label=r'Regularized to $\theta_*^\lambda$ ($\lambda = 10^{-3}$, Thm. 5.2 / Prop. 7.7 (local))',
                 color='#0072BD', linestyle='-.', linewidth=2.0)

    ax2.set_ylim([1e-4, 1e1])
    ax2.set_xlim([1, N_steps])
    ax2.set_xlabel('Adaptation Steps ($n$)')
    ax2.set_ylabel('Distance to Target Parameter')
    ax2.set_title(r'Equilibrium Bias vs Exponential Convergence ($\delta = 0.05$)')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5)

    fig.tight_layout()
    fig.savefig(str(FIG_DIR / 'fig_part1_timescale_and_reg.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_part1_timescale_and_reg.pdf")

    # ==============================================================
    # DIAGNOSTIC FIGURE: fig_diagnostic_loglog_vs_semilogy.pdf
    # ==============================================================
    fig_diag, (ax_semi, ax_loglog, ax_rescale) = plt.subplots(1, 3, figsize=(16.0, 4.2), dpi=300)

    # (a) Semilogy: Slope proportional to -delta * (lambda_min + lambda)
    for i in range(4):
        ax_semi.semilogy(steps, E_reg_sweep[i, :], label=rf'$\delta = {deltas[i]:.3f}$',
                         color=colors[i], linewidth=1.8)
    ax_semi.set_ylim([1e-4, 1e1])
    ax_semi.set_xlim([1, N_steps])
    ax_semi.set_xlabel('Adaptation Steps ($n$)')
    ax_semi.set_ylabel(r'$\|\theta_n - \theta_*^\lambda\|$')
    ax_semi.set_title(r'(a) Semilogy: Slopes $\propto -\delta(\lambda_{\min}(H_*) + \lambda)$')
    ax_semi.grid(True, which='both', linestyle=':', alpha=0.6)
    ax_semi.legend(loc='upper right')

    # (b) Log-Log Scale
    for i in range(4):
        ax_loglog.loglog(steps, E_reg_sweep[i, :], label=rf'$\delta = {deltas[i]:.3f}$',
                          color=colors[i], linewidth=1.8)
    ax_loglog.set_ylim([1e-4, 1e1])
    ax_loglog.set_xlim([1, N_steps])
    ax_loglog.set_xlabel(r'$\log_{10}$ Adaptation Steps ($n$)')
    ax_loglog.set_ylabel(r'$\log_{10} \|\theta_n - \theta_*^\lambda\|$')
    ax_loglog.set_title(r'(b) Log-Log Scale (Decay Curves)')
    ax_loglog.grid(True, which='both', linestyle=':', alpha=0.6)
    ax_loglog.legend(loc='upper right')

    # (c) Rescaled Time Collapse: tau = delta * n
    tau_master = np.linspace(0, 800, 500)
    theta_init = np.array([-0.25, 0.45, 0.0, 0.0])
    init_err = np.linalg.norm(theta_init - theta_star_lambda)
    ax_rescale.semilogy(tau_master, init_err * np.exp(-rate_theory * tau_master),
                        'k--', linewidth=2.2, label=r'Theory: $\exp(-(\lambda_{\min}+\lambda)\tau)$')
    for i in range(4):
        tau = deltas[i] * steps
        ax_rescale.semilogy(tau, E_reg_sweep[i, :], label=rf'$\delta = {deltas[i]:.3f}$',
                            color=colors[i], linewidth=1.6)
    ax_rescale.set_ylim([1e-4, 1e1])
    ax_rescale.set_xlim([0, 800])
    ax_rescale.set_xlabel(r'Rescaled Slow Time $\tau = \delta \cdot n$')
    ax_rescale.set_ylabel(r'$\|\theta_n - \theta_*^\lambda\|$')
    ax_rescale.set_title(r'(c) Rescaled Time Collapse (Master Decay)')
    ax_rescale.grid(True, which='both', linestyle=':', alpha=0.6)
    ax_rescale.legend(loc='upper right')

    fig_diag.tight_layout()
    fig_diag.savefig(str(SUPP_DIR / 'fig_diagnostic_loglog_vs_semilogy.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig_diag)
    print("Saved fig_diagnostic_loglog_vs_semilogy.pdf")

if __name__ == '__main__':
    generate_plots()

# ==============================================================================
# ARCHIVAL CITATION
#   Galatolo, S., & Lippi, V. (2026). Autonomous Neural Adaptive Control as a
#   Self-Consistent Transfer Operator. Submitted to Journal of Nonlinear Science.
# ==============================================================================
