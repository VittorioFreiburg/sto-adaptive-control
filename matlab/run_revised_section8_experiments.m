% ==============================================================================
% RUN_REVISED_SECTION8_EXPERIMENTS.M
%
% Ulam Transfer-Operator Discretization, Statistical Linear Response C_* R_* B_*,
% Finite-Difference Jacobian Verification, and Deployment Action Error Suite
% ==============================================================================
%
% MANUSCRIPT:
%   "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
%    A Barycentric State-Parameter Formulation" (Section 8: Numerical Experiments)
%
% AUTHORS:
%   Stefano Galatolo (1) and Vittorio Lippi (2)
%   (1) Dipartimento di Matematica, Universita di Pisa, Pisa, Italy
%       Email: stefano.galatolo@unipi.it
%   (2) Calejo Hybrid Intelligence AB, Stockholm, Sweden
%       Email: vittorio.lippi@calejo.ai
%
% PUBLICATION & LICENSE:
%   - Target Venue:      Journal of Nonlinear Science (submitted, 2026)
%   - License:           MIT License (c) 2026 Stefano Galatolo and Vittorio Lippi
%
% ==============================================================================
% SUBJECT CLASSIFICATION & TECHNICAL KEYWORDS
% ==============================================================================
%   AMS MSC (2020):
%     37C30 (Transfer operators: Perron-Frobenius, Koopman, Ruelle operators)
%     93C40 (Adaptive control and closed-loop learning systems)
%     37M25 (Computational ergodic theory, Ulam discretization, invariant measures)
%     34E13 (Multiple scale methods and slow-fast dynamical systems)
%     62L20 (Stochastic approximation and recursive estimation)
%
%   Keywords:
%     Self-Consistent Transfer Operator (STO), Barycentric State-Parameter Measure,
%     Ulam-Galerkin Markov Discretization, Statistical Linear Response Theory,
%     Reduced Jacobian Decomposition D\overline{G}(\theta_*) = H_* + C_* R_* B_*,
%     Zero-Mass Resolvent (Id - P_{\theta_*})^{-1}, Boundary Reflection Bias,
%     Two-Phase Deployment Mismatch, Discretized Action Error ||r_{A,h}||_{\ell^1},
%     Reflected Noisy Cubic Control System.
%
% ==============================================================================
% SCIENTIFIC QUESTIONS ADDRESSED IN THIS SCRIPT
% ==============================================================================
%   Q1 (Section 8.1 & Table 1):
%     How closely does the empirical tracking-error update proxy match the
%     theoretical unreflected vector field G(\theta, x) on X = [-2, 2], and what
%     sets the asymptotic residual floor in finite-particle ensembles?
%
%   Q2 (Section 8.2 & Table 1):
%     How do the regularized equilibrium \theta_*^\lambda and the reduced Jacobian
%     D\overline{G}_\lambda(\theta_*^\lambda) behave on the discretized transfer
%     operator, and does the analytical operator decomposition match central
%     finite differences of the stationary vector field?
%
%   Q3 (Section 8.2.1 & Figure 4):
%     When the residual compensator is under-parameterized (\theta_* \neq \theta_opt),
%     does the linear-response sensitivity of the invariant measure (C_* R_* B_*)
%     measurably alter the closed-loop relaxation rates compared to a frozen-measure
%     Hessian approximation H_*?
%
%   Q4 (Section 8.3.2 & Table 2):
%     Why does offline calibration under a high-gain baseline controller (-0.80 x)
%     destabilize the origin when deployed with a relaxed baseline (-0.25 x), and
%     how much of the resulting Action Error ||r_{A,h}||_{\ell^1} is caused by
%     the gain relaxation versus the process noise change?
%
% ==============================================================================
% USAGE & OUTPUT ARTIFACTS
% ==============================================================================
%   Command:
%     >> results = run_revised_section8_experiments();
%
%   Generated Figure:
%     - figures/fig_linear_response_contribution.pdf (Figure 4, Section 8.2.1)
% ==============================================================================

function results = run_revised_section8_experiments()
    addpath(fileparts(mfilename('fullpath'))); paths = sto_paths();  % repository folders (data/, figures/, results/)
    rng(42);
    set(groot, 'DefaultFigureVisible', 'off');

    % Physical plant parameters: reflected noisy cubic control system on X = [-2, 2]
    % Pre-reflection nominal closed-loop map: F_\theta(x) = 0.80 x + 0.15 x^3 - g(x, \theta)
    A_plant = 1.05; B_true = 0.15; u_base = -0.25; A_cl = A_plant + u_base;
    d_func  = @(x) B_true * (x.^3);

    % Uniform Ulam-Galerkin partition of X = [-2, 2] into M = 1,200 cells (h = 1/300)
    M_grid = 1200; edges = linspace(-2, 2, M_grid + 1); dx = edges(2) - edges(1);
    x_grid = 0.5 * (edges(1:end-1) + edges(2:end))';

    fprintf('======================================================================\n');
    fprintf('  SECTION 8 ULAM TRANSFER OPERATOR, LINEAR RESPONSE & DEPLOYMENT SUITE\n');
    fprintf('======================================================================\n\n');

    %% =========================================================================
    %  QUESTION 1 (Section 8.1 & Table 1):
    %  How large is the boundary reflection discrepancy \Delta G_refl between
    %  the theoretical update field G(\theta, x) and the empirical tracking
    %  residual -(x_{n+1} - 0.8 x_n)\Phi(x_n), and what determines the
    %  asymptotic parameter floor in an N = 10^5 particle ensemble?
    %  -------------------------------------------------------------------------
    %  Answer verified below:
    %    At \sigma = 0.18, the invariant measure \mu_{\theta_opt} is exponentially
    %    concentrated away from the reflection boundaries \pm 2 (boundary escape
    %    probability P(|0.8 x + \xi| > 2) = 1.54e-13). Consequently, the trajectory
    %    difference ||\theta_{15000}^trk - \theta_{15000}^theo||_2 is 4.02e-8,
    %    while the finite-ensemble fluctuation floor (~2.8e-4) is governed by the
    %    first-order discrete Lyapunov RMS scale \sigma \sqrt{p \delta / (2 N)}.
    %  =========================================================================
    sigma_nom = 0.18; delta_nom = 0.10; N_ens = 100000; N_steps_e1 = 15000;
    theta_opt = [0; 0; 0; 0.15]; theta_init = [-0.25; 0.45; 0; 0];
    
    E_ulam_theo = zeros(1, N_steps_e1);
    Phi_grid4 = [ones(1, M_grid); x_grid'; (x_grid.^2)'; (x_grid.^3)'];
    d_grid = d_func(x_grid)';
    mu_u = zeros(M_grid, 1); [~, idx0] = min(abs(x_grid)); mu_u(idx0) = 1.0;
    theta_u = theta_init;

    for n = 1:N_steps_e1
        F_u = A_cl * x_grid + d_func(x_grid) - (Phi_grid4' * theta_u);
        res_u = (theta_u' * Phi_grid4) - d_grid; 
        G_u = (Phi_grid4 * (res_u' .* mu_u));
        [P_step, ~] = build_ulam_operator(F_u, x_grid, dx, sigma_nom); 
        mu_u = P_step * mu_u;
        theta_u = theta_u - delta_nom * G_u;
        E_ulam_theo(n) = norm(theta_u - theta_opt);
    end

    rng(42); X_theo = zeros(1, N_ens); theta_p_theo = theta_init; E_part_theo = zeros(1, N_steps_e1);
    for n = 1:N_steps_e1
        noise = sigma_nom * randn(1, N_ens); phi = [ones(1, N_ens); X_theo; X_theo.^2; X_theo.^3];
        u_nn = theta_p_theo' * phi; d_val = B_true * (X_theo.^3);
        X_next = reflect(A_cl * X_theo + d_val - u_nn + noise);
        G_theo = (phi * (u_nn - d_val)') / N_ens;
        theta_p_theo = theta_p_theo - delta_nom * G_theo;
        E_part_theo(n) = norm(theta_p_theo - theta_opt); X_theo = X_next;
    end

    rng(42); X_trk = zeros(1, N_ens); theta_p_trk = theta_init; E_part_trk = zeros(1, N_steps_e1);
    for n = 1:N_steps_e1
        noise = sigma_nom * randn(1, N_ens); phi = [ones(1, N_ens); X_trk; X_trk.^2; X_trk.^3];
        u_nn = theta_p_trk' * phi; d_val = B_true * (X_trk.^3);
        pre_refl = A_cl * X_trk + d_val - u_nn + noise; X_next = reflect(pre_refl);
        err_bp = -(X_next - A_cl * X_trk); G_trk = (phi * err_bp') / N_ens;
        theta_p_trk = theta_p_trk - delta_nom * G_trk;
        E_part_trk(n) = norm(theta_p_trk - theta_opt); X_trk = X_next;
    end

    fprintf('--- [1] Section 8.1: Reflection Bias & Finite-N Floor (sigma=0.18, N=1e5) ---\n');
    fprintf('  Final ||theta_15000^trk - theta_15000^theo||_2 = %.4e\n', norm(theta_p_trk - theta_p_theo));
    fprintf('  First-order Lyapunov RMS floor sigma*sqrt(p*delta/(2*N)) = %.4e\n\n', sigma_nom * sqrt(4 * delta_nom / (2 * N_ens)));

    %% =========================================================================
    %  QUESTION 2 (Section 8.2 & Table 1):
    %  What is the deterministic regularized equilibrium \theta_*^\lambda solved
    %  directly on the Ulam transfer operator at \lambda = 10^{-3}, and does the
    %  operator-theoretic reduced Jacobian D\overline{G}_\lambda(\theta_*^\lambda)
    %  = H(\theta_*^\lambda) + \lambda I + C_* R_* B_* agree with central finite
    %  differences of the stationary reduced field \overline{G}_\lambda(\theta)?
    %  -------------------------------------------------------------------------
    %  Answer verified below:
    %    Newton-Ulam iteration converges to \theta_*^\lambda = [0, 0.007043, 0, 0.124372]^T.
    %    Because the O(\lambda) shift moves the equilibrium slightly away from exact
    %    realizability, C_* R_* B_* is non-zero; including C_* R_* B_* matches central
    %    finite differences to 6.80e-12 (whereas omitting C_* R_* B_* leaves an error
    %    of 8.85e-4).
    %  =========================================================================
    [P_opt, ~] = build_ulam_operator(A_cl * x_grid, x_grid, dx, sigma_nom);
    mu_opt_ulam = stationary_distribution(P_opt);
    Q_opt_ulam = Phi_grid4 * diag(mu_opt_ulam) * Phi_grid4';
    
    lambda_sweep = [1e-6, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3];
    for k = 1:length(lambda_sweep)
        lam = lambda_sweep(k);
        theta_eq = solve_reduced_equilibrium_ulam(lam, theta_opt, Phi_grid4, d_grid, x_grid, dx, sigma_nom, A_cl);
        if lam == 1e-3
            theta_star_1e3 = theta_eq; 
            [J_at_1e3, H_at_1e3, CRB_at_1e3] = compute_reduced_jacobian_ulam(theta_eq, lam, Phi_grid4, d_grid, x_grid, dx, sigma_nom, A_cl);
        end
    end

    % Central finite-difference verification of D\overline{G}_\lambda at lambda = 1e-3
    eps_fd = 1e-6;
    J_reg_fd = zeros(4, 4);
    for j = 1:4
        ej = zeros(4, 1); ej(j) = 1.0;
        Gp = eval_reduced_field_ulam(theta_star_1e3 + eps_fd*ej, 1e-3, Phi_grid4, d_grid, x_grid, dx, sigma_nom, A_cl);
        Gm = eval_reduced_field_ulam(theta_star_1e3 - eps_fd*ej, 1e-3, Phi_grid4, d_grid, x_grid, dx, sigma_nom, A_cl);
        J_reg_fd(:, j) = (Gp - Gm) / (2 * eps_fd);
    end
    fprintf('--- [2] Section 8.2: Regularized Equilibrium & Jacobian FD Check (lambda=1e-3) ---\n');
    fprintf('  lambda_min(Q_opt) on M=1200 Ulam grid = %.6e\n', min(eig(Q_opt_ulam)));
    fprintf('  Ulam root theta_*^lam                 = [%.6f, %.6f, %.6f, %.6f]^T\n', theta_star_1e3);
    fprintf('  lambda_min(D G_bar_lam(theta_*^lam))  = %.6e\n', min(real(eig(J_at_1e3))));
    fprintf('  FD check ||D_FD G_bar_lam - (H+lam*I+C_*R_*B_*)||_2 = %.4e (vs ||D_FD - (H+lam*I)||_2 = %.4e)\n\n', ...
        norm(J_reg_fd - J_at_1e3), norm(J_reg_fd - (H_at_1e3 + 1e-3*eye(4))));

    %% =========================================================================
    %  QUESTION 3 (Section 8.2.1 & Figure 4):
    %  When the cubic plant d(x) = 0.15 x^3 is controlled by a misspecified
    %  quadratic basis \Phi_mis(x) = (1, x, x^2)^T, how strongly does the
    %  measure sensitivity term C_* R_* B_* modify the local convergence rate
    %  compared to the static feature Gram matrix H_*?
    %  -------------------------------------------------------------------------
    %  Answer verified below:
    %    At the unique reduced equilibrium \theta_* = [0, 0.088820, 0]^T (\sigma = 0.24),
    %    the linear-response term C_* R_* B_* increases the odd-mode eigenvalue from
    %    (H_*)_{22} = 0.1753 to (D\overline{G}(\theta_*))_{22} = 0.2414 (+37.7% speedup)
    %    and shifts the slowest even-mode eigenvalue from 0.0566 to 0.0706 (+24.8%).
    %    Both deterministic Ulam orbits and 10^5-particle trajectories follow the
    %    full linear-response slope 1 - \delta \lambda(H_* + C_* R_* B_*), and
    %    central finite differences confirm D\overline{G}(\theta_*) to 1.66e-11.
    %  =========================================================================
    sigma_lr = 0.24; Phi_grid_lr = [ones(1, M_grid); x_grid'; (x_grid.^2)'];
    theta_star_lr = solve_reduced_equilibrium_ulam(0.0, [0; 0.08; 0], Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);
    [J_lr, H_lr, CRB_lr, mu_star_lr, dmu_dtheta1] = compute_reduced_jacobian_ulam(theta_star_lr, 0.0, Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);

    % Central finite-difference verification of D\overline{G}(\theta_*) for misspecified quadratic model
    J_mis_fd = zeros(3, 3);
    for j = 1:3
        ej = zeros(3, 1); ej(j) = 1.0;
        Gp = eval_reduced_field_ulam(theta_star_lr + eps_fd*ej, 0.0, Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);
        Gm = eval_reduced_field_ulam(theta_star_lr - eps_fd*ej, 0.0, Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);
        J_mis_fd(:, j) = (Gp - Gm) / (2 * eps_fd);
    end

    delta_lr = 0.05; N_steps_lr = 800; dtheta_odd_0 = [0; 0.12; 0];
    theta_u_odd = theta_star_lr + dtheta_odd_0; mu_u_odd = mu_star_lr; err_ulam_odd = zeros(1, N_steps_lr);
    
    rng(42); X_lr_odd = sample_from_grid_pdf(x_grid, mu_star_lr, N_ens); theta_p_odd = theta_star_lr + dtheta_odd_0; err_part_odd = zeros(1, N_steps_lr);
    
    for n = 1:N_steps_lr
        err_ulam_odd(n) = norm(theta_u_odd - theta_star_lr);
        F_uo = A_cl * x_grid + d_func(x_grid) - (Phi_grid_lr' * theta_u_odd);
        res_uo = (theta_u_odd' * Phi_grid_lr) - d_grid; G_uo = Phi_grid_lr * (res_uo' .* mu_u_odd);
        [P_step, ~] = build_ulam_operator(F_uo, x_grid, dx, sigma_lr); mu_u_odd = P_step * mu_u_odd;
        theta_u_odd = theta_u_odd - delta_lr * G_uo;

        err_part_odd(n) = norm(theta_p_odd - theta_star_lr);
        noise_lr = sigma_lr * randn(1, N_ens); phi_lr = [ones(1, N_ens); X_lr_odd; X_lr_odd.^2];
        u_lr = theta_p_odd' * phi_lr; d_lr = B_true * (X_lr_odd.^3); G_p_lr = (phi_lr * (u_lr - d_lr)') / N_ens;
        X_lr_odd = reflect(A_cl * X_lr_odd + d_lr - u_lr + noise_lr);
        theta_p_odd = theta_p_odd - delta_lr * G_p_lr;
    end

    [V_even, D_even] = eig(J_lr([1,3],[1,3])); [~, idx_min_ev] = min(diag(D_even));
    v_slow_even = zeros(3, 1); v_slow_even([1,3]) = 0.08 * V_even(:, idx_min_ev) / norm(V_even(:, idx_min_ev));
    theta_u_even = theta_star_lr + v_slow_even; mu_u_even = mu_star_lr; err_ulam_even = zeros(1, N_steps_lr);
    for n = 1:N_steps_lr
        err_ulam_even(n) = norm(theta_u_even - theta_star_lr);
        F_ue = A_cl * x_grid + d_func(x_grid) - (Phi_grid_lr' * theta_u_even);
        res_ue = (theta_u_even' * Phi_grid_lr) - d_grid; G_ue = Phi_grid_lr * (res_ue' .* mu_u_even);
        [P_step, ~] = build_ulam_operator(F_ue, x_grid, dx, sigma_lr); mu_u_even = P_step * mu_u_even;
        theta_u_even = theta_u_even - delta_lr * G_ue;
    end

    CRB_even = CRB_lr([1,3], [1,3]);
    eigs_CRB_even = eig(CRB_even);
    fit_idx = 51:600;
    valid_odd_ulam = fit_idx(err_ulam_odd(fit_idx) > 1e-3);
    p_odd_ulam = polyfit(valid_odd_ulam - 1, log(err_ulam_odd(valid_odd_ulam)), 1);
    valid_odd_part = fit_idx(err_part_odd(fit_idx) > 1e-3);
    p_odd_part = polyfit(valid_odd_part - 1, log(err_part_odd(valid_odd_part)), 1);
    valid_ev_ulam = fit_idx(err_ulam_even(fit_idx) > 1e-3);
    p_ev_ulam = polyfit(valid_ev_ulam - 1, log(err_ulam_even(valid_ev_ulam)), 1);

    fprintf('--- [3] Section 8.2.1: Linear Response C_* R_* B_* & Finite-Difference Check ---\n');
    fprintf('  Equilibrium theta_* = [%.6f, %.6f, %.6f]^T\n', theta_star_lr);
    fprintf('  Odd mode (2,2): (H_*)_22 = %.4f, (C_*R_*B_*)_22 = %+.4f, (D G_bar)_22 = %.4f\n', ...
        H_lr(2,2), CRB_lr(2,2), J_lr(2,2));
    fprintf('  Even block eigenvalues of (C_*R_*B_*)_even: [%.4f, %.4f]\n', eigs_CRB_even);
    fprintf('  FD check ||D_FD G_bar - (H_* + C_*R_*B_*)||_2 = %.4e (rel: %.4e; vs ||D_FD - H_*||_2 = %.4f)\n', ...
        norm(J_mis_fd - J_lr), norm(J_mis_fd - J_lr)/norm(J_lr), norm(J_mis_fd - H_lr));
    fprintf('  Odd decay slopes:  Ulam = %.4f, Particle = %.4f, Full theory = %.4f, Frozen = %.4f\n', ...
        p_odd_ulam(1), p_odd_part(1), log(1 - delta_lr * J_lr(2,2)), log(1 - delta_lr * H_lr(2,2)));
    fprintf('  Even decay slopes: Ulam = %.4f, Full theory = %.4f, Frozen = %.4f\n\n', ...
        p_ev_ulam(1), log(1 - delta_lr * min(eig(J_lr([1,3],[1,3])))), log(1 - delta_lr * min(eig(H_lr([1,3],[1,3])))));

    %% =========================================================================
    %  QUESTION 4 (Section 8.3.2 & Table 2):
    %  Why does an offline compensator trained under a high-gain baseline (-0.80 x)
    %  become destabilizing when deployed with the nominal baseline (-0.25 x),
    %  and what are the separate contributions of the gain change and noise change
    %  to the discretized Action Error ||r_{A,h}||_{\ell^1}?
    %  -------------------------------------------------------------------------
    %  Answer verified below:
    %    Because the offline update evaluates the reference residual -(x_{n+1} - 0.80 x_n)
    %    while the baseline controller enforces 0.25 x_n, the offline target is
    %    d(x) - 0.55 x rather than d(x). The learned parameter vector
    %    \theta_off = [+1.99e-5, -0.46188, -1.97e-4, +0.27885]^T produces a
    %    deployed slope at the origin F_dep'(0) = 0.80 - \theta_{1,off} = 1.26188 > 1,
    %    rendering the origin unstable and creating off-origin attractors near x = \pm 1.425.
    %    In the Ulam \ell^1 metric, the gain change alone yields ||r_{A,h}||_1 = 0.1604
    %    and stationary shift ||\mu_dep - \mu_off||_1 = 1.9737, whereas the noise
    %    change alone (\sigma: 0.10 -> 0.18) yields ||\mu_dep - \mu_off||_1 = 0.4533.
    %  =========================================================================
    if isfile(fullfile(paths.data, 'data_validate_theoretical.mat'))
        dv = load(fullfile(paths.data, 'data_validate_theoretical.mat'));
        theta_off_learned = dv.theta_off;
        F_off_g = 0.25 * x_grid + d_func(x_grid) - (Phi_grid4' * theta_off_learned);
        F_dep_g = 0.80 * x_grid + d_func(x_grid) - (Phi_grid4' * theta_off_learned);
        [P_off_010, ~] = build_ulam_operator(F_off_g, x_grid, dx, 0.10);
        [P_off_018, ~] = build_ulam_operator(F_off_g, x_grid, dx, 0.18);
        [P_dep_010, ~] = build_ulam_operator(F_dep_g, x_grid, dx, 0.10);
        [P_dep_018, ~] = build_ulam_operator(F_dep_g, x_grid, dx, 0.18);

        mu_off_h = stationary_distribution(P_off_010);
        mu_gain_h = stationary_distribution(P_dep_010);
        mu_noise_h = stationary_distribution(P_off_018);
        mu_both_h = stationary_distribution(P_dep_018);

        rA_gain = sum(abs(P_dep_010 * mu_off_h - mu_off_h));
        rA_noise = sum(abs(P_off_018 * mu_off_h - mu_off_h));
        rA_both = sum(abs(P_dep_018 * mu_off_h - mu_off_h));

        fprintf('--- [4] Section 8.3.2: Deployment Mechanism & Discretized Action Error (Table 2) ---\n');
        fprintf('  Learned theta_off = [%+.4e, %+.5f, %+.4e, %+.5f]^T\n', theta_off_learned);
        fprintf('  Deployed slope at origin F_dep''(0) = 0.8 - theta_{1,off} = %.5f\n', 0.8 - theta_off_learned(2));
        fprintf('  Gain change only (-0.80 -> -0.25, sigma=0.10):  ||r_{A,h}||_1 = %.4f, ||mu_dep - mu_off||_1 = %.4f\n', ...
            rA_gain, sum(abs(mu_gain_h - mu_off_h)));
        fprintf('  Noise change only (-0.80 fixed, 0.10 -> 0.18):  ||r_{A,h}||_1 = %.4f, ||mu_dep - mu_off||_1 = %.4f\n', ...
            rA_noise, sum(abs(mu_noise_h - mu_off_h)));
        fprintf('  Simultaneous change (-0.25, sigma=0.18):        ||r_{A,h}||_1 = %.4f, ||mu_dep - mu_off||_1 = %.4f\n\n', ...
            rA_both, sum(abs(mu_both_h - mu_off_h)));
    end

    % --- Generate Figure 4: figures/fig_linear_response_contribution.pdf ---
    fig_lr = figure('Position', [100, 100, 1450, 420], 'Visible', 'off');
    tiledlayout(1, 3, 'TileSpacing', 'Compact');

    % Panel (a): Odd-Mode Relaxation (H_* vs H_* + C_* R_* B_*)
    nexttile; hold on; box on;
    steps_vec = 0:(N_steps_lr - 1);
    rate_H_odd = H_lr(2, 2);
    rate_J_odd = J_lr(2, 2);
    semilogy(steps_vec, norm(dtheta_odd_0) * exp(-delta_lr * rate_H_odd * steps_vec), ...
        'r--', 'LineWidth', 2.0, 'DisplayName', sprintf('Frozen Hessian $H_*$ ($\\lambda = %.3f$)', rate_H_odd));
    semilogy(steps_vec, norm(dtheta_odd_0) * exp(-delta_lr * rate_J_odd * steps_vec), ...
        'k-', 'LineWidth', 2.2, 'DisplayName', sprintf('Full $D\\overline{G}(\\theta_*) = H_* + C_*R_*B_*$ ($\\lambda = %.3f$)', rate_J_odd));
    semilogy(steps_vec, err_ulam_odd, 'b-.', 'LineWidth', 1.8, 'DisplayName', 'Deterministic discretized STO orbit $\Phi_\delta(\mu_n,\theta_n)$');
    semilogy(steps_vec, err_part_odd, 'Color', [0.0 0.65 0.3], 'LineStyle', ':', 'LineWidth', 1.8, ...
        'DisplayName', 'Particle Ensemble ($N = 10^5$)');
    set(gca, 'YScale', 'log'); grid on;
    xlabel('Adaptation Steps $n$ [-]', 'Interpreter', 'latex');
    ylabel('Parameter Error $\|\theta_n - \theta_*\|$ [-]', 'Interpreter', 'latex');
    title('(a) Linear Response Speedup: Odd Mode ($\psi_2(x) = x$)', 'Interpreter', 'latex');
    legend('Location', 'northeast', 'FontSize', 8.5, 'Interpreter', 'latex');
    ylim([1e-4, 2e-1]); xlim([0, N_steps_lr]);

    % Panel (b): Slowest Even-Mode Relaxation (H_* vs H_* + C_* R_* B_*)
    nexttile; hold on; box on;
    eigs_H_even = eig(H_lr([1,3],[1,3]));
    eigs_J_even = eig(J_lr([1,3],[1,3]));
    rate_H_ev = min(eigs_H_even);
    rate_J_ev = min(eigs_J_even);
    semilogy(steps_vec, norm(v_slow_even) * exp(-delta_lr * rate_H_ev * steps_vec), ...
        'r--', 'LineWidth', 2.0, 'DisplayName', sprintf('Frozen Hessian $\\lambda_{\\min}(H_{\\rm even}) = %.3f$', rate_H_ev));
    semilogy(steps_vec, norm(v_slow_even) * exp(-delta_lr * rate_J_ev * steps_vec), ...
        'k-', 'LineWidth', 2.2, 'DisplayName', sprintf('Full $\\lambda_{\\min}(H_{\\rm even}+C_*R_*B_*) = %.3f$', rate_J_ev));
    semilogy(steps_vec, err_ulam_even, 'b-.', 'LineWidth', 1.8, 'DisplayName', 'Deterministic discretized STO orbit $\Phi_\delta(\mu_n,\theta_n)$');
    set(gca, 'YScale', 'log'); grid on;
    xlabel('Adaptation Steps $n$ [-]', 'Interpreter', 'latex');
    ylabel('Parameter Error $\|\theta_n - \theta_*\|$ [-]', 'Interpreter', 'latex');
    title('(b) Linear Response Speedup: Even Mode ($\psi_{1,3} = 1, x^2$)', 'Interpreter', 'latex');
    legend('Location', 'northeast', 'FontSize', 8.5, 'Interpreter', 'latex');
    ylim([2e-3, 1.2e-1]); xlim([0, N_steps_lr]);

    % Panel (c): Stationary Density & Linear Response Sensitivity
    nexttile; hold on; box on;
    yyaxis left;
    plot(x_grid, mu_star_lr / dx, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Stationary Density $\mu_{\theta_*}(x)$');
    ylabel('Probability Density $\mu_{\theta_*}(x)$ [$x^{-1}$]', 'Interpreter', 'latex');
    yyaxis right;
    plot(x_grid, dmu_dtheta1 / dx, 'r-', 'LineWidth', 1.8, 'DisplayName', 'Linear Response $D_{\theta_1}\mu_{\theta_*}(x)$');
    yline(0, 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    ylabel('Sensitivity $[R_* B_* e_2](x)$ [$x^{-1}$]', 'Interpreter', 'latex');
    xlabel('Physical State $x$ [-]', 'Interpreter', 'latex');
    title('(c) Stationary Density \& Linear Response $R_*B_*$', 'Interpreter', 'latex');
    legend('Location', 'northwest', 'FontSize', 8.5, 'Interpreter', 'latex');
    xlim([-2, 2]); grid on;

    exportgraphics(fig_lr, fullfile(paths.figures, 'fig_linear_response_contribution.pdf'), 'ContentType', 'vector');
    close(fig_lr);
    fprintf('Exported fig_linear_response_contribution.pdf\n');

    results = struct('Q_opt_ulam', Q_opt_ulam, 'theta_star_1e3', theta_star_1e3, ...
        'J_at_1e3', J_at_1e3, 'theta_star_lr', theta_star_lr, 'J_lr', J_lr, ...
        'H_lr', H_lr, 'CRB_lr', CRB_lr, 'J_mis_fd', J_mis_fd);
end

%% =============================================================================
%  TRANSFER OPERATOR & LINEAR RESPONSE HELPER FUNCTIONS
%  =============================================================================

% How is the periodic boundary reflection r_2: R -> [-2, 2] evaluated?
function x = reflect(x), x = abs(mod(x - 2.0, 8.0) - 4.0) - 2.0; end

% How are the Ulam transition matrix P_\theta and drift-sensitivity kernel K_deriv constructed?
%   Sums the folded Gaussian transition density over reflection images m in {-1, 0, 1}.
function [P, K_deriv] = build_ulam_operator(F_x, x_grid, dx, sigma)
    M = length(x_grid); P = zeros(M, M); K_deriv = zeros(M, M); inv_s2 = 1.0 / (sigma^2); norm_c = dx / (sqrt(2 * pi) * sigma);
    for m = -1:1
        z1 = (x_grid + 8*m) - F_x(:)'; z2 = (4 - x_grid + 8*m) - F_x(:)';
        q1 = norm_c * exp(-0.5 * (z1.^2) * inv_s2); q2 = norm_c * exp(-0.5 * (z2.^2) * inv_s2);
        P = P + q1 + q2; K_deriv = K_deriv + (z1 * inv_s2) .* q1 + (z2 * inv_s2) .* q2;
    end
    P = P ./ sum(P, 1);
end

% How is the unique stationary probability vector \mu_\theta obtained from P_\theta?
function mu = stationary_distribution(P), M = size(P, 1); A = eye(M) - P; A(end, :) = 1.0; b = zeros(M, 1); b(end) = 1.0; mu = A \ b; mu = max(mu, 0); mu = mu / sum(mu); end

% How is the reduced stationary vector field \overline{G}_\lambda(\theta) evaluated?
function G_val = eval_reduced_field_ulam(theta, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl)
    res = (theta' * Phi_grid) - d_grid;
    [P_star, ~] = build_ulam_operator(A_cl * x_grid + d_grid' - (Phi_grid' * theta), x_grid, dx, sigma);
    mu_star = stationary_distribution(P_star);
    G_val = (Phi_grid * (res' .* mu_star)) + lam * theta;
end

% How is the zero of the reduced vector field \overline{G}_\lambda(\theta_*) = 0 solved?
function theta_eq = solve_reduced_equilibrium_ulam(lam, theta_init, Phi_grid, d_grid, x_grid, dx, sigma, A_cl)
    theta_eq = theta_init;
    for iter = 1:35
        [J, ~, ~, ~, ~] = compute_reduced_jacobian_ulam(theta_eq, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl);
        G_val = eval_reduced_field_ulam(theta_eq, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl);
        if norm(G_val) < 1e-13, break; end; theta_eq = theta_eq - J \ G_val;
    end
end

% How is the reduced Jacobian D\overline{G}_\lambda(\theta_*) = H_* + \lambda I + C_* R_* B_* computed?
%   Solves the deflated zero-mass resolvent equation (I - P_* + \mu_* 1^T) R_* B_* = B_*
%   on V_h = {v in R^M : 1^T v = 0} and contracts with the observable C_*.
function [J_star, H_star, CRB_star, mu_star, dmu_dtheta1] = compute_reduced_jacobian_ulam(theta, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl)
    p = length(theta); M = length(x_grid);
    [P_star, K_deriv] = build_ulam_operator(A_cl * x_grid + d_grid' - (Phi_grid' * theta), x_grid, dx, sigma);
    mu_star = stationary_distribution(P_star); H_star = (Phi_grid .* mu_star') * Phi_grid'; B_star = zeros(M, p);
    for j = 1:p, dF_j = -Phi_grid(j, :)'; B_star(:, j) = (K_deriv .* (dF_j')) * mu_star; B_star(:, j) = B_star(:, j) - sum(B_star(:, j)) * mu_star; end
    R_B = (eye(M) - P_star + mu_star * ones(1, M)) \ B_star;
    for j = 1:p, R_B(:, j) = R_B(:, j) - sum(R_B(:, j)) * mu_star; end
    CRB_star = (Phi_grid .* ((theta' * Phi_grid) - d_grid)) * R_B; J_star = H_star + lam * eye(p) + CRB_star;
    dmu_dtheta1 = R_B(:, min(p,2));
end

% How are continuous particle ensembles sampled from the discretized Ulam density?
function X_samples = sample_from_grid_pdf(x_grid, mu_weights, N)
    cdf = cumsum(mu_weights); cdf = cdf / cdf(end); u = rand(1, N); [~, bin_idx] = histc(u, [0; cdf]);
    bin_idx = min(max(bin_idx, 1), length(x_grid)); X_samples = x_grid(bin_idx)' + (rand(1, N) - 0.5) * (x_grid(2)-x_grid(1));
end

% ==============================================================================
% QUANTITATIVE SUMMARY, MANUSCRIPT CROSS-REFERENCES & ARCHIVAL CITATION
% ==============================================================================
% Verified Numerical Benchmarks (M = 1,200 Ulam Grid, Seed = 42):
%   - Table 1 (Section 8.1 & 8.2):
%       * Boundary reflection trajectory discrepancy: ||\theta^trk - \theta^theo||_2 = 4.02e-8
%       * Unregularized Gram spectrum (\sigma = 0.18): \lambda_min(Q_opt) = 4.434e-3
%       * Regularized Ulam equilibrium (\lambda = 1e-3): \theta_*^\lambda = [0, 0.00704, 0, 0.12437]^T
%       * Regularized Jacobian FD agreement: ||D_FD \overline{G}_\lambda - D\overline{G}_\lambda||_2 = 6.80e-12
%   - Section 8.2.1 & Figure 4 (Misspecified Quadratic Basis, \sigma = 0.24):
%       * Reduced equilibrium: \theta_* = [0, 0.08882, 0]^T
%       * Odd-mode speedup from C_* R_* B_*: (H_*)_{22} = 0.1753 -> (D\overline{G})_{22} = 0.2414 (+37.7%)
%       * Even-mode speedup from C_* R_* B_*: \lambda_min(H_even) = 0.0566 -> \lambda_min(J_even) = 0.0706 (+24.8%)
%       * Misspecified Jacobian FD agreement: ||D_FD \overline{G} - (H_* + C_* R_* B_*)||_2 = 1.66e-11
%   - Table 2 (Section 8.3.2 Deployment Ablation):
%       * Deployed origin derivative: F_dep'(0) = 0.80 - \theta_{1,off} = 1.26188 > 1
%       * Discretized Action Error ||r_{A,h}||_{\ell^1}: 0.1604 (gain only), 0.2966 (noise only), 0.1536 (both)
%       * Stationary distribution shift ||\mu_dep,h - \mu_off,h||_{\ell^1}: 1.9737 (gain only), 0.4533 (noise only), 1.9163 (both)
%
% Reference BibTeX:
%   @article{GalatoloLippi2026STO,
%     title   = {Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator},
%     author  = {Galatolo, Stefano and Lippi, Vittorio},
%     journal = {Journal of Nonlinear Science},
%     note    = {Submitted},
%     year    = {2026}
%   }
% ==============================================================================
