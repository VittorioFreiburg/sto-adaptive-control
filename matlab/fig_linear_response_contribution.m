% ==============================================================================
% FIG_LINEAR_RESPONSE_CONTRIBUTION.M
%
% Standalone Generator for Figure 4:
% Effect of the statistical linear-response term C_* R_* B_* on local relaxation rates
% ==============================================================================
%
% MANUSCRIPT:
%   "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
%    A Barycentric State-Parameter Formulation" (Section 8.2.1, Figure 4)
%
% AUTHORS:
%   Stefano Galatolo (1) and Vittorio Lippi (2)
%   (1) Dipartimento di Matematica, Universita di Pisa, Pisa, Italy
%       Email: stefano.galatolo@unipi.it
%   (2) Calejo Hybrid Intelligence AB, Stockholm, Sweden
%       Email: vittorio.lippi@calejo.ai
%
% Output Artifact:
%   - figures/fig_linear_response_contribution.pdf
%   - data/data_fig4_linear_response.mat
% ==============================================================================

function fig_linear_response_contribution()
    addpath(fileparts(mfilename('fullpath'))); paths = sto_paths();  % repository folders (data/, figures/, results/)
    fprintf('=== Generating Figure 4 (fig_linear_response_contribution.pdf) ===\n');
    t_start = tic;

    % Physical plant & Ulam-Galerkin parameters
    A_cl = 0.80; B_true = 0.15; d_func = @(x) B_true * (x.^3);
    M_grid = 1200; edges = linspace(-2, 2, M_grid + 1); dx = edges(2) - edges(1);
    x_grid = 0.5 * (edges(1:end-1) + edges(2:end))';
    d_grid = d_func(x_grid)';
    sigma_lr = 0.24; Phi_grid_lr = [ones(1, M_grid); x_grid'; (x_grid.^2)'];

    fprintf('  Solving misspecified equilibrium theta_* via Newton on Ulam operator...\n');
    theta_star_lr = solve_reduced_equilibrium_ulam(0.0, [0; 0.08; 0], Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);
    
    fprintf('  Computing analytical reduced Jacobian J_* = H_* + C_* R_* B_*...\n');
    [J_lr, H_lr, ~, mu_star_lr, dmu_dtheta1] = compute_reduced_jacobian_ulam(theta_star_lr, 0.0, Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);

    delta_lr = 0.05; N_steps_lr = 800; N_ens = 100000; dtheta_odd_0 = [0; 0.12; 0];
    theta_u_odd = theta_star_lr + dtheta_odd_0; mu_u_odd = mu_star_lr; err_ulam_odd = zeros(1, N_steps_lr);

    fprintf('  Simulating odd-mode relaxation (Ulam STO orbit & N=10^5 particle ensemble)...\n');
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

    fprintf('  Simulating slowest even-mode relaxation (Ulam STO orbit)...\n');
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

    % Save computed data for Python vector renderer / offline plotting
    steps_vec = 0:(N_steps_lr - 1);
    rate_H_odd = H_lr(2, 2); rate_J_odd = J_lr(2, 2);
    rate_H_ev = min(eig(H_lr([1,3],[1,3]))); rate_J_ev = min(eig(J_lr([1,3],[1,3])));
    save(fullfile(paths.data, 'data_fig4_linear_response.mat'), 'steps_vec', 'rate_H_odd', 'rate_J_odd', ...
        'dtheta_odd_0', 'err_ulam_odd', 'err_part_odd', 'v_slow_even', 'rate_H_ev', ...
        'rate_J_ev', 'err_ulam_even', 'x_grid', 'dx', 'mu_star_lr', 'dmu_dtheta1', ...
        'delta_lr', 'N_steps_lr');
    fprintf('  Saved data_fig4_linear_response.mat\n');

    % Render Figure 4
    fig4 = figure('Position', [100, 100, 1450, 420]);
    tiledlayout(1, 3, 'TileSpacing', 'Compact');

    % Panel (a): Odd Mode
    nexttile; hold on; box on;
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

    % Panel (b): Even Mode
    nexttile; hold on; box on;
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

    % Panel (c): Stationary Density & Linear Response
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

    exportgraphics(fig4, fullfile(paths.figures, 'fig_linear_response_contribution.pdf'), 'ContentType', 'vector');
    close(fig4);
    fprintf('  Exported fig_linear_response_contribution.pdf (elapsed time: %.1fs)\n', toc(t_start));
end

function x = reflect(x), x = abs(mod(x - 2.0, 8.0) - 4.0) - 2.0; end
function [P, K_deriv] = build_ulam_operator(F_x, x_grid, dx, sigma)
    M = length(x_grid); P = zeros(M, M); K_deriv = zeros(M, M); inv_s2 = 1.0 / (sigma^2); norm_c = dx / (sqrt(2 * pi) * sigma);
    for m = -1:1
        z1 = (x_grid + 8*m) - F_x(:)'; z2 = (4 - x_grid + 8*m) - F_x(:)';
        q1 = norm_c * exp(-0.5 * (z1.^2) * inv_s2); q2 = norm_c * exp(-0.5 * (z2.^2) * inv_s2);
        P = P + q1 + q2; K_deriv = K_deriv + (z1 * inv_s2) .* q1 + (z2 * inv_s2) .* q2;
    end
    P = P ./ sum(P, 1);
end
function mu = stationary_distribution(P), M = size(P, 1); A = eye(M) - P; A(end, :) = 1.0; b = zeros(M, 1); b(end) = 1.0; mu = A \ b; mu = max(mu, 0); mu = mu / sum(mu); end
function theta_eq = solve_reduced_equilibrium_ulam(lam, theta_init, Phi_grid, d_grid, x_grid, dx, sigma, A_cl)
    theta_eq = theta_init;
    for iter = 1:35
        [J, ~, ~, ~, ~] = compute_reduced_jacobian_ulam(theta_eq, lam, Phi_grid, d_grid, x_grid, dx, sigma, A_cl);
        res = (theta_eq' * Phi_grid) - d_grid; [P_star, ~] = build_ulam_operator(A_cl * x_grid + d_grid' - (Phi_grid' * theta_eq), x_grid, dx, sigma);
        mu_star = stationary_distribution(P_star); G_val = (Phi_grid * (res' .* mu_star)) + lam * theta_eq;
        if norm(G_val) < 1e-13, break; end; theta_eq = theta_eq - J \ G_val;
    end
end
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
function X_samples = sample_from_grid_pdf(x_grid, mu_weights, N)
    cdf = cumsum(mu_weights); cdf = cdf / cdf(end); u = rand(1, N); [~, bin_idx] = histc(u, [0; cdf]);
    bin_idx = min(max(bin_idx, 1), length(x_grid)); X_samples = x_grid(bin_idx)' + (rand(1, N) - 0.5) * (x_grid(2)-x_grid(1));
end
