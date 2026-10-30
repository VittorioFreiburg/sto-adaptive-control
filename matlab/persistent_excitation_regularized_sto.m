% ==============================================================================
% PERSISTENT_EXCITATION_REGULARIZED_STO.M
%
% Paper: "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
%         A Barycentric State-Parameter Formulation"
% Paper Authors: Stefano Galatolo and Vittorio Lippi
% Code Author:   Vittorio Lippi (vittorio.lippi@calejo.ai)
% Affiliation:   Calejo Hybrid Intelligence AB
% License:       MIT License (c) 2026 Stefano Galatolo and Vittorio Lippi
%
% --- Theoretical Framework & Semantic Indexing Metadata ----------------------
% Domain: Adaptive Control, Non-Equilibrium Statistical Mechanics, Neural Control
% Keywords: Self-Consistent Transfer Operator, Identifiability Trap, Persistent Excitation,
%           Proposition 7.7 (prop:duffing-small-regularization), Hessian Spectrum lambda_min(H_*),
%           Covariance Degeneracy, Closed-Loop Learning, Drift Return Map,
%           Ensemble Monte Carlo, Invariant Probability Measure, Duffing Plant.
%
% Mathematical Core & System Specification:
%   - Plant: Reflected Duffing Oscillator: x_{n+1} = r_2(1.05 x_n + 0.15 x_n^3 - 0.25 x_n - u_lin + xi_n)
%   - Regimes Evaluated:
%       1. Nominal Autonomous Adaptation: sigma = 0.18, lambda in {0, 1e-6, 1e-3}
%       2. Under-Excited Identifiability Trap: sigma = 0.10 (state collapses to narrow variance,
%          causing lambda_min(H_*) to plunge by 28x, trapping adaptation in a non-identifiable manifold)
%       3. Persistent Excitation: sigma = 0.18 (stochastic excitation restores global support,
%          enforcing positive definiteness of H_* and exact parameter recovery)
%
% Output Artifacts:
%   - Data:    data/data_part1_nominal.mat
%   - Figures: figures/fig_part1_nominal_convergence.pdf, figures/fig_part2_identifiability_trap_pe.pdf
%   - Metrics: results/persistent_excitation_metrics.log
% ==============================================================================
addpath(fileparts(mfilename('fullpath'))); paths = sto_paths();  % repository folders (data/, figures/, results/)
rng(42);
set(groot, 'DefaultFigureVisible', 'off');

N = 100000;
N_steps = 40000; % At x_ref=0 and sigma=0.18, 1/(delta*lambda_min) is ~2,300 steps (40k resolves full floor!)
delta = 0.10;
A_plant = 1.05;
B_true = 0.15;
x_ref = 0.0;
theta_star = [0; 0; 0; 0.15];
theta_init = [-0.25; 0.45; 0; 0];
calc_u_loose = @(x) -0.25 * (x - x_ref);

% --- Run 1: Persistent Excitation (sigma = 0.18) with lambda = {0, 1e-6, 1e-3} ---
sigma_pe = 0.18;
lambdas = [0, 1e-6, 1e-3];
E_pe = zeros(length(lambdas), N_steps);
theta_hist_lam3 = zeros(4, N_steps);

for l_idx = 1:length(lambdas)
    rng(42);
    lam = lambdas(l_idx);
    X = x_ref * ones(1, N);
    theta = theta_init;
    for n = 1:N_steps
        noise = sigma_pe * randn(1, N);
        phi = [ones(1, N); X; X.^2; X.^3];
        X_next = reflect(A_plant*X + B_true*X.^3 + calc_u_loose(X) - theta'*phi + noise);
        err_bp = -(X_next - 0.8*X);
        G = (phi * err_bp' / N) + lam * theta;
        theta = theta - delta * G;
        E_pe(l_idx, n) = norm(theta - theta_star);
        if l_idx == 3
            theta_hist_lam3(:, n) = theta;
        end
        X = X_next;
    end
    theta_final_pe(:, l_idx) = theta;
    if l_idx == 1
        mu_pe = X;
        H_pe = (phi * phi') / N;
    end
end
theta_star_lambda = mean(theta_hist_lam3(:, end-4999:end), 2);
E_pe_reg_star = sqrt(sum((theta_hist_lam3 - theta_star_lambda).^2, 1));

% --- Run 2: Under-Excited (sigma = 0.10, lambda = 0) ---
rng(42);
sigma_low = 0.10;
X_low = x_ref * ones(1, N);
theta_low = theta_init;
E_low = zeros(1, N_steps);
for n = 1:N_steps
    noise = sigma_low * randn(1, N);
    phi = [ones(1, N); X_low; X_low.^2; X_low.^3];
    X_next = reflect(A_plant*X_low + B_true*X_low.^3 + calc_u_loose(X_low) - theta_low'*phi + noise);
    err_bp = -(X_next - 0.8*X_low);
    G = (phi * err_bp' / N);
    theta_low = theta_low - delta * G;
    E_low(n) = norm(theta_low - theta_star);
    X_low = X_next;
end
mu_low = X_low;
H_low = (phi * phi') / N;

fprintf('x_ref = 0 Spectral Summary:\n');
fprintf('  sigma = 0.18: lambda_min(H_*) = %.4e, lambda_max(H_*) = %.4f\n', min(eig(H_pe)), max(eig(H_pe)));
fprintf('  sigma = 0.10: lambda_min(H_*) = %.4e, lambda_max(H_*) = %.4f\n', min(eig(H_low)), max(eig(H_low)));

% Also write results/persistent_excitation_metrics.log
fid_log = fopen(fullfile(paths.results, 'persistent_excitation_metrics.log'), 'w');
fprintf(fid_log, '=== Persistent Excitation Metrics ===\n');
fprintf(fid_log, '1. Spectral Properties:\n');
fprintf(fid_log, '   - lambda_min(H_*) [Local, sigma=0.10]: %.5e\n', min(eig(H_low)));
fprintf(fid_log, '   - lambda_min(H_*) [Global, sigma=0.18]: %.5e\n', min(eig(H_pe)));
fprintf(fid_log, '   - Increase Factor: %.2f\n', min(eig(H_pe)) / min(eig(H_low)));
fclose(fid_log);

save(fullfile(paths.data, 'data_part1_nominal.mat'), 'E_pe', 'theta_final_pe', 'mu_pe', 'H_pe', 'E_low', 'mu_low', 'H_low', 'lambdas', 'theta_star', 'theta_star_lambda', 'E_pe_reg_star', 'x_ref', 'N_steps');

% --- Plot Fig 2: figures/fig_part1_nominal_convergence.pdf ---
if usejava('desktop')
    x_eval = linspace(-2.2, 2.2, 500);
    phi_eval = [ones(1, 500); x_eval; x_eval.^2; x_eval.^3];

    fig1 = figure('Position', [100, 100, 1100, 400]);
    tiledlayout(1, 2, 'TileSpacing', 'Compact');

    nexttile; hold on; box on;
    semilogy(1:N_steps, E_pe(1, :), 'r-', 'LineWidth', 1.8, 'DisplayName', 'Unregularized to \theta_{opt} (\lambda = 0)');
    semilogy(1:N_steps, E_pe(2, :), 'b:', 'LineWidth', 2.0, 'DisplayName', 'Tiny Reg. to \theta_{opt} (\lambda = 10^{-6}, Prop. 7.7)');
    semilogy(1:N_steps, E_pe(3, :), 'Color', [0.93 0.69 0.13], 'LineStyle', '--', 'LineWidth', 1.8, 'DisplayName', 'Regularized to \theta_{opt} (\lambda = 10^{-3}, Prop. 7.7 Bias)');
    semilogy(1:N_steps, E_pe_reg_star, 'Color', [0.00 0.45 0.74], 'LineStyle', '-.', 'LineWidth', 2.0, 'DisplayName', 'Regularized to \theta_*^\lambda (\lambda = 10^{-3}, Thm. 7.4)');
    set(gca, 'YScale', 'log'); grid on;
    xlabel('Adaptation Steps (n)'); ylabel('Parameter Error Norm');
    title('Parameter Convergence (\sigma = 0.18, x_{ref} = 0)');
    legend('Location', 'northeast');
    ylim([1e-5, 1e0]);

    nexttile; hold on; box on;
    plot(x_eval, 0.8*x_eval, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Target Drift (0.8x)');
    plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - theta_final_pe(:,1)'*phi_eval, 'r--', 'LineWidth', 1.8, 'DisplayName', 'Unregularized (\lambda = 0)');
    plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - theta_final_pe(:,2)'*phi_eval, 'b:', 'LineWidth', 2.0, 'DisplayName', 'Tiny Reg. (\lambda = 10^{-6}, Prop. 7.7)');
    plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - theta_final_pe(:,3)'*phi_eval, 'Color', [0.93 0.69 0.13], 'LineStyle', '-.', 'LineWidth', 1.8, 'DisplayName', 'Regularized (\lambda = 10^{-3})');
    grid on; xlim([-2.2, 2.2]);
    xlabel('State x_n'); ylabel('Deterministic Drift F_\theta(x_n)');
    title('Closed-Loop Drift Return Map');
    legend('Location', 'northwest');

    exportgraphics(fig1, fullfile(paths.figures, 'fig_part1_nominal_convergence.pdf'), 'ContentType', 'vector');
end

% --- Plot Fig 5: figures/fig_part2_identifiability_trap_pe.pdf ---
fig2 = figure('Position', [100, 100, 1500, 400], 'Visible', 'off');
tiledlayout(1, 3, 'TileSpacing', 'Compact');

nexttile; hold on; box on;
histogram(mu_low, 120, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', 'b', 'DisplayName', '\sigma = 0.10 (Under-excited)');
histogram(mu_pe, 120, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', 'r', 'DisplayName', '\sigma = 0.18 (Persistently excited)');
xline(0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
xlim([-2, 2]); grid on; xlabel('State x'); ylabel('Density \mu_*(x)');
title('Stationary Physical Measure \mu_*(x)'); legend('Location', 'northwest');

nexttile; hold on; box on;
semilogy(1:N_steps, E_low, 'b-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.10 (Stalled)');
semilogy(1:N_steps, E_pe(1, :), 'r-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.18 (Convergent)');
set(gca, 'YScale', 'log'); grid on;
xlabel('Adaptation Steps (n)'); ylabel('||\theta_n - \theta_{opt}||');
title('Identifiability Trap vs Persistent Excitation'); legend('Location', 'northeast');

nexttile; hold on; box on;
plot(x_eval, 0.8*x_eval, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Target Drift (0.8x)');
plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - theta_low'*phi_eval, 'b-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.10 Fit');
plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - theta_final_pe(:,1)'*phi_eval, 'r-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.18 Fit');
xlim([-2.2, 2.2]); ylim([-2.5, 2.5]); grid on;
xlabel('State x_n'); ylabel('Drift F_\theta(x_n)');
title('Closed-Loop Drift Return Map'); legend('Location', 'northwest');

exportgraphics(fig2, fullfile(paths.figures, 'fig_part2_identifiability_trap_pe.pdf'), 'ContentType', 'vector');

function x = reflect(x)
    x = abs(mod(x - 2.0, 8.0) - 4.0) - 2.0;
end
