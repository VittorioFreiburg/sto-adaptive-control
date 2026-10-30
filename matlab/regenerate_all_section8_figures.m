% ==============================================================================
% REGENERATE_ALL_SECTION8_FIGURES.M
%
% Fast Vector Figure Regeneration from Precomputed Datasets and Ulam Operators
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
% SCIENTIFIC QUESTIONS VISUALIZED BY THIS SCRIPT
% ==============================================================================
%   - Figure 4 (figures/fig_linear_response_contribution.pdf, Section 8.2.1):
%     How much does the linear-response measure sensitivity C_* R_* B_* accelerate
%     odd-mode and even-mode parameter relaxation compared to the static Hessian H_*?
%
%   - Figure 5 (figures/fig_part2_identifiability_trap_pe.pdf, Section 8.3.1):
%     How does persistent stochastic excitation (\sigma = 0.18) overcome the
%     weak-excitation identifiability trap (\sigma = 0.10)?
%
%   - Figure 6 (figures/fig_part2_action_error_recovery.pdf, Section 8.3.2):
%     How does online STO adaptation recover from the two-phase deployment
%     mismatch and repair the closed-loop drift return map?
%
%   - Figure 7 (figures/fig_part2_escaping_trap_offline.pdf, Section 8.3.2):
%     How does matching the offline baseline gain to the deployment gain (-0.25 x)
%     eliminate the deployment Action Error prior to online learning?
%
%   - Figures 8 & 9 (figures/fig_part2_bifurcation_diagram.pdf & figures/fig_part2_density_shattering.pdf, Section 8.3.3):
%     How does the physical state distribution transition from damped alternating
%     concentration (\delta = 1.85) to macroscopic dispersion (\delta = 2.22)
%     across the linear stability threshold \delta_c \approx 2.00?
% ==============================================================================
function regenerate_all_section8_figures()
    addpath(fileparts(mfilename('fullpath'))); paths = sto_paths();  % repository folders (data/, figures/, results/)
    set(groot, 'DefaultFigureVisible', 'off');
    x_eval = linspace(-2.2, 2.2, 500);
    phi_eval = [ones(1, 500); x_eval; x_eval.^2; x_eval.^3];

    %% 1. FIGURE 5 (Section 8.3.1): How does persistent excitation (\sigma = 0.18) restore cubic identifiability?
    fprintf('Regenerating Fig 5 (fig_part2_identifiability_trap_pe.pdf)...\n');
    d1 = load(fullfile(paths.data, 'data_part1_nominal.mat'));
    % Recover theta_low at sigma=0.10 from final fit if not saved directly
    if ~isfield(d1, 'theta_low')
        rng(42); N = 100000; X_low = zeros(1, N); theta_low = [-0.25; 0.45; 0; 0];
        for n = 1:d1.N_steps
            noise = 0.10 * randn(1, N); phi = [ones(1, N); X_low; X_low.^2; X_low.^3];
            X_next = reflect(0.80*X_low + 0.15*X_low.^3 - theta_low'*phi + noise);
            G = (phi * (-(X_next - 0.8*X_low))') / N;
            theta_low = theta_low - 0.10 * G; X_low = X_next;
        end
    else
        theta_low = d1.theta_low;
    end
    fig5 = figure('Position', [100, 100, 1500, 400], 'Visible', 'off');
    tiledlayout(1, 3, 'TileSpacing', 'Compact');
    nexttile; hold on; box on;
    histogram(d1.mu_low, 120, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', 'b', 'DisplayName', '\sigma = 0.10 (Under-excited)');
    histogram(d1.mu_pe, 120, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', 'r', 'DisplayName', '\sigma = 0.18 (Persistently excited)');
    xline(0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    xlim([-2, 2]); grid on; xlabel('State x'); ylabel('Density \mu_*(x)');
    title('Stationary Physical Measure \mu_*(x)'); legend('Location', 'northwest');
    nexttile; hold on; box on;
    semilogy(1:d1.N_steps, d1.E_low, 'b-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.10 (Stalled)');
    semilogy(1:d1.N_steps, d1.E_pe(1, :), 'r-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.18 (Convergent)');
    set(gca, 'YScale', 'log'); grid on;
    xlabel('Adaptation Steps (n)'); ylabel('||\theta_n - \theta_{opt}||');
    title('Weak Excitation vs Persistent Excitation'); legend('Location', 'northeast');
    nexttile; hold on; box on;
    plot(x_eval, 0.8*x_eval, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Target Drift (0.8x)');
    plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - theta_low'*phi_eval, 'b-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.10 Fit');
    plot(x_eval, 0.8*x_eval + 0.15*x_eval.^3 - d1.theta_final_pe(:,1)'*phi_eval, 'r-', 'LineWidth', 1.8, 'DisplayName', '\sigma = 0.18 Fit');
    xlim([-2.2, 2.2]); ylim([-2.5, 2.5]); grid on;
    xlabel('State x_n'); ylabel('Drift F_\theta(x_n)');
    title('Closed-Loop Drift Return Map'); legend('Location', 'northwest');
    exportgraphics(fig5, fullfile(paths.figures, 'fig_part2_identifiability_trap_pe.pdf'), 'ContentType', 'vector');
    close(fig5);

    %% 2. FIGURES 6 & 7 (Section 8.3.2): How is deployment Action Error repaired online or prevented via gain matching?
    fprintf('Regenerating Figs 6 & 7 (from data_validate_theoretical.mat)...\n');
    dv = load(fullfile(paths.data, 'data_validate_theoretical.mat'));
    N_steps_adapt = length(dv.E_T_opt);
    fig6 = figure('Position', [100, 100, 1500, 400], 'Visible', 'off');
    tiledlayout(1, 3, 'TileSpacing', 'Compact');
    nexttile; hold on; box on;
    histogram(dv.mu_off, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.6 0.6 0.6]);
    histogram(dv.mu_deploy, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2.0, 'EdgeColor', 'r');
    histogram(dv.X_opt, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2.0, 'EdgeColor', 'b');
    xline(dv.x_ref, 'k--', 'LineWidth', 1.5);
    title('State Distribution Dispersion');
    xlabel('State X'); ylabel('Density');
    legend('\mu_{off} (Offline)', '\mu_{deploy} (Dispersed)', '\mu_* (Online STO)', 'Location', 'northwest');
    xlim([-2.2, 2.2]); grid on;

    nexttile; box on;
    yyaxis left;
    plot(1:N_steps_adapt, dv.E_T_opt, '-', 'LineWidth', 2.0, 'DisplayName', 'Parameter Error $\|\theta_n - \theta_{\rm opt}\|$');
    ylabel('Parameter Error $\|\theta_n - \theta_{\rm opt}\|$', 'Interpreter', 'latex');
    ylim([0, max(dv.E_T_opt)*1.1]);
    yyaxis right;
    plot(1:N_steps_adapt, dv.Action_Err, '-', 'LineWidth', 2.0, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Compensation error $\mathcal{E}_A(\theta_n)$');
    ylabel('Compensation Error $\mathcal{E}_A(\theta_n) = \|g(\cdot,\theta_n) - d\|_{L^2(\mu_n)}$', 'Interpreter', 'latex');
    ylim([0, max(dv.Action_Err)*1.1]);
    title('Online Recovery from Deployment Mismatch', 'Interpreter', 'latex');
    legend('Location', 'northeast', 'Interpreter', 'latex');
    xlabel('Adaptation Steps ($n$)', 'Interpreter', 'latex'); grid on;

    nexttile; hold on; box on;
    plot(x_eval, x_eval, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, 'DisplayName', 'Stability Boundary (y=x)');
    plot(x_eval, 0.8*(x_eval-dv.x_ref) + dv.x_ref, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Target Drift');
    drift_deploy = 0.80*x_eval + 0.15*x_eval.^3 - (dv.theta_off'*phi_eval);
    plot(x_eval, drift_deploy, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Deployed (Dispersed)');
    drift_opt = 0.80*x_eval + 0.15*x_eval.^3 - (dv.theta_opt_final'*phi_eval);
    plot(x_eval, drift_opt, 'b-', 'LineWidth', 2.0, 'DisplayName', 'Online STO');
    title('System Dynamics: Drift Return Map');
    xlabel('State X_n'); ylabel('Expected X_{n+1}');
    legend('Location', 'northwest');
    xlim([-2.2, 2.2]); ylim([-2.5, 2.5]); grid on;
    exportgraphics(fig6, fullfile(paths.figures, 'fig_part2_action_error_recovery.pdf'), 'ContentType', 'vector');
    close(fig6);

    fig7 = figure('Position', [100, 100, 600, 400], 'Visible', 'off');
    hold on; box on;
    histogram(dv.mu_off, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.75 0.75 0.75]);
    histogram(dv.mu_deploy, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', [0.85 0.2 0.2], 'LineStyle', ':');
    histogram(dv.mu_off_loose, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2.8, 'EdgeColor', [0.2 0.2 0.2]);
    histogram(dv.mu_deploy_loose, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.4, 'EdgeColor', [0 0.4470 0.7410], 'LineStyle', '--');
    title('Eliminating Deployment Mismatch via Gain Matching');
    xlabel('State X'); ylabel('Density');
    legend('\mu_{off} (Tight, gain = -0.80)', '\mu_{deploy} (Dispersed, gain = -0.25)', '\mu_{off} (Loose, gain = -0.25)', '\mu_{deploy} (Stable, gain = -0.25)', 'Location', 'northwest', 'FontSize', 8.5);
    xlim([-2.2, 2.2]); grid on;
    exportgraphics(fig7, fullfile(paths.figures, 'fig_part2_escaping_trap_offline.pdf'), 'ContentType', 'vector');
    close(fig7);

    %% 3. FIGURES 8 & 9 (Section 8.3.3): What happens when \delta crosses the linear stability threshold \delta_c \approx 2.00?
    fprintf('Regenerating Figs 8 & 9 (from data_bifurcation_sweep.mat)...\n');
    db = load(fullfile(paths.data, 'data_bifurcation_sweep.mat'));
    fig8 = figure('Position', [100, 100, 650, 420], 'Visible', 'off');
    hold on; box on;
    scatter(db.bifurcation_data_delta, db.bifurcation_data_state, 1, 'k', '.', 'HandleVisibility', 'off');
    xline(db.delta_stable, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\delta = %.2f (Damped alternating)', db.delta_stable));
    xline(db.delta_chaos, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\delta = %.2f (Dispersed)', db.delta_chaos));
    xline(2.000, 'm:', 'LineWidth', 1.5, 'DisplayName', ['Theoretical \delta_c = 2/\lambda_{max}(H_*) ' char(8776) ' 2.000']);
    yline(db.x_ref, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Target Setpoint x_{ref} = 0');
    title('Bifurcation Diagram: Physical State X vs Learning Rate \delta', 'Interpreter', 'tex');
    xlabel('Learning Rate \delta'); ylabel('Asymptotic Physical State X');
    legend('Location', 'southwest'); ylim([-2.2, 2.2]); xlim([1.50, 2.25]); grid on;
    exportgraphics(fig8, fullfile(paths.figures, 'fig_part2_bifurcation_diagram.pdf'), 'ContentType', 'vector');
    close(fig8);

    fig9 = figure('Position', [100, 100, 1100, 420], 'Visible', 'off');
    tiledlayout(1, 2, 'TileSpacing', 'Compact');
    n_c = 256;
    colormap_array = [linspace(0, 1, n_c)', zeros(n_c, 1), zeros(n_c, 1)];
    colormap_array(128:end, 2) = linspace(0, 1, 129)';
    colormap_array(192:end, 3) = linspace(0, 1, 65)';
    c_max = max(max(db.P_den_stable(:).^0.1), max(db.P_den_chaos(:).^0.1));
    nexttile;
    imagesc([1, db.N_steps_adapt], [-2, 2], db.P_den_stable.^0.1);
    colormap(gca, colormap_array); set(gca, 'YDir', 'normal'); clim([0, c_max]); colorbar;
    yline(0, '--w', 'LineWidth', 1.2);
    title(sprintf('Damped Alternating Concentration (\\delta = %.2f)', db.delta_stable));
    xlabel('Adaptation Steps (n)'); ylabel('State X');
    nexttile;
    imagesc([1, db.N_steps_adapt], [-2, 2], db.P_den_chaos.^0.1);
    colormap(gca, colormap_array); set(gca, 'YDir', 'normal'); clim([0, c_max]); colorbar;
    yline(0, '--w', 'LineWidth', 1.2);
    title(sprintf('Density Dispersion (\\delta = %.2f)', db.delta_chaos));
    xlabel('Adaptation Steps (n)'); ylabel('State X');
    exportgraphics(fig9, fullfile(paths.figures, 'fig_part2_density_shattering.pdf'), 'ContentType', 'vector');
    close(fig9);

    %% 4. FIGURE 4 (Section 8.2.1): How does the linear-response term C_* R_* B_* modify relaxation rates?
    fprintf('Regenerating Fig 4 (fig_linear_response_contribution.pdf)...\n');
    if isfile(fullfile(paths.data, 'data_fig4_linear_response.mat'))
        load(fullfile(paths.data, 'data_fig4_linear_response.mat'));
    else
        A_cl = 0.80; B_true = 0.15; d_func = @(x) B_true * (x.^3);
        M_grid = 1200; edges = linspace(-2, 2, M_grid + 1); dx = edges(2) - edges(1);
        x_grid = 0.5 * (edges(1:end-1) + edges(2:end))';
        d_grid = d_func(x_grid)';
        sigma_lr = 0.24; Phi_grid_lr = [ones(1, M_grid); x_grid'; (x_grid.^2)'];
        theta_star_lr = solve_reduced_equilibrium_ulam(0.0, [0; 0.08; 0], Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);
        [J_lr, H_lr, ~, mu_star_lr, dmu_dtheta1] = compute_reduced_jacobian_ulam(theta_star_lr, 0.0, Phi_grid_lr, d_grid, x_grid, dx, sigma_lr, A_cl);

        delta_lr = 0.05; N_steps_lr = 800; N_ens = 100000; dtheta_odd_0 = [0; 0.12; 0];
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
        steps_vec = 0:(N_steps_lr - 1);
        rate_H_odd = H_lr(2, 2); rate_J_odd = J_lr(2, 2);
        rate_H_ev = min(eig(H_lr([1,3],[1,3]))); rate_J_ev = min(eig(J_lr([1,3],[1,3])));
    end

    fig4 = figure('Position', [100, 100, 1450, 420], 'Visible', 'off');
    tiledlayout(1, 3, 'TileSpacing', 'Compact');
    nexttile; hold on; box on;
    steps_vec = 0:(N_steps_lr - 1);
    rate_H_odd = H_lr(2, 2); rate_J_odd = J_lr(2, 2);
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

    nexttile; hold on; box on;
    rate_H_ev = min(eig(H_lr([1,3],[1,3]))); rate_J_ev = min(eig(J_lr([1,3],[1,3])));
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
    fprintf('All Section 8 figures successfully regenerated!\n');
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

% ==============================================================================
% MANUSCRIPT CROSS-REFERENCES & ARCHIVAL CITATION
% ==============================================================================
% Figure-to-Subsection Mapping in Empirical_test.tex (Section 8):
%   - Figure 4: figures/fig_linear_response_contribution.pdf  (Section 8.2.1)
%   - Figure 5: figures/fig_part2_identifiability_trap_pe.pdf (Section 8.3.1)
%   - Figure 6: figures/fig_part2_action_error_recovery.pdf   (Section 8.3.2)
%   - Figure 7: figures/fig_part2_escaping_trap_offline.pdf   (Section 8.3.2)
%   - Figure 8: figures/fig_part2_bifurcation_diagram.pdf     (Section 8.3.3)
%   - Figure 9: figures/fig_part2_density_shattering.pdf      (Section 8.3.3)
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
