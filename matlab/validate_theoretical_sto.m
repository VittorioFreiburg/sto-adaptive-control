% ==============================================================================
% VALIDATE_THEORETICAL_STO.M
%
% Paper: "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
%         A Barycentric State-Parameter Formulation"
% Paper Authors: Stefano Galatolo and Vittorio Lippi
% Code Author:   Vittorio Lippi (vittorio.lippi@calejo.ai)
% Affiliation:   Calejo Hybrid Intelligence AB
% License:       MIT License (c) 2026 Stefano Galatolo and Vittorio Lippi
%
% --- Theoretical Framework & Semantic Indexing Metadata ----------------------
% Domain: Adaptive Control, Performative Prediction, Transfer Operator Theory
% Keywords: Action Error, Two-Phase Deployment, Distributional Shift, Invariance Residual,
%           Identifiability Trap, Baseline Gain Matching, Closed-Loop Identifiability,
%           Barycentric Formulation, Monokinetic Reduction, Drift Return Map,
%           Reflected Duffing Plant, Ensemble Particle Simulation.
%
% Mathematical Core & System Specification:
%   - Plant: Reflected Duffing Dynamics: x_{n+1} = r_2(1.05 x_n + 0.15 x_n^3 + u_n + xi_n)
%   - Offline Phase: High authority baseline u_base(x) = -0.80 x -> identifies theta_off
%   - Deployment Transition: Baseline relaxes to u_base(x) = -0.25 x -> creates Action Error
%     Invariance residual: R_A = P_{theta_off} mu_off - mu_off != 0
%   - Online STO Phase: Self-consistent adaptation Phi_delta drives theta_n -> theta_opt,
%     decaying the Action Error int (g_opt - g_n)^2 dmu_n and restoring stability.
%
% Output Artifacts:
%   - Data:    data/data_validate_theoretical.mat
%   - Figures: figures/fig_part2_action_error_recovery.pdf, figures/fig_part2_escaping_trap_offline.pdf
%   - Metrics: results/validate_theoretical_metrics.log
%
% NOTE: The figures written here are quick-look versions. The publication versions
%       used in the paper are (re)rendered from the saved .mat data by
%       regenerate_all_section8_figures.m (and python/plot_theorem74.py for Figure 3),
%       which reproduce_all.m always runs last.
% ==============================================================================
    addpath(fileparts(mfilename('fullpath'))); paths = sto_paths();  % repository folders (data/, figures/, results/)
    rng(42);

    % --- System Hyperparameters ---
    N = 100000;              
    N_steps_adapt = 6000;     
    delta_nom = 0.04; 
    sigma_online = 0.18;
    
    A_plant = 1.05;          
    B_true = 0.15;           
    noise_std = 0.10;        
    x_ref = 0.0;             
    
    theta_star = [0; 0; 0; 0.15];
    num_params = 4;
    
    calc_u_tight = @(x) -0.80 * (x - x_ref); 
    calc_u_loose = @(x) -0.25 * (x - x_ref); 

    %% PHASE 1: Identifiability Trap & Action Error
    fprintf('Phase 1: Simulating Identifiability Trap & Action Error Recovery...\n');
    
    % Offline Calibration (Tight Bounds)
    X_off = x_ref * ones(1, N);
    theta_off = [0; -0.45; 0; 0.28];
    delta_off = 0.002;
    for n = 1:3000
        noise = noise_std * randn(1, N);
        phi_off = [ones(1, N); X_off; X_off.^2; X_off.^3];
        u_lin = theta_off' * phi_off;
        
        x_target = 0.8 * (X_off - x_ref) + x_ref; 
        X_next = reflect(A_plant*X_off + B_true*X_off.^3 + calc_u_tight(X_off) - u_lin + noise);
        
        err_bp = -(X_next - x_target);
        G_off = phi_off * err_bp' / N;
        theta_off = theta_off - delta_off * G_off;
        X_off = X_next;
    end
    mu_off = X_off;
    
    % Deployment & Shattering (Using loose controller + theta_off)
    X_dep = x_ref * ones(1, N);
    for n = 1:1000
        noise = noise_std * randn(1, N);
        phi_dep = [ones(1, N); X_dep; X_dep.^2; X_dep.^3];
        u_lin = theta_off' * phi_dep;
        X_dep = reflect(A_plant*X_dep + B_true*X_dep.^3 + calc_u_loose(X_dep) - u_lin + noise);
    end
    mu_deploy = X_dep;
    
    % Online STO Adaptation
    E_T_opt    = zeros(1, N_steps_adapt);
    Action_Err = zeros(1, N_steps_adapt);
    
    X_opt = mu_deploy;  
    theta_opt = theta_off; 
    
    x_eval = linspace(-2.2, 2.2, 500);
    phi_eval = [ones(1, 500); x_eval; x_eval.^2; x_eval.^3];
    u_star_eval = theta_star' * phi_eval;
    
    for n = 1:N_steps_adapt
        noise_opt  = sigma_online * randn(1, N);
        phi_opt = [ones(1, N); X_opt; X_opt.^2; X_opt.^3];
        u_lin_opt = theta_opt' * phi_opt;
        
        x_target = 0.8 * (X_opt - x_ref) + x_ref;
        X_opt_next = reflect(A_plant*X_opt + B_true*X_opt.^3 + calc_u_loose(X_opt) - u_lin_opt + noise_opt);
        
        err_bp = -(X_opt_next - x_target); 
        G_opt = phi_opt * err_bp' / N;
        
        theta_opt = theta_opt - delta_nom * G_opt;
        
        E_T_opt(n) = norm(theta_opt - theta_star);
        u_star_opt = B_true * X_opt.^3;
        Action_Err(n) = sqrt(mean((u_lin_opt - u_star_opt).^2));
        
        X_opt = X_opt_next;
    end
    theta_opt_final = theta_opt;

    %% EXPERIMENT 2: Time-Scale Separation (Part 1, Figure 3 Left: Theorem 7.4)
    fprintf('Running Exp 2: Time-Scale Separation limits (sigma = 0.18, lambda = 1e-3)...\n');
    sigma_ts = 0.18;
    deltas = [0.005, 0.02, 0.05, 0.08, 2.02];
    N_steps_delta = 15000;
    lambda_ts = 1e-3;
    E_T_deltas = zeros(length(deltas), N_steps_delta);
    theta_init_ts = [-0.25; 0.45; 0; 0];
    
    % Obtain theta_*^lambda (from data/data_fast_theorem74.mat if available, otherwise initialized)
    if exist(fullfile(paths.data, 'data_fast_theorem74.mat'), 'file')
        d_fast = load(fullfile(paths.data, 'data_fast_theorem74.mat'));
        theta_star_lambda = d_fast.theta_star_lambda;
    else
        theta_star_lambda = [0.0002; 0.0070; -0.0001; 0.1245];
    end
    
    for d_idx = 1:length(deltas)
        rng(42);
        d_val = deltas(d_idx);
        X_opt_d = x_ref * ones(1, N);
        theta_opt_d = theta_init_ts;
        for n = 1:N_steps_delta
            noise_opt_d = sigma_ts * randn(1, N);
            phi_d = [ones(1, N); X_opt_d; X_opt_d.^2; X_opt_d.^3];
            u_lin_d = theta_opt_d' * phi_d;
            x_target_d = 0.8 * (X_opt_d - x_ref) + x_ref;
            X_opt_d_next = reflect(A_plant*X_opt_d + B_true*X_opt_d.^3 + calc_u_loose(X_opt_d) - u_lin_d + noise_opt_d);
            err_bp_d = -(X_opt_d_next - x_target_d);
            G_opt_d = phi_d * err_bp_d' / N + lambda_ts * theta_opt_d;
            
            theta_opt_d = theta_opt_d - d_val * G_opt_d;
            if any(isnan(theta_opt_d)) || any(abs(theta_opt_d) > 1e4)
                theta_opt_d = min(max(theta_opt_d, -1e4), 1e4);
            end
            E_T_deltas(d_idx, n) = norm(theta_opt_d - theta_star_lambda);
            X_opt_d = X_opt_d_next;
        end
    end

    %% EXPERIMENT 3: Regularized Field Shift & Comparison (Part 1, Figure 3 Right)
    fprintf('Running Exp 3: Regularized Field Shift (sigma = 0.18, delta = 0.05)...\n');
    delta_reg = 0.05;
    N_steps_reg = 15000;
    lambdas_reg = [0, 1e-3];
    E_T_unreg_d05 = zeros(1, N_steps_reg);
    E_T_reg_opt_d05 = zeros(1, N_steps_reg);
    E_T_reg_star_d05 = zeros(1, N_steps_reg);
    
    % 1) Unregularized lambda = 0
    rng(42);
    X_reg = x_ref * ones(1, N);
    theta_reg = theta_init_ts;
    for n = 1:N_steps_reg
        noise_reg = sigma_ts * randn(1, N);
        phi_reg = [ones(1, N); X_reg; X_reg.^2; X_reg.^3];
        u_lin_reg = theta_reg' * phi_reg;
        x_target_reg = 0.8 * (X_reg - x_ref) + x_ref;
        X_reg_next = reflect(A_plant*X_reg + B_true*X_reg.^3 + calc_u_loose(X_reg) - u_lin_reg + noise_reg);
        err_bp_reg = -(X_reg_next - x_target_reg);
        G_reg = phi_reg * err_bp_reg' / N;
        theta_reg = theta_reg - delta_reg * G_reg;
        E_T_unreg_d05(n) = norm(theta_reg - theta_star);
        X_reg = X_reg_next;
    end
    
    % 2) Regularized lambda = 1e-3
    rng(42);
    X_reg = x_ref * ones(1, N);
    theta_reg = theta_init_ts;
    for n = 1:N_steps_reg
        noise_reg = sigma_ts * randn(1, N);
        phi_reg = [ones(1, N); X_reg; X_reg.^2; X_reg.^3];
        u_lin_reg = theta_reg' * phi_reg;
        x_target_reg = 0.8 * (X_reg - x_ref) + x_ref;
        X_reg_next = reflect(A_plant*X_reg + B_true*X_reg.^3 + calc_u_loose(X_reg) - u_lin_reg + noise_reg);
        err_bp_reg = -(X_reg_next - x_target_reg);
        G_reg = (phi_reg * err_bp_reg' / N) + 1e-3 * theta_reg;
        theta_reg = theta_reg - delta_reg * G_reg;
        E_T_reg_opt_d05(n) = norm(theta_reg - theta_star);
        E_T_reg_star_d05(n) = norm(theta_reg - theta_star_lambda);
        X_reg = X_reg_next;
    end

    %% EXPERIMENT 4: Escaping Identifiability Trap
    fprintf('Running Exp 4: Escaping the Identifiability Trap...\n');
    X_off_loose = x_ref * ones(1, N);
    theta_off_loose = zeros(num_params, 1);
    % We use a smaller delta here because the loose controller produces wider variance
    % We use a stable learning rate here so it learns before escaping the bounds
    delta_loose = 0.015; 
    for n = 1:4000
        noise = noise_std * randn(1, N);
        phi_off_loose = [ones(1, N); X_off_loose; X_off_loose.^2; X_off_loose.^3];
        u_lin_loose = theta_off_loose' * phi_off_loose;
        x_target_loose = 0.8 * (X_off_loose - x_ref) + x_ref; 
        X_next_loose = reflect(A_plant*X_off_loose + B_true*X_off_loose.^3 + calc_u_loose(X_off_loose) - u_lin_loose + noise);
        err_bp_loose = -(X_next_loose - x_target_loose);
        G_off_loose = phi_off_loose * err_bp_loose' / N;
        theta_off_loose = theta_off_loose - delta_loose * G_off_loose;
        X_off_loose = X_next_loose;
    end
    mu_off_loose = X_off_loose;
    
    X_dep_loose = x_ref * ones(1, N);
    for n = 1:1000
        noise = noise_std * randn(1, N);
        phi_dep_loose = [ones(1, N); X_dep_loose; X_dep_loose.^2; X_dep_loose.^3];
        u_lin_dep_loose = theta_off_loose' * phi_dep_loose;
        X_dep_loose = reflect(A_plant*X_dep_loose + B_true*X_dep_loose.^3 + calc_u_loose(X_dep_loose) - u_lin_dep_loose + noise);
    end
    mu_deploy_loose = X_dep_loose;

    save(fullfile(paths.data, 'data_validate_theoretical.mat'), 'mu_off', 'mu_deploy', 'X_opt', 'E_T_opt', 'Action_Err', ...
        'theta_off', 'theta_opt_final', 'E_T_deltas', 'deltas', 'N_steps_delta', ...
        'E_T_unreg_d05', 'E_T_reg_opt_d05', 'E_T_reg_star_d05', 'lambdas_reg', 'N_steps_reg', 'mu_off_loose', 'mu_deploy_loose', ...
        'theta_star', 'theta_star_lambda', 'x_ref');

    %% VISUALIZATION
    fprintf('Generating Visualizations...\n');
    set(groot, 'DefaultFigureVisible', 'off');
    
    %% Part 1: Time-Scale Separation and Regularization Comparison (1x2)
    if usejava('desktop')
        fig_part1_ts = figure('Name', 'Part 1: Time-Scale and Regularization', 'Position', [100, 100, 1100, 400]);
        t_ts = tiledlayout(1, 2, 'TileSpacing', 'Compact');
        
        % Left panel: Theorem 7.4 Exponential Convergence to theta_*^lambda
        nexttile; hold on; box on;
        colors = lines(length(deltas));
        for d_idx = 1:length(deltas)
            semilogy(1:N_steps_delta, E_T_deltas(d_idx, :), 'LineWidth', 1.8, 'Color', colors(d_idx,:), ...
                'DisplayName', sprintf('\\delta = %.3f', deltas(d_idx)));
        end
        title('Theorem 7.4 Convergence to \theta_*^\lambda (\lambda = 10^{-3}, \sigma = 0.18)');
        xlabel('Adaptation Steps (n)'); ylabel('||\theta_n - \theta_*^\lambda||');
        legend('Location', 'northeast'); grid on;
        set(gca, 'YScale', 'log');
        ylim([1e-4, 1e1]);
        
        % Right panel: Comparison Non-Regularized vs Regularized
        nexttile; hold on; box on;
        semilogy(1:N_steps_reg, E_T_unreg_d05, 'r-', 'LineWidth', 2.0, 'DisplayName', 'Unregularized to \theta_{opt} (\lambda = 0)');
        semilogy(1:N_steps_reg, E_T_reg_opt_d05, 'Color', [0.93 0.69 0.13], 'LineStyle', '--', 'LineWidth', 1.8, 'DisplayName', 'Regularized to \theta_{opt} (\lambda = 10^{-3}, Prop. 7.7 Bias)');
        semilogy(1:N_steps_reg, E_T_reg_star_d05, 'Color', [0.00 0.45 0.74], 'LineStyle', '-.', 'LineWidth', 2.0, 'DisplayName', 'Regularized to \theta_*^\lambda (\lambda = 10^{-3}, Thm. 7.4)');
        set(gca, 'YScale', 'log'); grid on;
        title('Equilibrium Bias vs Exponential Convergence (\delta = 0.05)');
        xlabel('Adaptation Steps (n)'); ylabel('Distance to Target Parameter');
        legend('Location', 'northeast');
        ylim([1e-4, 1e1]);
        
        exportgraphics(fig_part1_ts, fullfile(paths.figures, 'fig_part1_timescale_and_reg.pdf'), 'ContentType', 'vector');
    end
    
    %% Part 2: Action Error Recovery Taxonomy (1x3)
    fig_part2_ae = figure('Name', 'Part 2: Action Error Recovery', 'Position', [100, 100, 1500, 400], 'Visible', 'off');
    t_ae = tiledlayout(1, 3, 'TileSpacing', 'Compact');
    
    % Left panel: State Distribution Shattering
    nexttile; hold on; box on;
    histogram(mu_off, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.6 0.6 0.6]);
    histogram(mu_deploy, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2.0, 'EdgeColor', 'r');
    histogram(X_opt, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2.0, 'EdgeColor', 'b');
    xline(x_ref, 'k--', 'LineWidth', 1.5);
    title('State Distribution Dispersion');
    xlabel('State X'); ylabel('Density');
    legend('\mu_{off} (Offline)', '\mu_{deploy} (Dispersed)', '\mu_* (Online STO)', 'Location', 'northwest');
    xlim([-2.2, 2.2]); grid on;
    
    % Middle panel: Action Error & Parameter Error Dual Axis
    nexttile; box on;
    yyaxis left;
    plot(1:N_steps_adapt, E_T_opt, '-', 'LineWidth', 2.0, 'DisplayName', 'Parameter Error');
    ylabel('Parameter Error || \theta_n - \theta_{opt} ||');
    ylim([0, max(E_T_opt)*1.1]);
    yyaxis right;
    plot(1:N_steps_adapt, Action_Err, '-', 'LineWidth', 2.0, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Compensation error $E_A$');
    ylabel('$E_A(\theta_n) = \|g(\cdot,\theta_n) - d\|_{L^2(\mu_n)}$', 'Interpreter', 'latex');
    ylim([0, max(Action_Err)*1.1]);
    title('Action Error Recovery');
    legend('Location', 'northeast');
    xlabel('Adaptation Steps'); grid on;
    
    % Right panel: Drift Return Map
    nexttile; hold on; box on;
    plot(x_eval, x_eval, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, 'DisplayName', 'Stability Boundary (y=x)');
    plot(x_eval, 0.8*(x_eval-x_ref) + x_ref, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Target Drift');
    drift_deploy = A_plant*x_eval + B_true*x_eval.^3 + calc_u_loose(x_eval) - (theta_off'*phi_eval);
    plot(x_eval, drift_deploy, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Deployed (Dispersed)');
    drift_opt = A_plant*x_eval + B_true*x_eval.^3 + calc_u_loose(x_eval) - (theta_opt_final'*phi_eval);
    plot(x_eval, drift_opt, 'b-', 'LineWidth', 2.0, 'DisplayName', 'Online STO');
    title('System Dynamics: Drift Return Map');
    xlabel('State X_n'); ylabel('Expected X_{n+1}');
    legend('Location', 'northwest');
    xlim([-2.2, 2.2]); ylim([-2.5, 2.5]); grid on;
    
    exportgraphics(fig_part2_ae, fullfile(paths.figures, 'fig_part2_action_error_recovery.pdf'), 'ContentType', 'vector');
    
    %% Part 2: Escaping Identifiability Trap via Baseline Gain Matching (Single panel)
    fig_part2_esc = figure('Name', 'Part 2: Escaping Identifiability Trap', 'Position', [100, 100, 600, 400], 'Visible', 'off');
    hold on; box on;
    histogram(mu_off, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'EdgeColor', [0.75 0.75 0.75]);
    histogram(mu_deploy, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', [0.85 0.2 0.2], 'LineStyle', ':');
    histogram(mu_off_loose, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2.8, 'EdgeColor', [0.2 0.2 0.2]);
    histogram(mu_deploy_loose, 150, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 1.4, 'EdgeColor', [0 0.4470 0.7410], 'LineStyle', '--');
    title('Escaping Weak Excitation via Gain Matching');
    xlabel('State X'); ylabel('Density');
    legend('\mu_{off} (Tight, gain = -0.80)', '\mu_{deploy} (Dispersed, gain = -0.25)', '\mu_{off} (Loose, gain = -0.25)', '\mu_{deploy} (Stable, gain = -0.25)', 'Location', 'northwest', 'FontSize', 8.5);
    xlim([-2.2, 2.2]); grid on;
    
    exportgraphics(fig_part2_esc, fullfile(paths.figures, 'fig_part2_escaping_trap_offline.pdf'), 'ContentType', 'vector');
    
    % Compute empirical Hessian at optimal measure to log spectral properties
    phi_opt = [ones(1, N); X_opt; X_opt.^2; X_opt.^3];
    H_star = (phi_opt * phi_opt') / N;
    lambda_eig = eig(H_star);
    lambda_min = min(lambda_eig);
    lambda_max = max(lambda_eig);
    delta_c_pred = 2 / lambda_max;

    fid_log = fopen(fullfile(paths.results, 'validate_theoretical_metrics.log'), 'w');
    fprintf(fid_log, '1. Action Error Recovery:\n');
    fprintf(fid_log, '   - Final Action Error:  %.5f\n', Action_Err(end));
    fprintf(fid_log, '2. Spectral Properties at Equilibrium (sigma=%.2f):\n', sigma_online);
    fprintf(fid_log, '   - lambda_min(H_*): %.5e\n', lambda_min);
    fprintf(fid_log, '   - lambda_max(H_*): %.5f\n', lambda_max);
    fprintf(fid_log, '   - Predicted delta_c = 2/lambda_max: %.5f\n', delta_c_pred);
    fclose(fid_log);
    
    fprintf('Done. Saved figures.\n');
    



function x = reflect(x)
    % Perfect infinitely-periodic continuous reflection into [-2, 2]
    x = abs(mod(x - 2.0, 8.0) - 4.0) - 2.0;
end
