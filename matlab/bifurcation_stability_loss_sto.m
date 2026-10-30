% ==============================================================================
% BIFURCATION_STABILITY_LOSS_STO.M
%
% Paper: "Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator:
%         A Barycentric State-Parameter Formulation"
% Paper Authors: Stefano Galatolo and Vittorio Lippi
% Code Author:   Vittorio Lippi (vittorio.lippi@calejo.ai)
% Affiliation:   Calejo Hybrid Intelligence AB
% License:       MIT License (c) 2026 Stefano Galatolo and Vittorio Lippi
%
% --- Theoretical Framework & Semantic Indexing Metadata ----------------------
% Domain: Adaptive Control, Nonlinear Dynamics, Bifurcation Theory, Chaos
% Keywords: Period-Doubling Bifurcation, Chaotic Density Shattering, Flip Bifurcation,
%           Critical Learning Rate delta_c, Spectral Radius, Loss of Stability,
%           Transfer Operator Spectral Decomposition, Multiplier Crossing -1,
%           Reflected Duffing Oscillator, Feigenbaum Cascade, Ergodic Invariant Density.
%
% Mathematical Core & System Specification:
%   - Plant: Reflected Duffing Dynamics: x_{n+1} = r_2(1.05 x_n + 0.15 x_n^3 - 0.25 x_n - u_lin + xi_n)
%   - Theory: In Theorem 7.4 / Remark 7.5, the leading slow eigenvalue is:
%       lambda_slow(delta) = 1 - delta * lambda_max(H_*) + o(delta)
%   - Critical Flip Threshold: When delta exceeds delta_c = 2 / lambda_max(H_*) approx 2.00,
%     the leading multiplier crosses -1, precipitating a period-doubling bifurcation.
%   - Regimes Simulated:
%       * delta = 1.85: Stable underdamped oscillatory concentration within stable basin.
%       * delta = 2.22: Beyond delta_c, leading to full chaotic density shattering across [-2, 2].
%
% Output Artifacts:
%   - Data:    data/data_bifurcation_sweep.mat
%   - Figures: figures/fig_part2_bifurcation_diagram.pdf, figures/fig_part2_density_shattering.pdf,
%              figures/supplementary/fig_part2_density_candidates.pdf
% ==============================================================================
addpath(fileparts(mfilename('fullpath'))); paths = sto_paths();  % repository folders (data/, figures/, results/)
rng(42);

N_particles_bifurcation = 100; 
N_steps_bifurcation = 1600;    

A_plant = 1.05;          
B_true = 0.15;           
noise_std = 0.05;        
x_ref = 0.0;             
calc_u_loose = @(x) -0.25 * (x - x_ref);
num_params = 4;

% Sweep delta around the breaking point
deltas = linspace(1.50, 2.25, 500);
N_settle = 1500;
N_record = 100;
bifurcation_data_delta = zeros(length(deltas) * N_record, 1);
bifurcation_data_state = zeros(length(deltas) * N_record, 1);

fprintf('Simulating Bifurcation Diagram (Physical State X)...\n');
for d_idx = 1:length(deltas)
    d_val = deltas(d_idx);
    X = x_ref * ones(1, N_particles_bifurcation);
    theta = [0; 0; 0; 0.15] + [1e-2; -1e-2; 0; 0];  % small perturbation around theta_opt
    
    for n = 1:N_steps_bifurcation
        noise = noise_std * randn(1, N_particles_bifurcation);
        phi = [ones(1, N_particles_bifurcation); X; X.^2; X.^3];
        u_lin = theta' * phi;
        
        x_target = 0.8 * (X - x_ref) + x_ref;
        X_next = reflect(A_plant*X + B_true*X.^3 + calc_u_loose(X) - u_lin + noise);
        
        err_bp = -(X_next - x_target);
        G = phi * err_bp' / N_particles_bifurcation;
        
        X = X_next;
        theta = theta - d_val * G;
        
        if n > N_settle && n <= N_settle + N_record
            idx = (d_idx - 1) * N_record + (n - N_settle);
            bifurcation_data_delta(idx) = d_val;
            bifurcation_data_state(idx) = X(1); % Option A: Record physical state X(1) in [-2, 2]
        end
    end
end

%% Part 2: Distribution Evolution (Stable vs Growing Chaos)
fprintf('Simulating Time-Evolution of Distribution (Stable vs Chaotic)...\n');
N_particles_dense = 10000;
N_steps_adapt = 150;

delta_stable = 1.85; % Stable underdamped concentration in (delta_osc, delta_c) \approx (1.00, 2.00)
delta_chaos  = 2.22; % Envelope growth into full chaotic density shattering beyond delta_c \approx 2.00

N_bins = 200;
edges = linspace(-2, 2, N_bins + 1); 

P_den_stable = zeros(N_bins, N_steps_adapt);
P_den_chaos  = zeros(N_bins, N_steps_adapt);

X_stab = x_ref * ones(1, N_particles_dense);
theta_stab = [0.10; -0.10; 0; 0]; % Non-zero constant/linear init to excite dominant mode at x_ref=0
for n = 1:N_steps_adapt
    noise = noise_std * randn(1, N_particles_dense);
    phi = [ones(1, N_particles_dense); X_stab; X_stab.^2; X_stab.^3];
    u_lin = theta_stab' * phi;
    
    x_target = 0.8 * (X_stab - x_ref) + x_ref;
    X_stab_next = reflect(A_plant*X_stab + B_true*X_stab.^3 + calc_u_loose(X_stab) - u_lin + noise);
    
    err_bp = -(X_stab_next - x_target);
    G = phi * err_bp' / N_particles_dense;
    
    X_stab = X_stab_next;
    theta_stab = theta_stab - delta_stable * G;
    P_den_stable(:, n) = histcounts(X_stab, edges, 'Normalization', 'pdf')';
end

% Save RNG state so candidate comparisons use the exact same noise realization
rng_after_stab = rng;

% Candidate sweep for Task 4 verification: envelope growth beyond delta_c \approx 2.000
delta_candidates = [2.18, 2.20, 2.22, 2.25];
P_den_cand = cell(1, length(delta_candidates));
for c_idx = 1:length(delta_candidates)
    rng(rng_after_stab);
    d_cand = delta_candidates(c_idx);
    X_c = x_ref * ones(1, N_particles_dense);
    theta_c = [0.10; -0.10; 0; 0];
    P_c = zeros(N_bins, N_steps_adapt);
    for n = 1:N_steps_adapt
        noise = noise_std * randn(1, N_particles_dense);
        phi = [ones(1, N_particles_dense); X_c; X_c.^2; X_c.^3];
        u_lin = theta_c' * phi;
        x_target = 0.8 * (X_c - x_ref) + x_ref;
        X_c_next = reflect(A_plant*X_c + B_true*X_c.^3 + calc_u_loose(X_c) - u_lin + noise);
        err_bp = -(X_c_next - x_target);
        G = phi * err_bp' / N_particles_dense;
        X_c = X_c_next;
        theta_c = theta_c - d_cand * G;
        P_c(:, n) = histcounts(X_c, edges, 'Normalization', 'pdf')';
    end
    P_den_cand{c_idx} = P_c;
    fprintf('Candidate delta = %.3f: final state std(X_150) = %.4f\n', d_cand, std(X_c));
end

% Extract the selected delta_chaos = 2.22 run
P_den_chaos = P_den_cand{3};

save(fullfile(paths.data, 'data_bifurcation_sweep.mat'), 'bifurcation_data_delta', 'bifurcation_data_state', ...
    'delta_stable', 'delta_chaos', 'P_den_stable', 'P_den_chaos', 'P_den_cand', ...
    'delta_candidates', 'N_steps_adapt', 'x_ref');

%% Visualization
fprintf('Generating Visualizations...\n');
set(groot, 'DefaultFigureVisible', 'off');

%% Part 2: Bifurcation Diagram (Single Panel - Option A: Physical State X)
fig_bif_diag = figure('Name', 'Part 2: Bifurcation Diagram', 'Position', [100, 100, 650, 420], 'Visible', 'off');
hold on; box on;
scatter(bifurcation_data_delta, bifurcation_data_state, 1, 'k', '.', 'HandleVisibility', 'off');
xline(delta_stable, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\delta = %.2f (Damped alternating)', delta_stable));
xline(delta_chaos, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\delta = %.2f (Dispersed)', delta_chaos));
xline(2.000, 'm:', 'LineWidth', 1.5, 'DisplayName', ['Theoretical \delta_c = 2/\lambda_{max}(H_*) ' char(8776) ' 2.000']);
yline(x_ref, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Target Setpoint x_{ref} = 0');
title('Bifurcation Diagram: Physical State X vs Learning Rate \delta', 'Interpreter', 'tex');
xlabel('Learning Rate \delta');
ylabel('Asymptotic Physical State X');
legend('Location', 'southwest');
ylim([-2.2, 2.2]);
xlim([1.50, 2.25]);
grid on;
exportgraphics(fig_bif_diag, fullfile(paths.figures, 'fig_part2_bifurcation_diagram.pdf'), 'ContentType', 'vector');

%% Part 2: Density Evolution & Shattering (1x2 Layout)
fig_density = figure('Name', 'Part 2: Density Evolution', 'Position', [100, 100, 1100, 420], 'Visible', 'off');
t_den = tiledlayout(1, 2, 'TileSpacing', 'Compact');

n_c = 256;
colormap_array = [linspace(0, 1, n_c)', zeros(n_c, 1), zeros(n_c, 1)]; 
colormap_array(128:end, 2) = linspace(0, 1, 129)'; 
colormap_array(192:end, 3) = linspace(0, 1, 65)';

c_max = max(max(P_den_stable(:).^0.1), max(P_den_chaos(:).^0.1));

nexttile;
imagesc([1, N_steps_adapt], [-2, 2], P_den_stable.^0.1); 
colormap(gca, colormap_array); set(gca, 'YDir', 'normal'); 
clim([0, c_max]);
colorbar;
yline(0, '--w', 'LineWidth', 1.2);
title(sprintf('Damped Alternating Concentration (\\delta = %.2f)', delta_stable)); 
xlabel('Adaptation Steps (n)'); ylabel('State X');

nexttile;
imagesc([1, N_steps_adapt], [-2, 2], P_den_chaos.^0.1); 
colormap(gca, colormap_array); set(gca, 'YDir', 'normal'); 
clim([0, c_max]);
colorbar;
yline(0, '--w', 'LineWidth', 1.2);
title(sprintf('Density Dispersion (\\delta = %.2f)', delta_chaos)); 
xlabel('Adaptation Steps (n)'); ylabel('State X');

exportgraphics(fig_density, fullfile(paths.figures, 'fig_part2_density_shattering.pdf'), 'ContentType', 'vector');

%% Optional Diagnostic: 1x4 Candidate Sweep Figure
fig_cand = figure('Name', 'Candidate Delta Sweep', 'Position', [100, 100, 1600, 380], 'Visible', 'off');
tiledlayout(1, 4, 'TileSpacing', 'Compact');
for c_idx = 1:length(delta_candidates)
    nexttile;
    imagesc([1, N_steps_adapt], [-2, 2], P_den_cand{c_idx}.^0.1);
    colormap(gca, colormap_array); set(gca, 'YDir', 'normal');
    clim([0, c_max]); colorbar;
    yline(x_ref, 'w--', 'LineWidth', 1.2);
    title(sprintf('\\delta_{chaos} = %.3f', delta_candidates(c_idx)));
    xlabel('Adaptation Steps (n)'); ylabel('State X');
end
exportgraphics(fig_cand, fullfile(paths.supp, 'fig_part2_density_candidates.pdf'), 'ContentType', 'vector');

%% Spectral check: lambda_max(H_*) at sigma = noise_std (source of delta_c = 2/lambda_max)
N_H = 100000; X_H = x_ref * ones(1, N_H); theta_H = [0; 0; 0; 0.15];
for n = 1:2000
    phi_H = [ones(1, N_H); X_H; X_H.^2; X_H.^3];
    X_H = reflect(A_plant*X_H + B_true*X_H.^3 + calc_u_loose(X_H) - theta_H'*phi_H + noise_std*randn(1, N_H));
end
phi_H = [ones(1, N_H); X_H; X_H.^2; X_H.^3];
lam_H = eig((phi_H * phi_H') / N_H);
fprintf('sigma=%.2f: lambda_max(H_*)=%.4f, delta_c=2/lambda_max=%.4f\n', noise_std, max(lam_H), 2/max(lam_H));

function x = reflect(x)
    % Perfect infinitely-periodic continuous reflection into [-2, 2]
    x = abs(mod(x - 2.0, 8.0) - 4.0) - 2.0;
end
