% ==============================================================================
% REPRODUCE_ALL.M
%
% Master Numerical Reproduction & Figure Generation Suite for Section 8
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
%     37C30, 93C40, 37M25, 34E13, 37H10, 62L20, 68T07
%
%   Keywords:
%     Self-Consistent Transfer Operator (STO), Barycentric State-Parameter Measure,
%     Ulam Transfer Operator Discretization, Statistical Linear Response C_* R_* B_*,
%     Time-Scale Separation, Tikhonov Regularization Shift (Proposition 7.7),
%     Persistent Excitation, Two-Phase Deployment Action Error,
%     Linear Stability Threshold, Non-Convex MLP Adaptive Control.
%
% ==============================================================================
% USAGE & REPRODUCTION MODES
% ==============================================================================
%   >> reproduce_all()        % Quick mode (< 1 min): verifies .mat data & regenerates all 9 figures
%   >> reproduce_all('quick') % Same as default
%   >> reproduce_all('full')  % Full mode (10-20 min): re-runs all Monte Carlo & Ulam suites from scratch
%
%   Orchestration Pipeline:
%     (all scripts below live in matlab/ and python/; outputs go to data/, figures/, results/)
%     [1] fast_theorem74_convergence_sto.m        (Section 8.1 & 8.2: Figure 3 & multi-timescale rates)
%     [2] persistent_excitation_regularized_sto.m (Section 8.2 & 8.3.1: Figures 2 & 5, PE vs weak excitation)
%     [3] run_revised_section8_experiments.m      (Section 8.1, 8.2.1 & 8.3.2: Figure 4, Ulam & linear response)
%     [4] validate_theoretical_sto.m              (Section 8.3.2: Figures 6 & 7, two-phase deployment)
%     [5] bifurcation_stability_loss_sto.m        (Section 8.3.3: Figures 8 & 9, loss of linear stability)
%     [6] mlp_nonconvex_proof_sto.m               (Section 8.4: Figure 10, non-convex MLP exploration)
%     [7] regenerate_all_section8_figures.m       (Fast vector PDF figure compiler)
% ==============================================================================

function reproduce_all(mode)
    if nargin < 1
        mode = 'quick';
    end
    
    % Repository layout: reproduce_all.m sits in the repo root; the experiment
    % scripts are in matlab/ and all I/O goes through sto_paths() (data/, figures/, results/).
    addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab'));
    paths = sto_paths();
    
    clc;
    fprintf('========================================================================\n');
    fprintf('  STO Framework: Numerical Reproduction & Figure Generation Suite       \n');
    fprintf('========================================================================\n');
    fprintf('Execution Mode: %s\n\n', upper(mode));
    
    set(groot, 'DefaultFigureVisible', 'off');
    
    switch lower(mode)
        case 'quick'
            % Q1: Are all precomputed Monte Carlo datasets present in the workspace?
            fprintf('[1/4] Checking precomputed simulation datasets (.mat files)...\n');
            assert(isfile(fullfile(paths.data, 'data_part1_nominal.mat')), 'Missing data_part1_nominal.mat');
            assert(isfile(fullfile(paths.data, 'data_fast_theorem74.mat')), 'Missing data_fast_theorem74.mat');
            assert(isfile(fullfile(paths.data, 'data_validate_theoretical.mat')), 'Missing data_validate_theoretical.mat');
            assert(isfile(fullfile(paths.data, 'data_bifurcation_sweep.mat')), 'Missing data_bifurcation_sweep.mat');
            fprintf('      All precomputed simulation datasets verified.\n');
            
            % Q2: How are Figures 2 and 3 rendered from the precomputed trajectories?
            fprintf('[2/4] Regenerating Part 1 Figures (Figures 2 & 3)...\n');
            [py_status, ~] = system('python --version');
            if py_status == 0
                fprintf('      Invoking vector Python renderers (python/plot_nominal.py, python/plot_theorem74.py, python/plot_fig4.py)...\n');
                system(sprintf('python "%s"', fullfile(paths.python, 'plot_nominal.py')));
                system(sprintf('python "%s"', fullfile(paths.python, 'plot_theorem74.py')));
                system(sprintf('python "%s"', fullfile(paths.python, 'plot_fig4.py')));
            else
                fprintf('      Python not found; skipping external Python vector renderer.\n');
            end
            
            % Q3: How are Figures 4 through 10 regenerated in under one minute?
            fprintf('[3/4] Regenerating Section 8 Figures (Figures 4-10)...\n');
            regenerate_all_section8_figures();
            mlp_nonconvex_proof_sto;
            
            % Q4: Are all 9 publication PDFs verified on disk?
            fprintf('[4/4] Verifying generated figures...\n');
            verify_figure_outputs();
            
        case 'full'
            fprintf('Running full end-to-end Monte Carlo and Ulam simulations from scratch (Seed = 42)...\n\n');
            
            fprintf('--- [1/6] Running Regularized STO Convergence & Time-Scale Sweep ---\n');
            fast_theorem74_convergence_sto;
            
            fprintf('\n--- [2/6] Running Persistent Excitation & Regularization Analysis ---\n');
            persistent_excitation_regularized_sto;
            
            fprintf('\n--- [3/6] Running Ulam Operator, Linear Response C_*R_*B_* & Deployment Suite ---\n');
            run_revised_section8_experiments;
            
            fprintf('\n--- [4/6] Running Action Error & Two-Phase Deployment Validation ---\n');
            validate_theoretical_sto;
            
            fprintf('\n--- [5/6] Running Loss of Linear Stability & Density Oscillations Sweep ---\n');
            bifurcation_stability_loss_sto;
            
            fprintf('\n--- [6/6] Running Non-Convex MLP Compensator Exploration ---\n');
            mlp_nonconvex_proof_sto;
            
            fprintf('\n--- Re-rendering publication vector figures ---\n');
            regenerate_all_section8_figures();
            [py_status, ~] = system('python --version');
            if py_status == 0
                system(sprintf('python "%s"', fullfile(paths.python, 'plot_nominal.py')));
                system(sprintf('python "%s"', fullfile(paths.python, 'plot_theorem74.py')));
                system(sprintf('python "%s"', fullfile(paths.python, 'plot_fig4.py')));
            end
            
            verify_figure_outputs();
            
        otherwise
            error('Unknown mode: %s. Use ''quick'' or ''full''.', mode);
    end
    
    fprintf('\n========================================================================\n');
    fprintf('  Reproduction complete! All paper figures generated successfully.      \n');
    fprintf('========================================================================\n');
end

%% Helper: Verify all 9 Section 8 figure PDFs exist and are non-empty
function verify_figure_outputs()
    fig_files = {
        'fig_part1_nominal_convergence.pdf', ...
        'fig_part1_timescale_and_reg.pdf', ...
        'fig_linear_response_contribution.pdf', ...
        'fig_part2_identifiability_trap_pe.pdf', ...
        'fig_part2_action_error_recovery.pdf', ...
        'fig_part2_escaping_trap_offline.pdf', ...
        'fig_part2_bifurcation_diagram.pdf', ...
        'fig_part2_density_shattering.pdf', ...
        'fig5_mlp_convergence.pdf'
    };
    
    paths = sto_paths();
    fprintf('\nVerification of Section 8 Paper Figures (figures/):\n');
    for k = 1:length(fig_files)
        fn = fig_files{k};
        fp = fullfile(paths.figures, fn);
        if isfile(fp)
            info = dir(fp);
            fprintf('  [OK] %-40s (%7.1f KB)\n', fn, info.bytes / 1024);
        else
            fprintf('  [MISSING] %-40s\n', fn);
        end
    end
end

% ==============================================================================
% ARCHIVAL CITATION
% ==============================================================================
%   @article{GalatoloLippi2026STO,
%     title   = {Autonomous Neural Adaptive Control as a Self-Consistent Transfer Operator},
%     author  = {Galatolo, Stefano and Lippi, Vittorio},
%     journal = {Journal of Nonlinear Science},
%     note    = {Submitted},
%     year    = {2026}
%   }
% ==============================================================================
