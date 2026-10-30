function P = sto_paths()
% STO_PATHS  Canonical repository folders used by every script in this package.
%
%   P = sto_paths() returns a struct with absolute paths (independent of the
%   MATLAB current folder) and creates the output folders if they are missing:
%
%       P.root      repository root
%       P.data      data/                  precomputed .mat datasets (inputs + outputs)
%       P.figures   figures/               the 9 PDF figures used in the paper
%       P.supp      figures/supplementary/ diagnostic figures not used in the paper
%       P.results   results/               plain-text metric logs
%       P.python    python/                vector-figure renderers (called via system())
%
%   This file lives in <root>/matlab/, so the root is the parent of its folder.
%   It contains no numerical code and has no influence on the experiments.

    here = fileparts(mfilename('fullpath'));
    P.root    = fileparts(here);
    P.data    = fullfile(P.root, 'data');
    P.figures = fullfile(P.root, 'figures');
    P.supp    = fullfile(P.root, 'figures', 'supplementary');
    P.results = fullfile(P.root, 'results');
    P.python  = fullfile(P.root, 'python');

    folders = {P.data, P.figures, P.supp, P.results};
    for k = 1:numel(folders)
        if ~isfolder(folders{k}), mkdir(folders{k}); end
    end
end
