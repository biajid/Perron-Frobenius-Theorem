%% Stanford Web Graph: sparse PageRank in MATLAB
% Downloads the real 2002 Stanford web graph automatically on first run.
% Source: https://snap.stanford.edu/data/web-Stanford.html
% SNAP format: each non-comment line is "FromNodeId ToNodeId".
% Our convention: A(i,j)=1 when page j links TO page i.
%
% Run from MATLAB:
%   stanford_pagerank
%
% Outputs (in a folder named pagerank_output beside this script):
%   pagerank_top20.csv, pagerank_convergence.png,
%   pagerank_top20.png, pagerank_results.mat
%
% Requires MATLAB with websave, gunzip, sparse, readtable support.
% No toolboxes are required.

clear; clc; close all;

%% 1. Settings and output folders
sourceURL = 'https://snap.stanford.edu/data/web-Stanford.txt.gz';
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    workDir = pwd;
else
    workDir = fileparts(scriptPath);
end

dataDir = fullfile(workDir, 'pagerank_data');
outputDir = fullfile(workDir, 'pagerank_output');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

archiveFile = fullfile(dataDir, 'web-Stanford.txt.gz');
edgeFile = fullfile(dataDir, 'web-Stanford.txt');
nodeCountFromSNAP = 281903;  % Official dataset metadata

dValues = [0.70, 0.85, 0.95];
tol = 1e-10;              % L1 distance between consecutive iterates
maxIter = 700;
numTop = 20;

%% 2. Download and extract automatically (once)
if ~isfile(edgeFile)
    if ~isfile(archiveFile)
        fprintf('Downloading the SNAP Stanford web graph...\n%s\n', sourceURL);
        try
            websave(archiveFile, sourceURL);
        catch ME
            error('Could not download dataset from SNAP: %s\nURL: %s', ...
                ME.message, sourceURL);
        end
    else
        fprintf('Using cached archive: %s\n', archiveFile);
    end
    fprintf('Extracting archive...\n');
    gunzip(archiveFile, dataDir);
else
    fprintf('Using cached edge file: %s\n', edgeFile);
end

%% 3. Read directed edge list
% Header lines beginning with # are comments.
% For SNAP: the first column is the SOURCE, the second is the TARGET.
fid = fopen(edgeFile, 'rt');
if fid < 0, error('Cannot open %s', edgeFile); end
edgeData = textscan(fid, '%f%f', ...
    'CommentStyle', '#', ...
    'Delimiter', {' ', '\t'}, ...
    'MultipleDelimsAsOne', true, ...
    'CollectOutput', true);
fclose(fid);

edges = edgeData{1};
if isempty(edges) || size(edges,2) ~= 2
    error('Dataset contains no valid two-column edge list.');
end

source = edges(:,1);
target = edges(:,2);
clear edges edgeData;

if any(~isfinite(source)) || any(~isfinite(target)) || ...
   any(source ~= floor(source)) || any(target ~= floor(target))
    error('Invalid node identifiers encountered.');
end

% SNAP web-Stanford normally uses IDs 1,...,281903.
% Also accept 0-based IDs if an alternate export is supplied.
if min([source;target]) == 0
    source = source + 1;
    target = target + 1;
    fprintf('Converted 0-based node IDs to MATLAB 1-based IDs.\n');
end

n = max(nodeCountFromSNAP, max([source;target]));
if min([source;target]) < 1
    error('Node IDs must be positive after conversion.');
end
mRaw = numel(source);
fprintf('Loaded %d nodes (metadata-based) and %d input edges.\n', n, mRaw);

%% 4. Construct sparse adjacency matrix, A(target,source)=1
% Parallel/duplicate edges count as one hyperlink.
A = spones(sparse(target, source, 1, n, n));
clear source target;
numEdges = nnz(A);
outDegree = full(sum(A, 1))';  % Number of outgoing links per source page
isDangling = (outDegree == 0);
numDangling = nnz(isDangling);
fprintf('Unique directed edges: %d\n', numEdges);
fprintf('Dangling pages: %d (%.2f%%)\n', numDangling, ...
    100*numDangling/n);

%% 5. Construct sparse link transition matrix W
% We deliberately do NOT fill dangling columns with 1/n.
% Their missing probability mass is added implicitly in iteration.
scale = zeros(n, 1);
nonDangling = ~isDangling;
scale(nonDangling) = 1 ./ outDegree(nonDangling);
W = A * spdiags(scale, 0, n, n);
clear A scale;
fprintf('Sparse W: %d nonzeros; density %.6f%%\n', ...
    nnz(W), 100*nnz(W)/double(n)^2);

%% 6. Power iteration for each damping value
% xNext = d*(W*x + sum(x(dangling))/n * ones(n,1)) ...
%         + (1-d)/n * ones(n,1)
% This is equivalent to the standard Google matrix, without storing it.
rankVectors = zeros(n, numel(dValues));
history = nan(maxIter, numel(dValues));
iterations = zeros(numel(dValues),1);
runSeconds = zeros(numel(dValues),1);
fixedPointResidual = zeros(numel(dValues),1);

for caseIdx = 1:numel(dValues)
    d = dValues(caseIdx);
    x = ones(n,1)/n;  % Initial uniform probability distribution
    tic;
    for k = 1:maxIter
        danglingMass = sum(x(isDangling));
        xNext = d*(W*x) + ((d*danglingMass + 1-d)/n);
        delta = norm(xNext - x, 1);
        history(k,caseIdx) = delta;
        x = xNext;
        if delta < tol
            break;
        end
    end
    runSeconds(caseIdx) = toc;
    iterations(caseIdx) = k;
    % Check fixed-point equation using the same sparse operator.
    y = d*(W*x) + ((d*sum(x(isDangling)) + 1-d)/n);
    fixedPointResidual(caseIdx) = norm(y-x,1);
    rankVectors(:,caseIdx) = x;

    fprintf('\nd=%.2f: iterations=%d, time=%.3f sec\n', ...
        d, k, runSeconds(caseIdx));
    fprintf('  sum(rank)=%.15f\n', sum(x));
    fprintf('  min(rank)=%.6e, fixed-point L1 residual=%.3e\n', ...
        min(x), fixedPointResidual(caseIdx));
    if k == maxIter && delta >= tol
        warning('d=%.2f reached maxIter without reaching tolerance.',d);
    end
end

%% 7. Top-20 rankings for the conventional d=0.85
[~, refIdx] = min(abs(dValues - 0.85));
[sortedScores, sortedIDs] = sort(rankVectors(:,refIdx), 'descend');
numTop = min(numTop,n);
TopPages = table(sortedIDs(1:numTop), sortedScores(1:numTop), ...
    'VariableNames', {'PageID', 'PageRankScore'});
disp('Top pages with d = 0.85 (SNAP node IDs):');
disp(TopPages);
writetable(TopPages, fullfile(outputDir,'pagerank_top20.csv'));

%% 8. Graphs
figure('Color','w'); hold on;
for caseIdx = 1:numel(dValues)
    semilogy(1:iterations(caseIdx), history(1:iterations(caseIdx),caseIdx), ...
        'LineWidth',1.5, ...
        'DisplayName',sprintf('d = %.2f',dValues(caseIdx)));
end
grid on;
xlabel('Iteration k');
ylabel('||x^{(k+1)} - x^{(k)}||_1');
title('PageRank convergence: SNAP Stanford Web Graph');
legend('Location','northeast');
exportgraphics(gcf, fullfile(outputDir,'pagerank_convergence.png'), ...
    'Resolution',200);

figure('Color','w');
bar(sortedScores(1:numTop));
grid on;
xlabel('Rank position');
ylabel('Stationary probability');
title(sprintf('Top %d PageRank scores (d = %.2f)',numTop,dValues(refIdx)));
exportgraphics(gcf, fullfile(outputDir,'pagerank_top20.png'), ...
    'Resolution',200);

%% 9. Save numeric results for the LaTeX report
Summary = table(dValues(:),iterations,runSeconds,fixedPointResidual, ...
    'VariableNames', {'Damping','Iterations','Seconds','L1Residual'});
disp('Experiment summary:');
disp(Summary);
writetable(Summary,fullfile(outputDir,'pagerank_experiment_summary.csv'));
save(fullfile(outputDir,'pagerank_results.mat'), ...
    'rankVectors','dValues','iterations','runSeconds', ...
    'fixedPointResidual','numDangling','numEdges','n','tol');

fprintf('\nResults and figures saved in:\n%s\n', outputDir);
