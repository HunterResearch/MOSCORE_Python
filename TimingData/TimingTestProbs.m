% Refer to the timing tables in the paper for all of the following.  Some
% of the bigger test problems did not complete for some algorithms, for
% instance.

% Pick which test problems you want to run (be sure you add the path of
% where the files are if they are saved in different locations)
% fixed = we kept a fixed number of Pareto systems
% rand = we allowed the number of Pareto systems to be random
load('FixedProblems3D10.mat');
% load('RandProblems3D10.mat');
% load('FixedProblems4D10.mat');
% load('RandProblems4D10.mat');
% load('FixedProblems5D10.mat');
% load('RandProblems5D10.mat');

% the following line sets the problem size:
%   1 = 20 systems
%   2 = 50 systems
%   3 = 100 systems
%   4 = 250 systems
%   5 = 500 systems
%   6 = 1000 systems
%   7 = 2000 systems
a=1; 

for b=1:10       % cycle through the 10 problems for each problem size a

    objVals = objs{b,a};
    numsys = size(objVals,1);
    corrb = corrs(b,a);
    c = eye(numobj)+(ones(numobj)-eye(numobj))*corrb;
    covs = repmat(c,1,1,numsys);
    
    fprintf('Number of systems = %d, Corr = %3.2f, Problem Iteration: %d \n',numsys,corrb,b);
    
    % In Matlab, we would set up the EstObj structure here, based on the
    % objVals and covs
    
    % Then we would run the allocation (timing it) and record the time and
    % the optimal alphas
    
    % We save the time and Alphas in a .mat file, then loop to the
    % next problem
    
end