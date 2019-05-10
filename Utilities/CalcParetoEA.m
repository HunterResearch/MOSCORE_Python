function [myprob] = CalcParetoEA(myprob)
% INPUTS:
%        myprob is a structured array. It contains the fields:
%               - the system index 'num'
%               - the estimated objective function value so far, 'obj'
%                 in column vector form
%               - the estimated covariance matrix so var, 'cov'
%               - the estimated inverse covariance matrix so far, 'invcov'
%                 (SEE ProblemStruct.m in the ORACLE subfolder for an
%                 example.)
% OUTPUTS:
%        myprob is a structured array just like myprob, except it contains
%        the new fields:
%               - obj1, which is the value of the first objective
%               - pareto, which is a numbered indicator of the Pareto set.
%                 if the field for pareto equals zero, then the system is
%                 non-Pareto. For Pareto systems, the "Pareto number" is
%                 based on the ordering of the first objective field, obj1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %GET THE NUMBER OF SYSTEMS AND NUMBER OF OBJECTIVES
    numsys = numel(myprob);
    %[numobj,~] = size(myprob(1).obj); %objectives are column vectors

    %CONVERT OBJECTIVE VALUES TO A MATRIX FORMAT
    objvals = [myprob(1:numsys).obj]';

    %EXTRACT THE FIRST OBJECTIVE AND ADD A FIELD CALLED obj1
    obj1Cell = num2cell(objvals(:,1)',1)'; %transpose and transpose back
    [myprob(:).obj1]=obj1Cell{:};

    %CALCULATE THE PARETO FRONT AND ADD LOGICAL pareto FIELD
    Pfront = paretofront(objvals); % determine the true Pareto set. (Function does not sort.)
    frontCell = num2cell(Pfront); % convert the logical to a cell
    [myprob(:).paretonum]=frontCell{:}; % add the paretonum field to the myprob structure

    % LABEL THE PARETO SYSTEMS BY THEIR ORDERING ON OBJECTIVE 1
    % First, sort structure array by the first objective function value
    [myprob] = nestedSortStruct(myprob,'obj1');
    % Now create the labels for the Pareto systems
    paretonumber = 0;
    for i=1:numsys
        if myprob(i).paretonum == 1
            paretonumber = paretonumber+1;
            myprob(i).paretonum = paretonumber;
        else
            myprob(i).paretonum = Inf;
        end
    end
    % Now sort the structure array by the paretos
    [myprob] = nestedSortStruct(myprob,'paretonum');
