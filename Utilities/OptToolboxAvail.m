%%
%% Return: true if Optimization Toolbox is available in Matlab
%%
function retval = OptToolboxAvail
    retval = license('test','Optimization_Toolbox');
end