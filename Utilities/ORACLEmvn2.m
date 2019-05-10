function [iseedV,FnEst,CovEst] = ORACLEmvn2(iseedV,sysindex,sSize)

%==========================================================================
%PURPOSE: This function is an oracle. The objective observations  
%         are generated from a multivariate normal distribution 
%         with prespecified mean and covariance matrix. This function 
%         should be replaced by a user's oracle as desired.
%==========================================================================
%Updateable FEATURES: 
%          This function could (but currently does not) rely on Matlab's 
%          number generator. To control the seed for this generator, use
%          myseed = 555555; rng(myseed); <-- this code resets all of
%          Matlab's randn-based generators. It is best to set this seed at
%          the outermost level, since it is globally controlled.
%==========================================================================
%INPUTS:  iseedV is the seed for the random number generation algorithm. It
%                is a vector containing six entries.
%          sSize is how many samples to obtain at the system sysindex
%==========================================================================
%OUTPUTS: iseedV is the next seed.
%          FnEst is a sSize-by-1 vector containing the function estimate 
%                for sample size sSize at system sysindex.
%==========================================================================

global systems %do this for speed; 

[iseedV,X]=MVNgenSus(iseedV,systems(sysindex).obj,systems(sysindex).cov);
SumEst = X;
Sum2Est = X*X'; 
if sSize>1
    for i=2:sSize
        [iseedV,X]=MVNgenSus(iseedV,systems(sysindex).obj,systems(sysindex).cov);
        SumEst = SumEst+X;
        Sum2Est = Sum2Est + X*X';
    end
end
FnEst = SumEst/sSize;
CovEst = Sum2Est/sSize - FnEst*FnEst'; %COV = E[XY]-E[X]E[Y]
    
    

