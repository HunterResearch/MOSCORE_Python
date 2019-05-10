function [systems]=LOAD_PROBLEM2D()

obj_vals =       [ 2 10;
                   6  6;
                  10  3;
                   4  8;
                   5  9;
                   3 15;
                   8  4;
                   12 5;
                   7  8;
                   9  7];
              
 objvar = 4;
 objcorr = .35;
 
[systems] = ProblemStruct(obj_vals,objvar,objcorr);

%save('/Users/ericapplegate/Desktop/MORS-SCORE/SusansMatlab/MOCBA/ORACLE/ProblemTrue.mat')