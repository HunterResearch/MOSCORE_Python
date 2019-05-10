function [systems]=LOAD_PROBLEM()

obj_vals =       [ 8 36 60;
                  12 32 52;
                  14 38 54;
                  16 46 48;
                   4 42 56;
                  18 40 62;
                  10 44 58;
                  20 34 64;
                  22 28 68;
                  24 40 62;
                  26 38 64;
                  28 40 66;
                  30 42 62;
                  32 44 64;
                  26 40 66;
                  28 42 64;
                  32 38 66;
                  30 40 62;
                  34 42 64;
                  26 44 60;
                  28 38 66;
                  32 40 62;
                  30 46 64;
                  32 44 66;
                  30 40 64];
              
 objvar = 8*8;
 objcorr = .85;
 
[systems] = ProblemStruct(obj_vals,objvar,objcorr);

%save('/Users/ericapplegate/Desktop/MORS-SCORE/SusansMatlab/MOCBA/ORACLE/ProblemTrue.mat')