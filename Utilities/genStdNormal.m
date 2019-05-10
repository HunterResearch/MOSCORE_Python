function [seedV,Z]=genStdNormal(seedV)
%Use the polar method of Law, 4th ed, p. 454

[seedV,U1]=MRG32k3a(seedV);
[seedV,U2]=MRG32k3a(seedV);

%      Create a single stream and designate it as the current global stream:
%         s = RandStream('mt19937ar','Seed',1)
%         RandStream.setGlobalStream(s);
%
%      Create three independent streams:
%         [s1,s2,s3] = RandStream.create('mrg32k3a','NumStreams',3);
%         r1 = rand(s1,100000,1); r2 = rand(s2,100000,1); r3 = rand(s3,100000,1);
%         corrcoef([r1,r2,r3])

