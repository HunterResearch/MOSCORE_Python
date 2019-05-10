function [seed,u] = mrg32k3a(seed)
%
%..........................................................................
%     Raghu Pasupathy     June 2012.                                   
%     
%   This is an implementation of Pierre L'Ecuyer's Random Number Generator,
%   MRG32K3A. ("Good Parameters and Implementations for Combined Multiple 
%   Recursive Random Number Generators", Operations Research 47, pp.
%   159-164, 1999.)
%
%   
%..........................................................................
%     input:  seed(1),seed(2),seed(3) positive integers < m1, not all zero.                                           
%     input:  seed(4),seed(5),seed(6) positive integers < m2, not all zero.
%
%     output: seed                      (seeds to be used in the next call)
%             u                         (a pseudorandom number in (0,1)    
%..........................................................................

s1 = 1403580;
t1 = 810728;
s2 = 527612;
t2 = 1370589;
m1 = 4294967087;  % = 2^32 - 209
m2 = 4294944443;  % = 2^32 - 22853
m3 = 4294967088;  % = 2^32 - 208

p1 = mod( ( s1 * seed(1) ) - ( t1 * seed(2) ), m1);
p2 = mod( ( s2 * seed(4) ) - ( t2 * seed(5) ), m2);

z = mod( ( p1 - p2 ), m1 );

if ( z > 0 )
    u = z / m3;
elseif ( z == 0 )
    u = m1 / m3;
end

seed(1) = seed(2); seed(2) = seed(3); seed(3) = p1;
seed(4) = seed(5); seed(5) = seed(6); seed(6) = p2;


%--------------------------------------------------------------------------