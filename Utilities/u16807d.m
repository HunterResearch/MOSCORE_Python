function [seed,u16807d] = u16807d(seed)
u16807d=0;
while (u16807d<=0 || u16807d>=1)
    seed = mod(seed*16807,2147483647);
    u16807d=seed/2147483648;
end
end