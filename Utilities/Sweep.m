        function phantms = Sweep(paretos)

% find size of Pareto matrix
[numpar,numobj] = size(paretos);

if numobj==1
    phantms = min(paretos);
else
phantms=ones(1,numobj)*0;
k=1:1:numobj;

%choose dimension to sweep randomly
d = randi([1,numobj],1);

% dimensions other than d
T = k(k~=d);

%sort by descending dimension d
paretos=sortrows(paretos,-d);

%sweep dimension d from max to min, with projections in dimensions T
for i=1:numpar-(numobj-1)        %need at least numobj-1 Paretos to form a phantom
    maxy=paretos(i,d);  %value of dth coordinate for created phantoms
    tempPars=paretos((i+1):numpar,:);   %paretos in front of current max pareto
    tempPars(:,d)=[];   % delete dth dimension of remaining paretos
    [~, tempPars2]=find_pareto_frontier(tempPars);    %find pareto front
    phants = Sweep(tempPars2);   %use module to find phantoms
    
    % Make sure phantoms are projections into back plane
    for j=1:numobj-1
        ind = phants(:,j)>paretos(i,T(j));
        phants = phants(ind,:);
    end
    
    % add back dth dimension, with appropriate max value
    phan=ones(size(phants,1),numobj)*maxy;
    phan(:,T) = phants;
    phantms=[phantms; phan];  %add phantoms to list
end
phantms(1,:)=[];
end

