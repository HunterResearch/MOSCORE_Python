function phantoms = SweepDriver(paretos,maxy);

% find size of Pareto matrix
[numpar,numobj] = size(paretos);
phantoms=ones(1,numobj)*0;

v=1:1:numobj;
for b=1:numobj;
T = nchoosek(v,b);
    for i=1:nchoosek(numobj,b)
            tempPars = paretos(:,T(i,:));
            tempPars = unique(tempPars,'rows');
            % find temp paretos
            [idx, tempPars2]=find_pareto_frontier(tempPars);

            %do sweep
            phants = Sweep(tempPars2);

            phan=ones(size(phants,1),numobj)*maxy;
            phan(:,T(i,:)) = phants;
            phantoms=[phantoms; phan];
    end
end
phantoms(1,:)=[];
end

