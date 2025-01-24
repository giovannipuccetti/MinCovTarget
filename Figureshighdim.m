% This code computes the average computation time in seconds, 
% minimum envy and total welfare attained by MinCovTarget+
% over a number SIM of random simulations of the value matrix 
% for one single case (in the paper we give figures about n=30, d=300)

clear;
%fix the random generator seed
rng(1);
%T=Goods total value (value matrix standardized to this value;
% set T=10000 when d>=300 to have more meaningful valuations) 
T=1000;

%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% agents is a vector containing the FIXED numbers n of agents for which to
% draw the plot

% draw the plot versus the number d of goods
agents=[30];
N=size(agents,2);
% for each number of agents we give a vector with number of goods at which
% to compute values
xl=[300];
res=size(xl,2);
%SIM=number of REPETITIONS f simulation of value matrices
SIM=50;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v%%%%%%%%%%

%upper=upperbound for target value 
upper=T;
%TAR=number of target values used
TAR=51;
target=linspace(0,upper,TAR);

%multidimensional array to store computatation time of MinCovTarget
time_mc=zeros(N,res,SIM,TAR);
%multidimensional array to store envy found by MinCovTarget
envy_mc=zeros(N,res,SIM,TAR);
%multidimensional array to store % of max total welfare found by MinCovTarget
util_mc=zeros(N,res,SIM,TAR);
%multidimensional array to store maximum possible total welfare (does not
%depend on target)
maxsw=zeros(N,res,SIM);

%%%%%%%%%%%%%%%%%%%%%%%%SIMULATIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%here loop in agents
for ii=1:N
    n=agents(ii);
%print ii% to show progress
100*ii/N
% here loop in the number of goods
for jj=1:res
    d=xl(ii,jj);
%print d% to show progress
100*jj/res

%here loop for simulation of matrix value  
for uu=1:SIM 
    
%%%%%%%%%%%%%%%%%%%%%%VALUE MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V=VALUE MATRIX V consisting of the value v_{i,j} assigned by agent i to object j 
%in SPLIDDIT such matrix must have rowsums=T (agents can distribute T
%points amongst the goos with T=1000)

%alpha= expected ratio of items that each agent gives positive value to in
%case a more realistic generation of value matrices is used
V=valuematrixuniform(n,d,T);
%alpha=0.2;
% V=valuematrixrealistic(n,d,T,alpha);

%%%%%%%%%%%%%%%%%%%%%%MAX TOTAL WELFARE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% to maximize total welfare each item is given to (one of) the agent that values it most.
alloc=zeros(n,d);
[M,P]=max(V,[],1);
for iii=1:d
   alloc(P(iii),iii)=1;
end
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc.*repmat(V(k,:),n,1);
end
F=squeeze(sum(X,2));
maxsw(ii,jj,uu)=sum((diag(F)));


%here loop in the different target values
for kk=1:TAR  
%%%%%%%%%%%%%%%%%%%%%%%%MINCOVTARGET algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%creation of the 3-dim array X[i,j,k],i=1..n,j=1..d,k=1..n
%according to the mathematical framework defined in [2]
X=zeros(n,d,n);
for k=1:n
    for j=1:d
X(1,j,k)=V(k,j);
    end
end
%setting additional target columns
for k=1:n
X(k,d+1,k)=-target(kk);
end
beta=zeros(n,d);
for k=1:n
    for j=1:d
beta(k,j)=V(k,j)/V(1,j);
    end
end
%MinCov algorithm runs until no changes in objective for d consecutive iterations
counter=0;
%starting variance
variance=sum(var((sum(X,2)),1))/n;
tic
while counter<d
%randomly select an object j1 from the d items
j1 = randi([1 d],1,1);
%comparison vector - min with first occurrence
[M,I] = min((squeeze(sum(X,2)-X(:,j1,:)))*beta(:,j1));
%re-allocate object j1
X(:,j1,:)=0;
X(I,j1,:)=V(:,j1);
%variance objective functional
if sum(var((sum(X,2)),1))/n <variance
    variance=sum(var((sum(X,2)),1))/n;
    %variance
    counter=0;
else
    counter=counter+1;
end
end
%drop additional target column
X(:,(d+1),:) = [];
%computation of enlapsed time
time_mc(ii,jj,uu,kk)=toc;
%computation of envy
E=squeeze(sum(X,2));
envy_mc(ii,jj,uu,kk)=max(max(E-transpose(kron(diag(E),ones(1,n)))));
%computation of % of maximum total welfare attained
util_mc(ii,jj,uu,kk)=100*sum((diag(E)))/maxsw(ii,jj,uu);
%end of all four loops
end
end
end
end

% %%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE FOR COMP TIME AND ENVY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%x and y labels for each plot
yl=zeros(N,res);
ye=zeros(N,res);
yz=zeros(N,res);

%of no-envy allocation - best figure over all target values
tic;
[envy_min, target_min]=min(envy_mc,[],4);
add=toc;
%target value allowing for minimal envy 
%target_min=target(target_min);

%maximum total welfare attained by minimal envy allocations
totalw=zeros(N,res,SIM);
for ii=1:N
for jj=1:res
for uu=1:SIM
%%%over all minimal envy allocation choose max total welfare one
totalw(ii,jj,uu)=max(util_mc(ii,jj,uu,find(envy_mc(ii,jj,uu,:)==envy_min(ii,jj,uu))));
end
end
end

%envy as percentage of total value of goods
envy_min=100*envy_min/T;
%average envy
ye=sum(envy_min,3)/SIM;
%average computation time including selection of best target for envy
yl=sum(time_mc,3:4)/SIM+add/SIM;
%average % of max total welfare attained
yz=sum(totalw,3)/SIM;

%Save workspace
save('Figureshighdim.mat')
