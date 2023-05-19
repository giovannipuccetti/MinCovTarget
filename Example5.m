%This code reproduces the pedagogical example given in Section 5

clear all

%fix the random generator seed for reproducibility
rng(1);
%n=number of agents
n=4;
%d=number of goods
d=10;
%SPLIDDIT parameter (total value of goods)
T=1000;

%V=value matrix

V = [
   121    94   141   142    63    97   101    97    41   103;
   193    21   205   103   195     8   161    36    10    68;
    23    51    29   145   144   154   135   128    18   173;
   159    95   169    25   167   162    68     6   143     6];

%vector representing all the possible allocations of the goods
%uses permn function in https://it.mathworks.com/matlabcentral/fileexchange/7147-permn
%to be placed in working directory

P=permn(1:n,d);
N=n^d;

%%%%%%% BRUTE FORCE ALGORITHM  %%%%%%%

%vector to store brute force results
vari_bf=zeros(N,1);
envy_bf=zeros(N,1);
log_util_bf=zeros(N,1);
util_bf=zeros(N,1);

tic
for i=1:N
alloc=zeros(n,d);
for ii=1:d
   alloc(P(i,ii),ii)=1;
end
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc.*repmat(V(k,:),n,1);
end
vari_bf(i)=sum(var(sum(X,2),1))/n;
E=squeeze(sum(X,2));
envy_bf(i)=max(max(E-transpose(kron(diag(E),ones(1,n)))));
log_util_bf(i)=sum(log10(diag(E)));
util_bf(i)=sum((diag(E)));
end
time_bf=toc;

%tolerance 
eps=0.00001;

%find (exact) maximum Nash welfare and all allocations attaining it
log_util_o=max(log_util_bf);
log_util_o_index=find(log_util_bf>=log_util_o-eps);
%find (exact) maximum social welfare and all allocations attaining it
util_o=max(util_bf);
util_o_index=find(util_bf>=util_o-eps);
%find (exact) minimum envy and all allocations attaining it
envy_o=min(envy_bf);
envy_o_index=find(envy_bf<=envy_o+eps);
%find (exact) minimum inequality and all allocations attaining it
vari_o=min(vari_bf);
vari_o_index=find(vari_bf<=vari_o+eps);

%Maximum social welfare allocation

alloc_sw=zeros(n,d);
for ii=1:d
   alloc_sw(P(util_o_index,ii),ii)=1;
end
alloc_sw=reshape(alloc_sw,n,d);
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc_sw.*repmat(V(k,:),n,1);
end
vari_sw=sum(var(sum(X,2),1))/n;
%notice that in the paper we give transpose(E)
E_sw=squeeze(sum(X,2));
envy_sw=max(max(E_sw-transpose(kron(diag(E_sw),ones(1,n)))));
log_util_sw=sum(log10(diag(E_sw)));
util_sw=sum((diag(E_sw)));

%Analysis of minimum inequality allocations

[MM,mm]=min(envy_bf(vari_o_index));
util_bf(vari_o_index(mm));

%%%%%%% SPLIDDIT ALGORITHM  %%%%%%%

%linear objective function, maximize sum of agents log utilities (last n
%continuous variables)
f=-[sparse(n*d,1);ones(n,1)];
%integral variables are the first n*d
intcon=1:n*d;
%lower bound on integral variables
lb=[sparse(n*d,1);-Inf*ones(n,1)];
%upper bound on integral variables
ub=[ones(n*d,1);Inf*ones(n,1)];
%d equality constraints, each item is allocated to exactly one agent
Aeq=[kron(ones(1,n),speye(d)),sparse(d,n)];
beq=ones(d,1);

%SPLIDDIT MILP problem as described in Caragiannis et al. (2016) 
%we have a vector of n*d integer variable plus n continuous variables

%additional constraints that depend on the value matrix V

%first n inequality constraints, each agents has at least a value of 1 so that the log utility is at least 0
A=[-kron(ones(1,n),V).*kron(speye(n),ones(1,d)),sparse(n,n)];
b=-ones(n,1);
%second T/2 inequality constraints, describing SPLIDDIT approximation;
for r=1:T/2
    K=2*r-1;
A2=[(log10(K)-log10(K+1))*(kron(ones(1,n),V).*kron(speye(n),ones(1,d))),speye(n)];
b2=(log10(K)-K*(log10(K+1)-log10(K)))*ones(n,1);
%final matrix A for inequality constraint
A=[A;A2];
b=[b;b2];
end

%change Spliddit constraints to maxime social welfare
%just n inequality constraint
%A=[-(kron(ones(1,n),V).*kron(speye(n),ones(1,d))),speye(n)];
%b=sparse(n,1);

%MILP SOLUTION
%we bound the max time that MILP can use at each iteration to max_seconds seconds
max_seconds=120;
options = optimoptions('intlinprog','Display','off','MaxTime',max_seconds);
tic
[x,fval,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,[],options);
%computation of enlapsed time
time_sp=toc;

%store the quality of output of MILP
%most frequent cases are:
%exitpp=1 intlinprog converged to the solution x.
%exitpp=0, -2 intlinprog stopped prematurely (0). No integer feasible point found. 
%exitpp=2 intlinprog stopped prematurely (probably because of time limit). Integer feasible point found.

%reshape x into matrix xij to obtain an allocation matrix
%following the notation in the paper
alloc_sp=transpose(reshape(x(1:(n*d)),d,n));
%creation of the 3-dim array X[i,j,k],i=1..n,j=1..d,k=1..n
%according to the mathematical framework defined in [2]
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc_sp.*repmat(V(k,:),n,1);
end
%computation of variance functional 
vari_sp=sum(var(sum(X,2),1))/n;
%sum of log utilities obtained by agents 
E_sp=squeeze(sum(X,2));
log_util_sp=sum(log10(diag(E_sp)));
%sum of utilities obtain by agents 
util_sp=sum((diag(E_sp)));
%computation of envy
envy_sp=max(max(E_sp-transpose(kron(diag(E_sp),ones(1,n)))));
%end of SPLIDDIT

%Nash welfare solution found by Brute Force

alloc_nw=zeros(n,d);
for ii=1:d
   alloc_nw(P(log_util_o_index,ii),ii)=1;
end

%check that it is the same
max(alloc_nw-alloc_sp);

%%%%%%%%%%%%%%%%%%%%%%%%MINCOVTARGET+ algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%vector of target values
TAR=51;
upper=2000;
target=linspace(0,upper,TAR);

%vector to store MinCovTarget results

vari_mc=zeros(TAR,1);
envy_mc=zeros(TAR,1);
log_util_mc=zeros(TAR,1);
util_mc=zeros(TAR,1);
alloc_mt=zeros(TAR,n,d);
for uu=1:TAR
tau=repmat(target(uu), n, 1);
%creation of the 3-dim array X[i,j,k],i=1..n,j=1..d,k=1..n
%according to the mathematical framework defined in [2]
X=zeros(n,d,n);
for k=1:n
    for j=1:d
X(1,j,k)=V(k,j);
    end
end
%setting additional target column
for k=1:n
X(k,d+1,k)=-tau(k);
end
beta=zeros(n,d);
for k=1:n
    for j=1:d
beta(k,j)=V(k,j)/V(1,j);
    end
end

%starting allocation
alloc_mt(uu,1,:)=ones(1,d);

%MinCovTarget algorithm runs until no changes in objective for d consecutive iterations
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
alloc_mt(uu,:,j1)=0;
alloc_mt(uu,I,j1)=1;
%change allocation
%variance objective functional as defined in [2]
if sum(var((sum(X,2)),1))/n <variance
    variance=sum(var((sum(X,2)),1))/n;
    counter=0;
else
    counter=counter+1;
end
end
time_mc(uu)=toc;
%drop additional target column
X(:,(d+1),:) = [];
%computation of variance functional
vari_mc(uu)=sum(var(sum(X,2),1))/n;
%sum of log utilities obtain by agents 
E=squeeze(sum(X,2));
log_util_mc(uu)=sum(log10(diag(E)));
%sum of utilities obtain by agents 
util_mc(uu)=sum((diag(E)));
%computation of envy
envy_mc(uu)=max(max(E-transpose(kron(diag(E),ones(1,n)))));
%end of loop in uu (target values)
end

%find optimal allocations for MinCovTarget
%for envy first, then social welfare
tic
envy_mct=min(envy_mc);
min_envy_index=find(envy_mc == envy_mct);
[M,m]=max(util_mc(min_envy_index));
%[M,m]=max(log_util_mc(min_envy_index));
tt=min_envy_index(m);
time_mct=sum(time_mc)+toc;

envy_mct=envy_mc(tt);
util_mct=util_mc(tt);
log_util_mct=log_util_mc(tt);
vari_mct=vari_mc(tt);
alloc_mct=reshape(alloc_mt(tt,:,:),n,d);
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc_mct.*repmat(V(k,:),n,1);
end
E_mct=squeeze(sum(X,2));

%find optimal allocations for MinCov (Target value=0)
time_mcov=time_mc(1);
envy_mcov=envy_mc(1);
util_mcov=util_mc(1);
log_util_mcov=log_util_mc(1);
vari_mcov=vari_mc(1);
alloc_mcov=reshape(alloc_mt(1,:,:),n,d);
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc_mcov.*repmat(V(k,:),n,1);
end
E_mcov=squeeze(sum(X,2));

%AUXILIARY STATISTICS

%average level of the envy found by MinCovTarget+ over all the target
%values
mean(envy_mc);
%maximum level of social welfare attainable by a no-envy allocation
[R,r]=max(util_bf(envy_o_index));
optimal_solution_index=find((util_bf == R) & (envy_bf == 0));
P(optimal_solution_index,:);

%Save workspace
save('Example5.mat')