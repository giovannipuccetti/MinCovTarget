%This code reproduces an example where Spliddit fails to 
%find the maximum Nash welfare allocation

%This is allocation 171 in the file AD.mat which reproduces
%the case of d=10 goods to be allocated to n=4 agents

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

V =[ 17   150   136   105    69   157    86    83    90   107;
     28   182   163    63    40   165    11    82   136   130;
     86   145   143    45    74   108   122    56   111   110;
     26   146   124    82    75   187    62   163    61    74];

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

%Maximum Nash welfare allocation

alloc_nw=zeros(n,d);
for ii=1:d
   alloc_nw(P(log_util_o_index,ii),ii)=1;
end
alloc_nw=reshape(alloc_nw,n,d);
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc_nw.*repmat(V(k,:),n,1);
end
vari_nw=sum(var(sum(X,2),1))/n;
%notice that in the paper we give transpose(E)
E_nw=squeeze(sum(X,2));
envy_nw=max(max(E_nw-transpose(kron(diag(E_nw),ones(1,n)))));
log_util_nw=sum(log10(diag(E_nw)));
util_sw=sum((diag(E_nw)));

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

log_util_sp
log_util_o

%SPLIDDIT ALLOCATION
% ans =
% 
%          0         0    1.0000    1.0000    1.0000         0         0         0         0         0
%          0    1.0000         0         0         0         0         0         0    1.0000         0
%     1.0000         0         0         0    0.0000         0    1.0000         0         0    1.0000
%          0         0         0         0         0    1.0000         0    1.0000         0         0

%The optimal allocation wrt Nash welfare swaps item 3 and 10 to obtain a
%slighlty higher value
% 
% alloc_nw =
% 
%      0     0     0     1     1     0     0     0     0     1
%      0     1     0     0     0     0     0     0     1     0
%      1     0     1     0     0     0     1     0     0     0
%      0     0     0     0     0     1     0     1     0     0


% Notice that the optimal Nash welfare allocation is detected by MINCOVTARGET+ for the 40th
% value of the target in the code FAID.m -- that's why there is a 99.9% in
% the Spliddit success rate in the upper-right plot of Figure 4

%reshape(alloc_mct(171,40,:,:,1),n,d)
% ans =
% 
%      0     0     0   105    69     0     0     0     0   107
%      0   150     0     0     0     0     0     0    90     0
%     17     0   136     0     0     0    86     0     0     0
%      0     0     0     0     0   157     0    83     0     0
