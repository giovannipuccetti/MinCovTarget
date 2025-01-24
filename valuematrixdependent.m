% This MATLAB function generates a value matrix for 
% the problem of Fair Allocation of indivisible items

% valuematrixuniform = random integral valuation
% valuematrixrealistic= more realistic valuation
% valuematrixcopula=dependent valuations

%the function accepts 4 inputs:
%n=number of AGENTS, 
%d=number of indivisible GOODS, 
%T=SPLIDDIT parameter
%paiwise correlation amongst agents' valuations of the same good 

function z=valuematrixdependent(n,d,T,rho)

%%%%%%%%%%%%%%%%%%%%%%UNIFORM VALUE MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V=VALUE MATRIX V consisting of the value v_{i,j} assigned by agent i to object j 
%in SPLIDDIT such matrix must have rowsums=T (agents can distribute T
%points amongst the goos with T=1000)
%We generate uniform possibly dependent random integral valuations.

%rho_corr=correlation matrix
rho_corr=rho*ones(n,n);
v=ones(1,n);
rho_corr(1:(n+1):end) = v;
%simulation of Value matrix where each column is sampled from a Gaussian copula with given pairwise constant
%correlation rho
V=copularnd('Gaussian',rho_corr,d)';


V=round(T*bsxfun(@rdivide, V, sum(V,2)));
%make rowSums of V equal to T and V non-negative
counter=0;
%max number of adjustment iterations
max_counter=n;
while((min(min(V))<0)||(max(sum(V,2)~=T))||(min(sum(V,2))~=T))
  rS=sum(V,2);
for i=1:n
  if rS(i)>T
     [M,I] = max(V(i,:));
     V(i,I)=max(0,V(i,I)-rS(i)+T);
  end
  if rS(i)<T
      [m,I] = min(V(i,:));
      V(i,I)=min(T,V(i,I)-rS(i)+T);
  end
%end of for
  end
  counter=counter+1;
  if counter>max_counter
      error('Impossible to create value matrix due to SPIDDIT rules');
      return;
  end
%end of while
end

%Require that at leat one agent gives strictly positive utilities
%to all object (this is required by mincov/mincovtarget algoriths)
%this agent becomes player 1. This only matters when n is similar to d,
%and one cannot choose epsilon values because of the SPLIDDIT assumptions
P=min(V,[],2);
%if the first agent gives null value to at least one object
if min(V(1,:))==0
%if possible, WLOG put as first agent one giving positive values to all objects
if max(P)>0
[v,M]=max(P);
buf=V(1,:);
V(1,:)=V(M,:);
V(M,:)=buf;
end
end

% if all agents give 0 value to at least one item, re-define agent-1's values 
if max(P)==0
% we keep track of this 
%exitp(u)=1;
V(1,:)=ones(1,d);
if d>n
for i=1:(T-sum(V(1,:)))
    j=randi([1 d],1,1);
    V(1,j)=V(1,j)+1;
end
end
end

%final doublecheck (this can be dropped) 
 if ((min(min(V))<0)||(max(sum(V,2)~=T))||(min(sum(V,2))~=T))
      error('Impossible to create value matrix due to SPIDDIT rules');
      return;
 end

  z=V;