% This MATLAB function generates a value matrix for 
% the problem of Fair Allocation of indivisible items

% valuematrixuniform = random integral valuation
% valuematrixrealistic= more realistic valuation

%the function accepts 4 inputs:
%n=number of AGENTS, 
%d=number of indivisible GOODS, 
%T=SPLIDDIT parameter
%alpha= expected ratio of items that each agent gives positive value to


%example: valuematrixrealistic(4,10,1000,0.1)

function z=valuematrixrealistic(n,d,T,alpha)

%%%%%%%%%%%%%%%%%%%%%%REALISTIC VALUE MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V=VALUE MATRIX V consisting of the value v_{i,j} assigned by agent i to object j 
%in SPLIDDIT such matrix must have rowsums=T (agents can distribute T
%points amongst the goos with T=1000 in the original version)

%We first generate uniformly random integral valuations between 0 and T
V=T*rand(n,d);

%the number of items each player gives a positive value to is random and
%follows a truncated negative binomial distribution

%parameters of the negative binomial, chosen so that 
%the mean=alpha*d and variance=2*mean
R=alpha*d;
P=0.5;
pd = makedist('NegativeBinomial','R',R,'P',P);
%truncation to allow values between 1 and d
t = truncate(pd,1,d);
%item_null=number of items each player gives null value 
item_null=d-random(t,n,1);
%for each agent, the value of item_null objects (selected at random) is set
%equal to zero;
for i=1:n
V(i,randperm(d,item_null(i)))=0;
end
%rounding to integers such that each row of V sums up to T
V=round(T*bsxfun(@rdivide, V, sum(V,2)));
%We extract and remember the first row of V to allow for a different treatment for agent 1 
VV=V(1,:);
V(1,:)=[];
% we adjust remaining rows so that sum of the rows is equal to T
counter=0;
%max number of adjustment iterations
max_counter=n*d;
%adjustment of agents 2 to n to make sum of valuations equal to T
while(min(min(V))<0|(max(sum(V,2))-min(sum(V,2))>0))
  rS=sum(V,2);
for i=1:(n-1)
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
      error('Impossible to create value matrix due to SPIDDIT rules1');
      return;
  end
%end of while
end

%for agent 1, the value of item_null objects (selected at random) is set
%equal to 1;
VV(VV==0)=1;
%adjustment of agent 1 to n to make sum of valuations equal to T
counter=0;
while not(sum(VV) == T)
     [M,I] = max(VV);
     VV(I)=max(1,VV(I)-sum(VV)+T);
     counter=counter+1;
  if counter>max_counter
      error('Impossible to create value matrix due to SPIDDIT rules2');
      return;
  end
end

%final matrix
V=[VV;V];

%final doublecheck (this can be dropped) 
 if max(sum(V,2))-min(sum(V,2))>0
      error('Impossible to create value matrix due to SPIDDIT rules3');
      return;
 end
  if min(V(1,:))<1
      error('Impossible to create value matrix due to SPIDDIT rules4');
      return;
  end

 z=V;  