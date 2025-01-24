% This code plots the score function defined in Section 7 of the paper
% for a single scenario

%CODE STARTS

%n=number of AGENTS, 
%d=number of indivisible GOODS, 
%SIM=number of REPETITIONS of simulation of value matrices
%TAR=number of different TARGET values

clear;

% Choice of the scenario
% d has to be larger than n, d>=n

n=20;
d=200;
SIM=1000;
TAR=51;

%fix the random generator seed
rng(1);
%T=SPIDDIT total value (in principle this could be changed, notice however
%that computation time for the SPLIDDIT heavily depends on this value which must be even) 
T=1000;
%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%One has to set: #AGENTS, #GOODS, #SIMULATIONS, #TARGET VALUES

%upper=upperbound for target value 
upper=2*T;
target=linspace(0,upper,TAR);
%matrices and vectors to store results
time_mc=zeros(SIM,TAR);
vari_mc=zeros(SIM,TAR);
log_util_mc=zeros(SIM,TAR);
util_mc=zeros(SIM,TAR);
envy_mc=zeros(SIM,TAR);
prop_mc=zeros(SIM,TAR);
time_sp=zeros(1,SIM);
vari_sp=zeros(1,SIM);
log_util_sp=zeros(1,SIM);
util_sp=zeros(1,SIM);
envy_sp=zeros(1,SIM);
prop_sp=zeros(1,SIM);
exitp=zeros(1,SIM);
exitpp=zeros(1,SIM);
value=zeros(SIM,n,d);

%%%SPLIDDIT CONSTRAINTS MATRIX 
%%%that do not depend on value matrix V

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

%here loop for the simulation of value matrix V
for u=1:SIM  
    %print % of progress
    100*u/SIM
%exitp parameter (see creation of value matrix)
exitp(u)=0;

%%%%%%%%%%%%%%%%%%%%%%VALUE MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V=VALUE MATRIX V consisting of the value v_{i,j} assigned by agent i to object j 
%in SPLIDDIT such matrix must have rowsums=T (agents can distribute T
%points amongst the goos with T=1000)

%uniform valued matrix
V=valuematrixuniform(n,d,T);

%dependent valuations
%rho=0.5;
%V=valuematrixdependent(n,d,T,rho);

%store the value matrix
%value(u,:,:)=V;

%%%%%%%%%%%%%%%%%%%%%%%%SPLIDDIT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%SPIDDIT MILP problem as described in Caragiannis et al. (2016) 
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

%MILP SOLUTION
%we bound the max time that MILP can use at each iteration to max_seconds seconds
max_seconds=120;
options = optimoptions('intlinprog','Display','off','MaxTime',max_seconds);
tic
[x,fval,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,[],options);
%computation of enlapsed time
time_sp(u)=toc;

%store the quality of output of MILP
%most frequent cases are:
%exitpp=1 intlinprog converged to the solution x.
%exitpp=0, -2 intlinprog stopped prematurely (0). No integer feasible point found. 
%exitpp=2 intlinprog stopped prematurely (probably because of time limit). Integer feasible point found.

exitpp(u)=exitflag;

%if the MILP algorithm did not converge we set a fake allocation 
%that later on will be discarded
if not(exitflag == 1) 
x=ones(d*n+n);
end

%reshape x into matrix xij to obtain an allocation matrix
%following the notation in the paper
alloc=transpose(reshape(x(1:(n*d)),d,n));

%doublecheck (this can be dropped)
%sum of log utilities obtain by agents under alloc
%sumu=sum(log10(sum(V.*alloc,2)));
%sum of W[i] as defined in SPLIDDLIT
%sumw=(sum(x((n*d+1):(n*d+n))));
% if abs(sumu-sumw)>0.01
%      error('SPLIDDIT solution not found by MILP');
%       return;
% end

%creation of the 3-dim array X[i,j,k],i=1..n,j=1..d,k=1..n
%according to the mathematical framework defined in [2]
X=zeros(n,d,n);
for k=1:n
X(:,:,k)=alloc.*repmat(V(k,:),n,1);
end
%computation of variance functional 
vari_sp(u)=sum(var(sum(X,2),1))/n;
%sum of log utilities obtained by agents 
E=squeeze(sum(X,2));
log_util_sp(u)=sum(log10(diag(E)));
%sum of utilities obtain by agents 
util_sp(u)=sum((diag(E)));
%computation of envy
envy_sp(u)=max(max(E-transpose(kron(diag(E),ones(1,n)))));
%computation of #agents who receive no less than a fair share
prop_sp(u)=sum(diag(E)>=T/n);
%end of SPLIDDIT

if (u==896)
   EEE=E;
XXX=X;
end
%%%%%%%%%%%%%%%%%%%%%%%%MINCOVTARGET algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loop for target values to be used in MinCovTarget
%since the first target value is null, the first iteration runs the MINCOV algorithm 

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
%variance objective function
if sum(var((sum(X,2)),1))/n <variance
    variance=sum(var((sum(X,2)),1))/n;
    counter=0;
else
    counter=counter+1;
end
end
%drop additional target column
X(:,(d+1),:) = [];
%computation of enlapsed time
time_mc(u,uu)=toc;
%computation of variance functional
vari_mc(u,uu)=sum(var(sum(X,2),1))/n;
%sum of log utilities obtain by agents 
E=squeeze(sum(X,2));
%log utilities reached as a percentage of max Nash welfare
%log_util_mc(u,uu)=sum(log10(diag(E)))/log_util_sp(u);
log_util_mc(u,uu)=10^(sum(log10(diag(E))))/10^(log_util_sp(u));
%sum of utilities obtain by agents 
util_mc(u,uu)=sum((diag(E)));
%computation of envy
envy_mc(u,uu)=max(max(E-transpose(kron(diag(E),ones(1,n)))));
%computation of #agents who receive no less than a fair share
prop_mc(u,uu)=sum(diag(E)>=T/n);

%end of loop in uu (target values)
end
%end of loop in u  (simulations of value matrix)
end

%%%%%%%%%%%%%%%%%%%%%%%% ALLOCATION CHECK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% we drop SPLIDDIT solution that do not have exitflag=1
% we re-count all SIM and we keep track of this number

%number of not valid simulations
NEWSIM=size(exitpp(exitpp~=1),2);
%set of indexes of not valid simulation
NOTV=find(exitpp~=1);
%drop not valid simulations
time_mc(NOTV,:)= [];
vari_mc(NOTV,:)= [];
envy_mc(NOTV,:)= [];
log_util_mc(NOTV,:)= [];
util_mc(NOTV,:)= [];
time_sp(:,NOTV)= [];
vari_sp(:,NOTV)= [];
envy_sp(:,NOTV)= [];
log_util_sp(:,NOTV)= [];
util_sp(:,NOTV)= [];

% print % of success of SPLIDDIT
success=100*(1-NEWSIM/SIM);
%number of valid simulations
SIM=SIM-NEWSIM;


%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Comparison between MinCovTarget+ and Spliddit
%produce one plot showing the score function against target values

%collecting mean values for MinCovTarget across the SIM valid simulations
%only used in the paper for n=20, d=200
envy_m=sum(envy_mc,1)/SIM;
%rescaling
envy_m=(T-envy_m)/T;
%for computing log_util_m, we exclude -Inf values
%(seems to be relevant only for realistic matrix simulation)
for ij=1:TAR
NOLOG=find(log_util_mc(:,ij)~=-Inf);
log_util_m(ij)=mean(log_util_mc(NOLOG,ij));
end

%score function
%weight for envy (weight for Nash welfare will be 1-alpha)
alpha=0.50;

%values
figure;
t=tiledlayout(1,1);

title1=strcat('n= ',num2str(n), ', d= ',num2str(d), ', uniform valuations');
%title1=strcat('n= ',num2str(n), ', d= ',num2str(d), ', dependent');
title(t,title1,'fontweight','bold','fontsize',16);

font_size=10;

% Plot of score function
ax1 = nexttile;
plot(ax1,target,alpha*envy_m+(1-alpha)*log_util_m,'-','MarkerSize',18,'LineWidth',2);
xlabel(ax1,'Target values');
%lgd=legend('SPLIDDIT','MinCovTarget','MinCov');
%legend('Location','southeast');
lgd.FontSize = font_size;
title(ax1,'Score function');

% Print figure

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], 'PaperUnits', 'Inches', 'PaperSize', [12, 6])

saveas(gcf,'../Figures/FigureScore.pdf')

%Save workspace

save('Score.mat')


