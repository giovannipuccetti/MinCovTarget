% This MATLAB code compares various algorithms for the Fair Allocation of
% indivisible goods against various fairness criteria, in order to produce

%FigureBU.pdf

%ALGORITHMS COMPARED

% 1)  the SPLIDDIT algorithm as described in [1] Caragiannis et al. (2016) 
% 2)  the MinCovTarget+ algorithm

%HOW THE COMPARISON IS DESIGNED

% The comparison is performed on SIM simulation of a value matrix for
% SPLIDDIT and additionally on a set of TAR values of the target for
% MinCovTarget (the first target value is set =0 to reproduce MinCov in [2])

% Notice that this is not exactly the code for the algorithms in [2] but it
% is slighlty adapted to allow for a proper comparison with SPLIDDIT.
% In particular, the value matrix is created following SPLIDDIT rules.

%     References:
% [1] Caragiannis, I., D. Kurokawa, H. Moulin, A. D. Procaccia, N. Shah, and J. Wang (2019).
%     The unreasonable fairness of maximum nash welfare. 
%     ACM Transactions on Economics and Computation (TEAC) 7 (3), 1–32.
%     https://dl.acm.org/doi/10.1145/3355902
% [2] Cornilly, D., Puccetti, G., Rüschendorf, L., and S. Vanduffel (2020).  
%     Fair allocation of indivisible goods with minimum inequality or minimum envy criteria.
%     SSRN=https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512113

%CODE STARTS

%n=number of AGENTS, 
%d=number of indivisible GOODS, 
%SIM=number of REPETITIONS of simulation of value matrices
%TAR=number of different TARGET values

clear;

% d has to be larger than n, d>=n

n=10;
d=100;
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

%store a particular value matrix
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
log_util_mc(u,uu)=sum(log10(diag(E)));
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
prop_mc(NOTV,:)= [];
time_sp(:,NOTV)= [];
vari_sp(:,NOTV)= [];
envy_sp(:,NOTV)= [];
log_util_sp(:,NOTV)= [];
util_sp(:,NOTV)= [];
prop_sp(:,NOTV)= [];

% print % of success of SPLIDDIT
success=100*(1-NEWSIM/SIM);
%number of valid simulations
SIM=SIM-NEWSIM;


%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%comparison between MinCovTarget+ and Spliddit against fairness criteria

%tt=vector of target values to be considered for method 2)
%for each simulated matrix, we select the targets attaining
%minimal envy, and amongst them the one giving 
%maximal secondary criteria (can be changed)
tt=zeros(SIM,1);
tic;

envy_mct=min(envy_mc,[],2);
for u=1:SIM  
min_envy_index=find(envy_mc(u,:) == envy_mct(u));
%secondary objective function:max total welfare
%[M,m]=max(util_mc(u,min_envy_index));
%secondary objective function: max Nash welfare
[M,m]=max(log_util_mc(u,min_envy_index));
tt(u)=min_envy_index(m);
end

%computation time for selection of target to be added to MinCovTarget time
add=toc/SIM;
 
vari_mct=zeros(SIM,1);
envy_mct=zeros(SIM,1);
log_util_mct=zeros(SIM,1);
util_mct=zeros(SIM,1);
prop_mct=zeros(SIM,1);

for u=1:SIM  
vari_mct(u)=vari_mc(u,tt(u));
envy_mct(u)=envy_mc(u,tt(u));
%log_util_mct(u)=log_util_mc(u,tt(u));
log_util_mct(u)=100*10^(log_util_mc(u,tt(u)))/10^(log_util_sp(u));
util_mct(u)=util_mc(u,tt(u));
prop_mct(u)=prop_mc(u,tt(u));
end


%%%%Comparison between MinCovTarget+ and Spliddit

%On each box, the central mark indicates the median, the circle the mean, and the bottom and top edges of the box indicate the 25th and 75th 
%percentiles, respectively. The whiskers extend to the most extreme data points not considered outliers, 
% and the outliers are plotted individually using the '+' marker symbol. 
% The maximum whisker length specified as 1.0 times the interquartile range.

envy=[envy_mct,envy_sp'];
log_util=[log_util_mct,log_util_sp'];
prop=[prop_mct,prop_sp'];

epsi=0.1;

figure
%envy plot
t=tiledlayout(1,3);
title1=strcat('Scenario B, n= ',num2str(n), ', d= ',num2str(d),', uniform valuations. Comp. times: MCT+=',num2str(sum(sum(time_mc,2))/SIM+add,'%.2f'),' sec., SPL=',num2str(mean(time_sp),'%.2g'),' sec.');
title(t,title1, 'FontSize', 38)
t.TileSpacing = 'compact';
nexttile
Label1=strcat('MCT+, avg=',num2str(mean(envy(:,1)),'%.2f'));
Label2=strcat('SPL, avg=',num2str(mean(envy(:,2)),'%.2f'));
boxplot(envy,'Notch','off','Labels',{Label1;Label2},'Whisker',0)
hold on;
plot(mean(envy), '--o','MarkerSize',10,'MarkerEdgeColor','black')
ylim([-0.05 1])
yline(0,'--','No-Envy','Fontsize',16)
title('Envy')
set(gca,'Fontsize',28);

%Proportionality plot
nexttile
Label1=strcat('MCT+, avg=',num2str(mean(prop(:,1)),'%.2f'));
Label2=strcat('SPL, avg=',num2str(mean(prop(:,2)),'%.2f'));
boxplot(prop,'Notch','off','Labels',{Label1;Label2},'Whisker',0);
hold on;
plot(mean(prop), '--o','MarkerSize',10,'MarkerEdgeColor','black')
ylim([0 10.5])
yline(n,'--','Proportionality','Fontsize',16)
title('Agents with fair share')
set(gca,'Fontsize',28);

%Nash welfare plot
nexttile
Label3=strcat('MCT+, avg=',num2str(mean(log_util_mct),'%.2f'),'%');
boxplot(log_util_mct,'Notch','off','Labels',{Label3},'Whisker',0)
hold on;
plot(mean(log_util_mct), '--o','MarkerSize',10,'MarkerEdgeColor','black')
ylim([90 100.5])
yline(100,'--','Spliddit','Fontsize',16)
title('Maximum Nash welfare')
set(gca,'Fontsize',28);


% Print figure
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 25, 9], 'PaperUnits', 'Inches', 'PaperSize', [25, 9])
%select right name

saveas(gcf,'../Figures/FigureBU.pdf')

%Save workspace

save('BU.mat')
