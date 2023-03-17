% This MATLAB function compares various algorithms for the Fair Allocation of
% indivisible goods:

% 1)  the SPLIDDIT algorithm as described in [1] Caragiannis et al. (2016) 
% 2)  the MinCov and MinCovTarget algorithms as described in [2] Cornilly et al. (2021)

% The comparison is performed on SIM simulation of a value matrix for
% SPLIDDIT and additionally on sifferent TAR values of the target for
% MinCovTarget (the first target value is set =0 to reproduce MinCov)

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

% The code produces two figures (Figures 3-5 and 4-6 in the paper 
% Puccetti, G., Rüschendorf, L., and S. Vanduffel (2023)
% A comparison of algorithms for multi-player fair allocation of indivisible goods)
%

%the function accepts 4 inputs:
%n=number of AGENTS, 
%d=number of indivisible GOODS, 
%SIM=number of REPETITIONS of simulation of value matrices
%TAR=number of different TARGET values

function g=FAID(n,d,SIM,TAR)

%example tp obtain Figures 3 and 5
%FAID(4,10,1000,51)

%fix the random generator seed
rng(1);
%T=SPIDDIT total value (in principle this could be changed, notice however
%that computation time for the SPLIDDIT heavily depends on this value) (must be even) 
T=1000;

%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%One has to set: #AGENTS, #GOODS, #SIMULATIONS, #TARGET VALUES

%example if one wants toi run it directly 
%n=number of AGENTS
%n=20;
%d=number of indivisible GOODS
%d=200;
%SIM=1000;
%TAR=51;

%for a comparison amonst the algorithms
%(which are based on slighly different assumption), let d>=n
  if d<n
      error('Number of objects is less than the number of players');
      return;
  end
%upper=upperbound for target value 
upper=T;
target=linspace(0,upper,TAR);
%matrices and vectors to store results
time_mc=zeros(SIM,TAR);
vari_mc=zeros(SIM,TAR);
log_util_mc=zeros(SIM,TAR);
util_mc=zeros(SIM,TAR);
envy_mc=zeros(SIM,TAR);
time_sp=zeros(1,SIM);
vari_sp=zeros(1,SIM);
log_util_sp=zeros(1,SIM);
util_sp=zeros(1,SIM);
envy_sp=zeros(1,SIM);
exitp=zeros(1,SIM);
exitpp=zeros(1,SIM);

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

%more realistic matrix
%alpha= expected ratio of items that each agent gives positive value to in
%case a more realistic generation of value matrices is used
%alpha=0.2;
%V=valuematrixrealistic(n,d,T,alpha);

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
tic
%we bound the max time that MILP can use at each iteration to max_seconds seconds
max_seconds=120;
options = optimoptions('intlinprog','Display','off','MaxTime',max_seconds);
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
%end of SPLIDDIT

%loop for target values to be used in MinCovTarget
%since the first target value is null, the first iteration runs the MINCOV algorithm 
for uu=1:TAR
tau=repmat(target(uu), n, 1);

%%%%%%%%%%%%%%%%%%%%%%%%MINCOVTARGET algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
%variance objective functional as defined in [2]
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
100*(1-NEWSIM/SIM)
%number of valid simulations
SIM=SIM-NEWSIM;

%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%collecting mean values for MinCovTarget across the SIM iterations left

time_m=sum(time_mc,1)/SIM;
vari_m=sum(vari_mc,1)/SIM;
envy_m=sum(envy_mc,1)/SIM;
util_m=sum(util_mc,1)/SIM;

%for computing log_util_m, we exclude -Inf values
%(seems to be relevant only for realistic matrix simulation)
for ij=1:TAR
NOLOG=find(log_util_mc(:,ij)~=-Inf);
log_util_m(ij)=mean(log_util_mc(NOLOG,ij));
end

%create vector of values for SPLIDDIT

time_s=repmat(mean(time_sp),1,TAR);
vari_s=repmat(mean(vari_sp),1,TAR);
envy_s=repmat(mean(envy_sp),1,TAR);
log_util_s=repmat(mean(log_util_sp),1,TAR);
util_s=repmat(mean(util_sp),1,TAR);

%produce four plots of inequality/envy/log-utility/utility against choice of target
%values
% Create 1-by-4 tiled chart layout
figure;
tiledlayout(1,4);

font_size=10;

% Left plot (variance)
ax1 = nexttile;
plot(ax1,target,vari_s,'r',target,vari_m,'k',target(1),vari_m(1),'ob','MarkerSize',18,'LineWidth',2);
xlabel(ax1,'Target values');
ylabel(ax1,'Inequality');
lgd=legend('SPLIDDIT','MinCovTarget','MinCov');
legend('Location','southeast');
lgd.FontSize = font_size;
title(ax1,'Inequality');

% Middle left plot (envy)
ax2 = nexttile;
plot(ax2,target,envy_s,'r',target,envy_m,'k',target(1),envy_m(1),'ob','MarkerSize',18,'LineWidth',2);
xlabel(ax2,'Target values');
ylabel(ax2,'Envy');
lgd=legend('SPLIDDIT','MinCovTarget','MinCov');
legend('Location','northeast');
lgd.FontSize = font_size;
title(ax2,'Envy');

% Middle right plot (Total Log-Utility)
ax3 = nexttile;
plot(ax3,target,log_util_s,'r',target,log_util_m,'k',target(1),log_util_m(1),'ob','MarkerSize',18,'LineWidth',2);
xlabel(ax3,'Target values');
ylabel(ax3,'Total Log-Utility');
lgd=legend('SPLIDDIT','MinCovTarget','MinCov');
legend('Location','southeast');
lgd.FontSize = font_size;
title(ax3,'Log-Utility');

% Right plot (Total Utility)
ax4 = nexttile;
plot(ax4,target,util_s,'r',target,util_m,'k',target(1),util_m(1),'ob','MarkerSize',18,'LineWidth',2);
xlabel(ax4,'Target values');
ylabel(ax4,'Total Utility');
lgd=legend('SPLIDDIT','MinCovTarget','MinCov');
legend('Location','southeast');
lgd.FontSize = font_size;
title(ax4,'Utility');

% Print figure

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], 'PaperUnits', 'Inches', 'PaperSize', [12, 6])

%select right name
%uniform scenarios (Figure 4 in the paper)
saveas(gcf,'FigureAU1.pdf')
%saveas(gcf,'FigureBU1.pdf')
%saveas(gcf,'FigureCU1.pdf')

%realistic scenarios (Figure 6 in the paper)
%saveas(gcf,'FigureAR1.pdf')
%saveas(gcf,'FigureBR1.pdf')
%saveas(gcf,'FigureCR1.pdf')

%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
%computation of best method for various objective functions 
%three methods considered: 
%1) MinCovTarget with target value index 1 (=MinCov) 
%2) MinCovTarget with target value corresponding to minimal envy 
%3) MinCovTarget with largest target value
%4) SLPIDDIT

%vari=#success for variance for all the above methods
vari=zeros(1,4);
%envy=#success for envy for all the above  methods
envy=zeros(1,4);
%util=#success for log-utility for all the above methods
log_util=zeros(1,4);
%util=#success for utility for all the above methods
util=zeros(1,4);

%tt=target value index to be considered for method 3)
[minenvy,tt]=min(envy_m);
%find minimal variance across all methods
VA=[vari_mc(:,1),vari_mc(:,tt),vari_mc(:,TAR),vari_sp'];
[I_VA,MI_VA]=min(VA, [], 2);
%find minimal envy across all methods
EN=[envy_mc(:,1),envy_mc(:,tt),envy_mc(:,TAR),envy_sp'];
[I_EN,MI_EN]=min(EN, [], 2);
%find maximal total log-utility across all methods
LOG_UT=[log_util_mc(:,1),log_util_mc(:,tt),log_util_mc(:,TAR),log_util_sp'];
[I_LOG_UT,MI_LOG_UT]=max(LOG_UT, [], 2);
%find maximal total utility across all methods
UT=[util_mc(:,1),util_mc(:,tt),util_mc(:,TAR),util_sp'];
[I_UT,MI_UT]=max(UT, [], 2);
%count successes of each method against the four objective functions
for i=1:SIM
    for j=1:4
    %count variance success
    if VA(i,j)==VA(i,MI_VA(i))
        vari(j)=vari(j)+1;
    end
    %count envy success
    if EN(i,j)==EN(i,MI_EN(i))
        envy(j)=envy(j)+1;
    end
    %count log-utility success
    if LOG_UT(i,j)==LOG_UT(i,MI_LOG_UT(i))
        log_util(j)=log_util(j)+1;
    end
    %count utility success
    if UT(i,j)==UT(i,MI_UT(i))
        util(j)=util(j)+1;
    end
    %end of double for
    end
end

%add epsilon to 0 values t show them in the bar graph
eps=SIM/100;
vari(vari==0)=eps;
envy(envy==0)=eps;
util(util==0)=eps;
log_util(log_util==0)=eps;
%bar graph creation
X = categorical({'Inequality','Envy','Log-Utility','Utility'});
X = reordercats(X,{'Inequality','Envy','Log-Utility','Utility'});
y = round([vari*100/SIM; envy*100/SIM; log_util*100/SIM; util*100/SIM],1);
bbar = bar(X,y,'FaceColor','flat');
%format computation times for display
leg1=strcat('MinCov (',num2str(time_m(1),'%.2g'), ' s.)');
leg4=strcat('SPLIDDIT (',num2str(time_s(1),'%.2g'), ' s.)');
set(bbar, {'DisplayName'}, {leg1,'MinCovTarget1','MinCovTarget2',leg4}')
for k = 1:size(y,2)
    bbar(k).CData = k;
end
ylim([0 105])
legend('Location','bestoutside')
title1=strcat('n= ',num2str(n), ', d= ',num2str(d));
title(title1);


%add labels
xtips1 = bbar(1).XEndPoints;
ytips1 = bbar(1).YEndPoints;
labels1 = string(bbar(1).YData);
labels1(labels1=='1')='0';
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips2 = bbar(2).XEndPoints;
ytips2 = bbar(2).YEndPoints;
labels2 = string(bbar(2).YData);
labels2(labels2=='1')='0';
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips3 = bbar(3).XEndPoints;
ytips3 = bbar(3).YEndPoints;
labels3 = string(bbar(3).YData);
labels3(labels3=='1')='0';
text(xtips3,ytips3,labels3,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips4 = bbar(4).XEndPoints;
ytips4 = bbar(4).YEndPoints;
labels4 = string(bbar(4).YData);
labels4(labels4=='1')='0';
text(xtips4,ytips4,labels4,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gca,'Fontsize',14);


% Print figure
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], 'PaperUnits', 'Inches', 'PaperSize', [12, 6])
%select right name
%uniform scenarios (Figure 3 in the paper)
saveas(gcf,'FigureAU2.pdf')
%saveas(gcf,'FigureBU2.pdf')
%saveas(gcf,'FigureCU2.pdf')

%realistic scenarios (Figure 5 in the paper)
%saveas(gcf,'FigureAR2.pdf')
%saveas(gcf,'FigureBR2.pdf')
%saveas(gcf,'FigureCR2.pdf')
%%%end
