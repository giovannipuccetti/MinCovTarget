%This is an auxiliary code 
%which plots Figure 6 about the average computation time
%of MinCovTarget+ and the % of no-envy allocation found
%over a number SIM of random simulations of the value matrix 

clear;
%fix the random generator seed
rng(1);
%T=Goods total value (value matrix standardized to this value;
% set T=10000 when d>=300 to have more meaningful valuations) 
T=1000;

%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% agents is a vector containing the FIXED numbers n of agents for which to
% draw the plot

%%%%%for FIGURE 7%%%%%
% draw the plot versus the number d of goods
agents=[10 20 50];
N=size(agents,2);
% for each number of agents we give a vector with number of goods at which
% to compute values
xl=[50 100 150 200 250;
    50 100 150 200 250;
    50 100 150 200 250;];
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
%multidimensional array to store % of no-envy solution found by MinCovTarget
envy_mc=zeros(N,res,SIM,TAR);

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
%end of all four loops
end
end
end
end

% %%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE FOR COMP TIME AND ENVY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%x and y labels for each plot
yl=zeros(N,res);
ye=zeros(N,res);

%of no-envy allocation - best figure over all target values
tic;
[envy_min, target_min]=min(envy_mc,[],4);
add=toc;
envy_min=100*envy_min/T;
%target value allowing for minimal envy 
%target_min=target(target_min);

%average computation time including selection of best target for envy
yl=sum(time_mc,3:4)/SIM+add/SIM;
%average envy
ye=sum(envy_min,3)/SIM;

% Create 1-by-2 tiled chart layout
tiledlayout(1,2);
%auxiliary line
onesec=repmat(30,1,res);
onemin=repmat(1,1,res);

% Left plot (comp. time)
ax1 = nexttile;
plot(ax1,xl(1,:),yl(1,:),'r',xl(2,:),yl(2,:),'k',xl(3,:),yl(3,:),'b',xl(3,:),onesec,'k:','MarkerSize',18,'LineWidth',2);
text(ax1,xl(3,res-1),31,'30 secs.', 'FontSize',14);
xlabel(ax1,'Number of goods');
leg1=strcat('n= ',num2str(agents(1)));
leg2=strcat('n= ',num2str(agents(2)));
leg3=strcat('n= ',num2str(agents(3)));
lgd=legend(leg1,leg2,leg3);
legend('Location','northwest');
lgd.FontSize = 12;
title(ax1,'Computation Time (in sec.)');
xlim([xl(1,1),xl(N,res)])
ylim([0 37.5])

% % Right plot (envy)
ax2 = nexttile;
plot(ax2,xl(1,:),ye(1,:),'r',xl(2,:),ye(2,:),'k',xl(3,:),ye(3,:),'b',xl(3,:),onemin,'k:','MarkerSize',18,'LineWidth',2);
text(ax2,xl(1,1)+180,1.12,'1%','FontSize',14);
xlabel(ax2,'Number of goods');
%ylabel(ax1,'Inequality');
leg1=strcat('n= ',num2str(agents(1)));
leg2=strcat('n= ',num2str(agents(2)));
leg3=strcat('n= ',num2str(agents(3)));
lgd=legend(leg1,leg2,leg3);
legend('Location','northeast');
lgd.FontSize = 12;
title(ax2,'Minimal Envy (as % of goods total value)');
xlim([xl(1,1),xl(N,res)])
ylim([0 5])


% Print figure
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], 'PaperUnits', 'Inches', 'PaperSize', [12, 6])
saveas(gcf,'Figure6.pdf')

%Save workspace
save('Figure6.mat')
