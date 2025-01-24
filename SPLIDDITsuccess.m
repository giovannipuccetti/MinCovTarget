%This code compute/plots the % of success of SPLIDDIT
%in finding an optimal allocation in less than a given amount of time,
%for different number of agents/goods.

% The percentage is computed over a number SIM of random simulations of the
% value matrix 
clear;
%fix the random generator seed
rng(1);
%T=Goods total value (value matrix standardized to this value;
%set T=10000 when d>=300 to have more meaningful valuations) 
T=1000;

%Maximum time (in sec.) allowed to SPLIDDIT to find a solution
max_seconds=600;

%%%%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% agents is a vector containing the FIXED numbers n of agents for which to
% draw the plot versus the number d of goods
%agents=[10 20 30];
%if used just to compute one figure
agents=[30];
N=size(agents,2);
% for each number of agents we give a vector with number of goods at which
% to compute values
% xl=[100 150 200 250 300;
%     100 150 200 250 300;
%     100 150 200 250 300;];
%if used just to compute one figure
xl=[300];
res=size(xl,2);
%SIM=number of REPETITIONS f simulation of value matrices
SIM=50;

%array to store exit value of SPLIDDIT
exit_sp=zeros(N,res,SIM);

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


%%%SPLIDDIT CONSTRAINTS MATRIX 
%%%that do not depend on value matrix V but only on the number of
%%%agents/goods

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


%here loop for simulation of matrix value  
for uu=1:SIM 
uu
%%%%%%%%%%%%%%%%%%%%%%VALUE MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V=VALUE MATRIX V consisting of the value v_{i,j} assigned by agent i to object j 
%in SPLIDDIT such matrix must have rowsums=T (agents can distribute T
%points amongst the goos with T=1000)

%alpha= expected ratio of items that each agent gives positive value to in
%case a more realistic generation of value matrices is used
V=valuematrixuniform(n,d,T);

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
options = optimoptions('intlinprog','Display','off','MaxTime',max_seconds);
[x,fval,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,[],options);
%computation of enlapsed time

%store the quality of output of MILP
%most frequent cases are:
%exitpp=1 intlinprog converged to the solution x.
%exitpp=0, -2 intlinprog stopped prematurely (0). No integer feasible point found. 
%exitpp=2 intlinprog stopped prematurely (probably because of time limit). Integer feasible point found.

exit_sp(ii,jj,uu)=double((exitflag==1));
%end of all loops
end
end
end

% %%%%%%%%%%%%%%%%%%%%%%%% OUTPUT FIGURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%x and y labels for each plot
yl=zeros(N,res);
ye=zeros(N,res);

for ii=1:N
    for jj=1:res
    %mean %success across all simulations
    yl(ii,jj)=sum(sum(exit_sp(ii,jj,:)))/SIM;
end 
end

yl(1,:)
% yl(2,:)
% yl(3,:)

% %auxiliary line
% onesec=repmat(1,1,res);
% 
% % Plot of % of success
% ax1 = nexttile;
% plot(ax1,xl(1,:),yl(1,:),'r',xl(2,:),yl(2,:),'k',xl(3,:),yl(3,:),'b',xl(3,:),onesec,'k:','MarkerSize',18,'LineWidth',2);
% xlabel(ax1,'Number of goods');
% leg1=strcat('n= ',num2str(agents(1)));
% leg2=strcat('n= ',num2str(agents(2)));
% leg3=strcat('n= ',num2str(agents(3)));
% lgd=legend(leg1,leg2,leg3);
% legend('Location','northwest');
% lgd.FontSize = 12;
% title(ax1,'% success of SPLIDDIT');
% xlim([xl(1,1),xl(N,res)])
% ylim([0 1])
% 
% % Print figure
% set(gcf, 'Units', 'Inches', 'Position', [0, 0, 6, 6], 'PaperUnits', 'Inches', 'PaperSize', [12, 6])
% saveas(gcf,'Figure6.pdf')
% 
% %save workspace
save('Splidditsuccess')
