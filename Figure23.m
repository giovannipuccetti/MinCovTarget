% This MATLAB code produces the two value matrices illustrated in Figures 2
% and 3 by using two different designs.

clear;

%n=number of agents
n=4;
%d=number of goods
d=10;
%goods value
T=1000;
%fix the random generator seed for reproducibility
rng(1);

%%%%%%%%%%%%%%%%%%%%%%VALUE MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V=VALUE MATRIX V consisting of the value v_{i,j} assigned by agent i to object j 
%in SPLIDDIT such matrix must have rowsums=T (agents can distribute T
%points amongst the goos with T=1000)

%uniform valued matrix
V=valuematrixuniform(n,d,T)

%more realistic matrix with dependent valuations
%correlation parameter
rho=0.5;
V=valuematrixdependent(n,d,T,rho)
