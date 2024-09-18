# MinCovTarget

We provide the MatLab codes and workspaces to replicate the numerical analysis in 

Puccetti, G., Rüschendorf, L., and S. Vanduffel (2023)
MinCovTarget: a new standard for fair allocation

[LINK TO THE PAPER IN SSRN](https://ssrn.com/abstract=4456478)

MAIN FILES

- Example5.m  produces the pedagogical example given in Section 5

- Figure 23.m produces the value matrices given in Figures 2 and 3

- FAID.m produces all plots in Figure 4 and Figure 5. The corresponding workspaces are
AU/AD
BU/BD
CU/CD
.mat

- Figure6 (.m and .mat) produces Figure 6

- Figure7 (.m and .mat) produces Figure 7

- SPDcounterexample.m contains an example where Spliddit does not find the maxim Nash welfare allocation

AUXILIARY FILES necessary to run the codes

- valuematrixuniform.m simulates a uniform value matrix
- valuematrixdependent.m simulates a dependent value matrix

For example 5 please download:
- permn.m (see https://it.mathworks.com/matlabcentral/fileexchange/7147-permn)

IMPORTANT: Notice that the function intlinprog, necessary to solve all the SPLIDDIT MILP problems,
needs the MATLAB R2023b Prerelease to be run on a native Apple-Silicon processor.  
