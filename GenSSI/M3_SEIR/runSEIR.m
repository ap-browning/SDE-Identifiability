
% Symbolic parameters for identifiability analysis
syms p1 p2 p3 mu i0

% Options
options.reportCompTime = true;

% Structural identifiability analysis (ODE)
diary SEIR_ODE_Result.txt
genssiMain('SEIR_ODE',7,[p1;p2;p3;mu;i0],options);
diary off

% Structural identifiability analysis (SDE Mean-field closure)
diary SEIR_MeanField_Result.txt
genssiMain('SEIR_MeanField',7,[p1;p2;p3;mu;i0],options);
diary off

% Structural identifiability analysis (SDE Pairwise closure)
diary SEIR_Pairwise_Result.txt
genssiMain('SEIR_Pairwise',7,[p1;p2;p3;mu;i0],options);
diary off

% Structural identifiability analysis (SDE Gaussian closure)
diary SEIR_Gaussian_Result.txt
genssiMain('SEIR_Gaussian',7,[p1;p2;p3;mu;i0],options);
diary off