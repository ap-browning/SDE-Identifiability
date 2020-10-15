
% Symbolic parameters for identifiability analysis
syms p1 p2 p3 p4

% Options
options.reportCompTime = true;

% Structural identifiability analysis (O1)
diary TwoPoolO1_Result.txt
genssiMain('TwoPoolO1',7,[p1;p2;p3;p4],options);
diary off

% Structural identifiability analysis (O2)
diary TwoPoolO2_Result.txt
genssiMain('TwoPoolO2',7,[p1;p2;p3;p4],options);
diary off