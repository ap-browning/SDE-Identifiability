
% Symbolic parameters for identifiability analysis
syms p1 p2

% Options
options.reportCompTime = true;

% Structural identifiability analysis (O1)
diary BirthDeathO1_Result.txt
genssiMain('BirthDeathO1',7,[p1;p2],options);
diary off

% Structural identifiability analysis (O2)
diary BirthDeathO2_Result.txt
genssiMain('BirthDeathO2',7,[p1;p2],options);
diary off
