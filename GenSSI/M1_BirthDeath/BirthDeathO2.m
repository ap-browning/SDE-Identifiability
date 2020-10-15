function model = BirthDeathO2()

    % Symbolic variables
    syms m1 m2
    syms p1 p2

    % Parameters
    model.sym.p = [p1;p2];

    % State variables
    model.sym.x = [m1;m2];

    % Control vectors (g)
    model.sym.g = [0;0];

    % Autonomous dynamics (f)
    model.sym.xdot = [    (p1 - p2) * m1
                      2 * (p1 - p2) * m2 + (p1 + p2) * m1
                     ];

    % Initial conditions
    model.sym.x0 = [50;0];
        
    % Observables
    model.sym.y = [m1;m2];
end

