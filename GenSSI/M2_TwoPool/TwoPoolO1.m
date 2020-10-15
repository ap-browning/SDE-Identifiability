function model = TwoPoolO1()

    % Symbolic variables
    syms m10 m01
    syms p1 p2 p3 p4

    % Parameters
    model.sym.p = [p1;p2;p3;p4];

    % State variables
    model.sym.x = [m10;m01];

    % Control vectors (g)
    model.sym.g = [0;0];

    % Autonomous dynamics (f)
    model.sym.xdot = [p4 * m01 - (p1 + p3) * m10
                      p3 * m10 - (p2 + p4) * m01];

    % Initial conditions
    model.sym.x0 = [100;0];
        
    % Observables
    model.sym.y = m10;
end

