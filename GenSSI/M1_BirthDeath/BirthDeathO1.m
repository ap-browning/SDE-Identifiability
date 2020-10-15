function model = BirthDeathO1()

    % Symbolic variables
    syms m1
    syms p1 p2

    % Parameters
    model.sym.p = [p1;p2];

    % State variables
    model.sym.x = m1;

    % Control vectors (g)
    model.sym.g = 0;

    % Autonomous dynamics (f)
    model.sym.xdot = (p1 - p2) * m1;

    % Initial conditions
    model.sym.x0 = 50;
        
    % Observables
    model.sym.y = m1;
end

