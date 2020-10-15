function model = SEIR_ODE()

    % Symbolic variables
    syms S E I R
    syms p1 p2 p3 mu e0

    % Parameters
    model.sym.p = [p1;p2;p3;mu;e0];

    % State variables
    model.sym.x = [S;E;I;R];

    % Control vectors (g)
    model.sym.g = zeros(size(model.sym.x));

    % Autonomous dynamics (f)
    model.sym.xdot = [
                       -p1 * S * I
                        p1 * S * I - p2 * E
                        p2 * E - p3 * I
                        p3 * I
                     ];

    % Initial conditions
    model.sym.x0 = [500 - e0; e0; 10 / mu; 10 / mu];

    % Observables
    model.sym.y = [mu*I;mu*R];

end
