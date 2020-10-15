function model = TwoPoolO2()

    % Symbolic variables
    syms m10 m01 m20 m02 m11
    syms p1 p2 p3 p4

    % Parameters
    model.sym.p = [p1;p2;p3;p4];

    % State variables
    model.sym.x = [m10;m01;m20;m02;m11];

    % Control vectors (g)
    model.sym.g = [0;0;0;0;0];

    % Autonomous dynamics (f)
    model.sym.xdot = [p4 * m01 - (p1 + p3) * m10
                      p3 * m10 - (p2 + p4) * m01
                      p4 * (m01 + 2*m11) + (p1 + p3)*(m10 - 2*m20)
                      p3 * (m10 + 2*m11) + (p2 + p4)*(m01 - 2*m02)
                      -(p1 + p2) * m11 - p3 * (m10 + m11 - m20) - p4 * (m10 - m02 + m11)];

    % Initial conditions
    model.sym.x0 = [100;0;0;0;0];
        
    % Observables
    model.sym.y = [m10;m20];
end

