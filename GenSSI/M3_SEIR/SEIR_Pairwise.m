function model = SEIR_Pairwise()

    % Symbolic variables
    syms m1000 m0100 m0010 m0001 m2000 m0200 m0020 m0002 m1100 m1010 m1001 m0110 m0101 m0011
    syms p1 p2 p3 mu i0

    % Parameters
    model.sym.p = [p1;p2;p3;mu;i0];

    % State variables
    model.sym.x = [m1000;m0100;m0010;m0001;m2000;m0200;m0020;m0002;m1100;m1010;m1001;m0110;m0101;m0011];

    % Control vectors (g)
    model.sym.g = zeros(size(model.sym.x));

    % Autonomous dynamics (f)
    model.sym.xdot = [
                        -m1010*p1                        
                        -m0100*p2 + m1010*p1                        
                        -m0010*p3 + m0100*p2                        
                         m0010*p3                        
                        (m1000*m1010*p1 - 2*m1010*m2000*p1)/m1000                        
                        (m0100^2*p2 + 2*m0110*m1100*p1 - (2*m0200*p2 - m1010*p1)*m0100)/m0100                        
                        (m0100 + 2*m0110)*p2 - 2*m0020*p3 + m0010*p3                        
                        m0010*p3 + 2*m0011*p3                        
                        (-((m1010*p1 + m1100*p2)*m1000 - m1010*m2000*p1)*m0100 - m0110*m1000*m1100*p1)/(m0100*m1000)                        
                        (-(m1010*p3 - m1100*p2)*m0010 - m0020*m1010*p1)/m0010                        
                        (m0010*m1010*p3 - m0011*m1010*p1) / m0010                        
                        (-((p2 + p3)*m0110 - m0200*p2 + m0100*p2)*m0010 + m0020*m1010*p1)/m0010                        
                        (-(m0101*p2 - m0110*p3)*m0010 + m0011*m1010*p1)/m0010                        
                        m0020*p3 + m0101*p2 - m0011*p3 - m0010*p3                        
                     ]; 

    % Initial conditions
    model.sym.x0 = [500 - i0; i0; 10 / mu; 10 / mu; 0; 0; 0; 0; (500 - i0)*i0; (500 - i0) * 10 / mu; (500 - i0) * 10 / mu; i0 * 10 / mu; i0 * 10 / mu; 100/mu^2    ];
        
    % Observables
    model.sym.y = [mu*m0010;mu*m0001;mu^2*m0020;mu^2*m0002;mu^2*m0011];
    
end

