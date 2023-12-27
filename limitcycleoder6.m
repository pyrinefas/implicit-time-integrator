% Test Gauss-Legendre 3-stage with Newton-Raphson for a system of ODEs
tic;
f = @(t, u) [-u(1) - u(2) + u(1) / sqrt(u(1)^2 + u(2)^2); u(1) - u(2) + u(2) / sqrt(u(1)^2 + u(2)^2)];  % Updated System of ODEs
dfdu = @(t, u) [-1/sqrt(u(1)^2 + u(2)^2) + u(1)^2 / (u(1)^2 + u(2)^2)^(3/2), -1/sqrt(u(1)^2 + u(2)^2) + u(1) * u(2) / (u(1)^2 + u(2)^2)^(3/2); 
                1/sqrt(u(1)^2 + u(2)^2) - u(1) * u(2) / (u(1)^2 + u(2)^2)^(3/2), 1/sqrt(u(1)^2 + u(2)^2) - u(2)^2 / (u(1)^2 + u(2)^2)^(3/2)];  % Updated Jacobian matrix
u0 = [sqrt(2)/2; -sqrt(2)/2];  % Updated Initial conditions
N = 1000;  % Number of time steps

u_values = gauss_legendre_newton_system(f, dfdu, u0, N);
disp(u_values);
T_GM6 = toc;

function u = gauss_legendre_newton_system(f, dfdu, u0, N)
    t0 = 0;
    tf = 20;
    h = (tf - t0) / N;

    % Coefficients for the 3-stage Gauss-Legendre method
    c = [0.5 - sqrt(15)/10, 0.5, 0.5 + sqrt(15)/10];
    A = [5/36, 2/9 - sqrt(15)/15, 5/36 - sqrt(15)/30;
         5/36 + sqrt(15)/24, 2/9, 5/36 - sqrt(15)/24;
         5/36 + sqrt(15)/30, 2/9 + sqrt(15)/15, 5/36];
    b = [5/18, 4/9, 5/18];

    t = t0:h:tf;

    u = zeros(2, N+1);
    u(:, 1) = u0;

    for i = 1:N
        % Initial guess for k values using forward Euler
        k_euler = f(t(i), u(:, i)) .* f(t(i), u(:, i));
        k_guess = [k_euler, k_euler, k_euler];

        % Newton-Raphson iterations for k-values
        for j = 1:1000  % We'll iterate a maximum of 10 times
            F = zeros(2, 3);
            J = eye(2);  % Jacobian matrix initialized to identity
            
            for s = 1:3
                Us = u(:, i) + h * sum(A(s,:) .* k_guess, 2);
                F(:, s) = k_guess(:, s) - f(t(i) + c(s)*h, Us);
                
                for r = 1:3
                    J = J - h * A(s, r) * dfdu(t(i) + c(s)*h, Us);
                end
            end
            
            % Update k_guess using the inverse of the Jacobian matrix
            delta_k = J \ F;
            k_guess = k_guess - delta_k;
            
            if norm(delta_k, 'fro') < 1e-7
                break;  % Convergence criteria met
            end
        end
        
        u(:, i+1) = u(:, i) + h * sum(b .* k_guess, 2);
    end
end
