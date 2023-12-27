% Test Gauss-Legendre 3-stage with Newton-Raphson for a system of ODEs
tic;
f = @(t, u) [10*(u(2)-u(1)); u(1)*(28-u(3))-u(2); u(1)*u(2)-3*u(3)];  % Lorenz system
dfdu = @(t, u) [-10, 10, 0; 28-u(3), -1, -u(1); u(2), u(1), -3];  % Jacobian
u0 = [5; 5; 5];  % Initial conditions
N = 100000;  % Number of time steps

u_values = gauss_legendre_newton_system(f, dfdu, u0, N);
u_values
% 3D Plotting
figure;
plot3(u_values(1,:), u_values(2,:), u_values(3,:));
xlabel('x1');
ylabel('x2');
zlabel('x3');
title('3D Plot of the Lorenz System Trajectories');
grid on;

% Save the figure
filename = 'LorenzSystemTrajectories_Gauss_6.png'; 
saveas(gcf, filename);

T_GM6=toc;

% ... [rest of the code before gauss_legendre_newton_system function] ...

function u = gauss_legendre_newton_system(f, dfdu, u0, N)
    t0 = 0;
    tf = 10;
    h = (tf - t0) / N;

    % Coefficients for the 3-stage Gauss-Legendre method
    c = [0.5 - sqrt(15)/10, 0.5, 0.5 + sqrt(15)/10];
    A = [5/36, 2/9 - sqrt(15)/15, 5/36 - sqrt(15)/30;
         5/36 + sqrt(15)/24, 2/9, 5/36 - sqrt(15)/24;
         5/36 + sqrt(15)/30, 2/9 + sqrt(15)/15, 5/36];
    b = [5/18, 4/9, 5/18];

    t = t0:h:tf;

    u = zeros(3, N+1);
    u(:, 1) = u0;

    for i = 1:N
        % Initial guess for k values using forward Euler
        k_euler = f(t(i), u(:, i));
        k_guess = [k_euler, k_euler, k_euler];

        % Newton-Raphson iterations for k-values
        for j = 1:500  % We'll iterate a maximum of 10 times
            F = zeros(3, 3);
            J = eye(3);  % Jacobian matrix initialized to identity inside the loop
            
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
