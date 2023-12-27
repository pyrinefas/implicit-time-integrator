
alpha = 1;
L = 2*pi; 
Nx = 200; 
x = linspace(0, L, Nx); 
t0 = 0; 
T = 1; 
Nt = 400; 
t = linspace(t0, T, Nt+1); 
dt = (T-t0)/Nt;
tol = 1e-7; 
t_specific = 1;

u0_x = sin(x); 


u0_k = fft(u0_x);

U = zeros(Nx, Nt+1);
U(:, 1) = u0_k; 


k = [0:Nx/2-1, 0, -Nx/2+1:-1] * (2*pi/L); 


for ti = 2:Nt+1
    current_t = t(ti);
    
    f_xt = cos(x + current_t) + sin(x + current_t); 
    f_k = fft(f_xt);
    
    for ki = 1:Nx
        current_k = k(ki);
        % Function for the heat equation term and the source term
        func = @(t,u) -alpha * current_k^2 * u + f_k(ki);
        func_prim=@(t,u) -alpha * current_k^2;
        
        % Previous step in k-space
        y_k = U(ki, ti-1);
         % Defining the function for the implicit scheme
       g = @(z) z - y_k - dt * func(ti + dt/2, 0.5*(z + y_k));
g_prime = @(z) 1 - 0.5 * dt * func_prim(ti + dt/2, 0.5*(z + y_k));

        % Newton's method for the non-linear equation
        z_new = y_k;
        for iter = 1:1000 
            z_old = z_new;
            z_new = z_old - g(z_old) / g_prime(z_old);
            if abs(z_new - z_old) < tol
                break;
            end
        end
        U(ki, ti) = z_new;
    end
end
% Inverse Fourier transform to get back to x space
u_xt = zeros(Nt+1, Nx);
for ti = 1:Nt+1
    u_xt(ti, :) = ifft(U(:, ti), 'symmetric');
end
% Plot the results
figure;
[X, T_grid] = meshgrid(x, t);
surf(X, T_grid, u_xt, 'EdgeColor', 'none');
axis tight; 
xlabel('x');
ylabel('t');
zlabel('u(x, t)');
title('Temperature evolution in a rod');
view(30, 45);
shg;

% Extracting the solution at t = 1 from the numerical solution matrix
solution_at_T = u_xt(end, :); 

% Analytical solution at t = 1
u_analytical_t1 = sin(x + t_specific);

% Plotting both the analytical and numerical solutions at t = 1
figure;
hold on; 
plot(x, u_analytical_t1, '-', 'DisplayName', 'constructed sol.'); 
plot(x, solution_at_T, '--', 'DisplayName', 'GLM2.'); 
hold off; 
xlabel('x');
ylabel('u(x, 1)');
title('');
legend; 
grid on;








error = abs(u_analytical_t1 - solution_at_T);
% Compute L1 norm
L1_norm = norm(error, 1)/Nx;

% Compute L2 norm
L2_norm = norm(error, 2)/Nx;

% Compute Linf norm
LInf_norm = norm(error, Inf);

% Display the results
fprintf('L1 norm: %f\n', L1_norm);
fprintf('L2 norm: %f\n', L2_norm);
fprintf('LInf norm: %f\n', LInf_norm);