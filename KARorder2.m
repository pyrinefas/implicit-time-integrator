% Define the system of ODEs
tic;
f = @(t, y) [y(2)*y(3); y(1)*y(3); -2*y(1)*y(2)]; % Updated ODEs
dfdy = @(t, y) [0, y(3), y(2); y(3), 0, y(1); -2*y(2), -2*y(1), 0]; % Updated Jacobian matrix

N = 1000;
t0 = 0;
T = 20;
y0 = [1; 2; -3]; % Updated initial conditions
h = (T - t0) / N;
tol = 1e-8;

y = imp_midpt(f, dfdy, t0, T, y0, h, tol, N);

% Plotting
t = t0:h:T;
plot(t, y(1, :), 'r'); % x1 vs. t
hold on;
plot(t, y(2, :), 'b'); % x2 vs. t
plot(t, y(3, :), 'g'); % x3 vs. t
legend('x1', 'x2', 'x3');
T_GM2 = toc;

function y = imp_midpt(f, dfdy, t0, T, y0, h, tol, N)
    t = t0:h:T;
    n = length(t);
    y = zeros(3, n);
    y(:, 1) = y0;
    
    for k = 1:n-1
        g = @(z) z - y(:, k) - h*f(t(k) + 0.5*h, 0.5*(z + y(:, k)));
        gp = @(z) eye(3) - h*dfdy(t(k) + 0.5*h, 0.5*(z + y(:, k)));
        y(:, k+1) = newton(g, gp, y(:, k), tol, N);
    end
end

function sol = newton(f, fp, x0, tol, N)
    for i = 1:N
        delta = fp(x0) \ f(x0);
        x0 = x0 - delta;
        
        if norm(delta) < tol
            break;
        end
    end
    sol = x0;
end
