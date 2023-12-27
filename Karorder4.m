% Initialization
tic;

% Parameters
alpha = 1; % diffusivity coefficient
L = 2*pi; % length of the domain
Nx = 600; % number of spatial discretization points
x = linspace(0, L, Nx); % spatial domain
T = 1; % Final time
Nt = 5000; % number of time steps
dt = T / Nt; % time step size
tol = 1e-8; % Tolerance for the iterative method
t = linspace(0, T, Nt+1); % time domain
t_specific = 1;

% Initial condition
u0_x = sin(x); % Initial temperature distribution

% Fourier transform of the initial condition
u0_k = fft(u0_x);

% Define k domain for Fourier modes
k_domain = [0:Nx/2-1, 0, -Nx/2+1:-1] * (2*pi/L);
k_series = [u0_k];  % Store Fourier values at each timestep

% Gauss-Legendre method for U(k,t)
for i = 1:Nt
    u_k = gauss_step(k_series(end,:), @fourier_dynamics, dt, tol, 1, 100);
    k_series = [k_series; u_k];
end

% Compute the inverse Fourier transform of U(k,t)
u_values = real(ifft(k_series, [], 2));  % Take real part since we expect the result to be real


% Fourier transformed dynamics
function [kd] = fourier_dynamics(k_value)
    % Return a time derivative based on the Fourier-transformed equation
    global alpha k_domain;
   kd = -alpha .* (k_domain.^2) .* k_value;
end

function k_next = gauss_step(k, dynamics, dt, threshold, damping, max_iterations)
    global Nx;
    sq3 = sqrt(3);
    if damping > 1 || damping <= 0
        error('damping should be between 0 and 1.');
    end

    % Use explicit Euler steps as initial guesses
    k_dot = dynamics(k);
    k1_guess = k + (1/2 - sq3/6) * dt * k_dot;
    k2_guess = k + (1/2 + sq3/6) * dt * k_dot;

    a11 = 1/4;
    a12 = 1/4 - sq3/6;
    a21 = 1/4 + sq3/6;
    a22 = 1/4;

    error_func = @(k1, k2) k1 - dynamics(k + a11 * dt * k1 + a12 * dt * k2) - k2 + dynamics(k + a21 * dt * k1 + a22 * dt * k2);
    err = error_func(k1_guess, k2_guess);
    iteration = 1;
    
    while (norm(err) > threshold && iteration < max_iterations)
        iteration = iteration + 1;
        
        k_next = [k1_guess; k2_guess] - damping * err;
        k1_guess = k_next(1:Nx);
        k2_guess = k_next(Nx+1:end);

        err = error_func(k1_guess, k2_guess);
    end
    
    if norm(err) > threshold
        error('Newton did not converge by %d iterations.', max_iterations);
    end
    
    k_next = k + dt / 2 * (k1_guess + k2_guess);
end
