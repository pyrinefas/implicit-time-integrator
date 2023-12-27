% Parameters
T = 20;
x0 = [sqrt(2)/2; -sqrt(2)/2];  % Initial conditions

% Evaluate x1 and x2 at T = 20
x1_at_T = (1 - (1 - sqrt(x0(1)^2 + x0(2)^2)) * exp(-T)) * cos(T + atan(x0(2) / x0(1)));
x2_at_T = (1 - (1 - sqrt(x0(1)^2 + x0(2)^2)) * exp(-T)) * sin(T + atan(x0(2) / x0(1)));

% Display the results
disp('x_1 at T=20:');
disp(x1_at_T);
disp('x_2 at T=20:');
disp(x2_at_T);
