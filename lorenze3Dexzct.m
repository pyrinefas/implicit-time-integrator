% Main Script for solving the system of ODEs using Verner's 9th Order Runge-Kutta method

% Main execution
T = 10;
x0 = [5; 5; 5];  % Updated initial condition
t = linspace(0, T, 200000);  % Time span

solution = RungeKutta9(@system_of_odes, x0, t);


% 3D Plot of the results
figure;
plot3(solution(1,:), solution(2,:), solution(3,:));
xlabel('x_1(t)');
ylabel('x_2(t)');
zlabel('x_3(t)');
title('3D Plot of the Lorenz System Trajectories');
grid on;
% Save the figure
filename = 'LorenzSystemTrajectories_exat.png'; % You can also use .jpg, .tif, .pdf, etc.
saveas(gcf, filename);

disp('Value of x_1 at T=20:');
disp(solution(1,end));
disp('Value of x_2 at T=20:');
disp(solution(2,end));
disp('Value of x_3 at T=20:');
disp(solution(3,end));

% Verner's 9th Order Runge-Kutta method function definition
function x = RungeKutta9(f, x0, t)
    h = t(2) - t(1);
    nt = numel(t);
    nx = numel(x0);

    x = zeros(nx, nt);
    x(:, 1) = x0;
    for s = 1:(nt - 1)
        ts = t(s);
        xs = x(:,s);
        s6 = sqrt(6);
        
        k1 = f(ts, xs);
        k2 = f(ts + h/12, xs + h*k1/12);
       k3 = f(ts + h/9, xs + h/27*(k1 + 2*k2));
        k4 = f(ts + h/6, xs + h/24*(k1 + 3*k3));
        k5 = f(ts + (2+2*s6)*h/15, xs + h/375*((4+94*s6)*k1 - (282+252*s6)*k3 + (328+208*s6)*k4));
        k6 = f(ts + (6+s6)*h/15, xs + h*((9-s6)*k1/150 + (312+32*s6)*k4/1425 + (69+29*s6)*k5/570));
        k7 = f(ts + (6-s6)*h/15, xs + h*((927-347*s6)*k1/1250 + (-16248+7328*s6)*k4/9375 + (-489+179*s6)*k5/3750 + (14268-5798*s6)*k6/9375));
        k8 = f(ts + 2*h/3, xs + h/54*(4*k1 + (16-s6)*k6 + (16+s6)*k7));
        k9 = f(ts + h/2, xs + h/512*(38*k1 + (118-23*s6)*k6 + (118+23*s6)*k7 - 18*k8));
        k10 = f(ts + h/3, xs + h*(11*k1/144 + (266-s6)*k6/864 + (266+s6)*k7/864 - k8/16 - 8*k9/27 ));
        k11 = f(ts + h/4, xs + h*((5034-271*s6)*k1/61440 + (7859-1626*s6)*k7/10240 + (-2232+813*s6)*k8/20480 + (-594+271*s6)*k9/960 + (657-813*s6)*k10/5120));
        k12 = f(ts + 4*h/3, xs + h*((5996-3794*s6)*k1/405 + (-4342-338*s6)*k6/9 + (154922-40458*s6)*k7/135 + (-4176+3794*s6)*k8/45 + (-340864+242816*s6)*k9/405 + (26304-15176*s6)*k10/45 - 26624*k11/81));
        k13 = f(ts + 5*h/6, xs + h*((3793 + 2168*s6)*k1/103680 + (4042+2263*s6)*k6/13824 + (-231278+40717*s6)*k7/69120 + (7947 - 2168*s6)*k8/11520 + (1048-542*s6)*k9/405 + (-1383+542*s6)*k10/720 + 2624*k11/1053 + 3*k12/1664));
        k14 = f(ts + h, xs + h*(-137*k1/1296 + (5642-337*s6)*k6/864 + (5642+337*s6)*k7/864 - 299*k8/48 + 184*k9/81 - 44*k10/9 - 5120*k11/1053 - 11*k12/468 + 16*k13/9));

        dx = h * (103/1680*k1 - 27/140*k8 + 76/105*k9 - 201/280*k10 + 1024/1365*k11 + 3/7280*k12 + 12/35*k13 + 9/280*k14);
        x(:, s+1) = xs + dx;
    end
end

% Updated System of ODEs function definition
function dx = system_of_odes(t, x)
    dx = zeros(3, 1);
    dx(1) = 10*(x(2) - x(1));
    dx(2) = x(1)*(28 - x(3)) - x(2);
    dx(3) = x(1)*x(2) - 3*x(3);
end
