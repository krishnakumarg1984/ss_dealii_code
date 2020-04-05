clear; close all; clc;
x = linspace(0,5,64);
n_time_steps = 20;
t_end = 10;
t_vector = linspace(0,t_end,n_time_steps);
A = 10;
omega = pi/10;
u_mms = zeros(length(x),length(t_vector));

for time_idx = 1:length(t_vector)
    u_mms(:,time_idx) = A*sin(omega*t_vector(time_idx)).*(x.^3 - 10.0*x.^2 + 25.0*x);
end

%% 
t_plot_sec = linspace(0,t_end,10); % time-snapshots to plot in seconds
% t_plot_sec = 5; % time-snapshots to plot in seconds

plot_time_indices_of_t_vector = knnsearch(t_vector',t_plot_sec');
clf;clc;hold on;
for plot_no = 1:length(plot_time_indices_of_t_vector)
    t_vector_index_to_plot = plot_time_indices_of_t_vector(plot_no);
    plot(x,u_mms(:,t_vector_index_to_plot));
    legendstr{plot_no} = num2str(t_vector(t_vector_index_to_plot));
end
hold off;
legend(legendstr);
box on;
grid on;
shg;