clear; close all;clc;

%% user input (note that i = i_pulse)

i = -6.1538;  % <--- pls check if this is indeed negative for discharge

%% constants
i0_neg = 12;  % <--- please check this

F = 96487; 
R = 8.314;
T = 298.15;

%% Calculation

phi_el_negcc = sprintf('At neg cc., phi_el = %g',(2*R*T/F)*asinh(i/(2*i0_neg)))