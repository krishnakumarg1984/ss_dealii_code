clear; close; clc;

b = 5;
x = linspace(0,b,100);
% y = exp(-((x-0.5*b).^2)/(0.125*b));
y = exp(-((x-0.3*b).^2)/(0.5*b));
plot(x,y);
shg;