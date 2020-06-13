clear; close; clc;

b = 5;
x = linspace(0,b,100);
y = (4*x/b).*(1-(x/b));
plot(x,y);
