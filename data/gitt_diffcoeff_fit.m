function gp = gitt_diffcoeff_fit(gp)

warning('off','symbolic:sym:sym:DeprecateExpressions');
%% run control
% Note that, because we have not specified otherwise, the default fitness
% function performing multigene regression (regressmulti_fitfun) will
% automatically be used.
gp.runcontrol.pop_size = 500;  % population size of xyz models
gp.runcontrol.num_gen = 150;
gp.runcontrol.verbose = 15;  
gp.runcontrol.runs = 3;        % perform n runs that are merged at the end
gp.runcontrol.timeout = 600;   % each run terminates after xyz seconds
gp.runcontrol.parallel.auto = true; % enable Parallel Computing if installed
gp.runcontrol.showBestInputs = true;
gp.runcontrol.showValBestInputs = true;

%% selection

gp.selection.tournament.size = 20;  % in percentage
gp.selection.tournament.p_pareto = 0.3;
gp.selection.elite_fraction = 0.3; % approx. 1/3 models copied to next gen

% % % genes
% gp.genes.max_genes = 15;
% gp.treedef.max_depth = 2; 

% %maximum depth of sub-trees created by mutation operator
% gp.treedef.max_mutate_depth = 8;
%% Generate training data & inform the g'gp' object about it

filename = './gitt_diff_vs_stoichiometry.csv';
opts = detectImportOptions(filename);
opts.VariableNames = {'soc','D'};
soc_vs_diffcoeff_gitt = readtable(filename,opts);
D_max = max(soc_vs_diffcoeff_gitt.D);

gp.userdata.ytrain = (soc_vs_diffcoeff_gitt.D)/D_max;
gp.userdata.xtrain = soc_vs_diffcoeff_gitt.soc;

gp.userdata.ytest = gp.userdata.ytrain;
gp.userdata.xtest = gp.userdata.xtrain;

% plot(gp.userdata.xtrain,gp.userdata.ytrain,'x');
% hold on;
% % ymodel = gpmodel(gp.userdata.xtrain);
% % plot(gp.userdata.xtrain,ymodel);
% fplot(@(x1) (17064190001150103*x1)/8796093022208 - (4975686852254931*exp(x1))/8796093022208 - (5825119272014035*sin(x1))/4398046511104 + (5329036641513191*x1^2)/35184372088832 + 4927890676361203/8796093022208,[0.3 0.9]);
% % fplot(@(x1) 1944.0*x1 - 565.67*exp(x1) - 1324.5*sin(x1) + 151.46*x1^2 + 560.24,[0.3 0.9]);
% hold off;shg;

%% Function nodes

% gp.nodes.functions.name = {'times','minus','plus','rdivide','square',...
%     'sin','cos','exp','mult3','add3','sqrt','cube','power','negexp',...
%     'neg','abs','log'};

gp.nodes.functions.name = {'times','minus','plus','rdivide','square',...
    'sin','cos','exp','mult3','add3','sqrt','cube','power','negexp',...
    'neg','abs','log','tanh'};

% gp.nodes.functions.name = {'times','minus','plus','rdivide','square',...
%     'sin','cos','mult3','add3','sqrt','cube','power','negexp',...
%     'neg','abs','log'};
