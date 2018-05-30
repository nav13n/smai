clear all


rand('state',2);
randn('state',2);

ntr = 300;
nte = 300;

% Generate samples

mu_ytr1    = 1;       % Mean
sigma_ytr1 = 1.5;     % Sigma
ntr1       = 0.4 * ntr; % Number of points

mu_ytr2    = 2.5;
sigma_ytr2 = 0.5;
ntr2       = 0.6 * ntr;

% Generate the data.
ytr1 = (randn(1,ntr1) * sigma_ytr1) + mu_ytr1;
ytr2 = (randn(1,ntr2) * sigma_ytr2) + mu_ytr2;

ytr = cat(2,ytr1, ytr2);

% Draw nte samples for yte from gaussian model 
% pte(y) = N (y; 2.5, 0.5)
mu_yte    = 2.5;
sigma_yte = 0.5;
yte       = mu_yte + sigma_yte * randn(1,ntr);

% Create xtr according to the relation
% xi = yi + 3 + εi, where the noise {εi}n is independently drawn following N (ε; 0, 1.52)
mu_noise    = 0;
sigma_noise = 1.52;
noise       = mu_noise + sigma_noise * randn(1,ntr);

xtr = ytr + 3 + noise;
xte = yte + 3 + noise;

ydisp = linspace(-2,5,500);
xdisp = linspace(1,10,100);
disp(size(ydisp))
ptr_y1disp = pdf_Gaussian(ydisp, mu_ytr1, sigma_ytr1);
ptr_y2disp = pdf_Gaussian(ydisp, mu_ytr2, sigma_ytr2);
pte_ydisp  = pdf_Gaussian(ydisp, mu_yte, sigma_yte);
w_ytrdisp  = pte_ydisp./(ptr_y1disp + ptr_y2disp);

disp(size(ptr_y1disp))

x = cat(2, xtr, xte);
n = ntr + nte;
U = zeros(n);
u_hat = zeros(ntr, 1);
V_hat = zeros(n, ntr);
alpha = zeros(ntr, 1);
K_x = 0.5;
K_y = 0.5;
K_sq_x = K_x*K_x;
K_sq_y = K_y*K_y;
dim = 1;

for i = 1:n
    for j = 1:n
        U(i,j) = ((pi*K_sq_x)^(dim/2))*exp(-((x(i)-x(j))^2)/4*(K_sq_x));
    end
end

for l=1:n
    sum = 0;
    for j = 1:nte
        u = exp(-((xte(j)-x(l))^2)/(2*K_sq_x));
        sum = sum + u;
    end
    sum = sum/nte;
    u_hat(l,1) = sum;
end

for f=1:n
    for g=1:ntr
         x_f = x(f);
         y_g = ytr(g);
         sum = 0; 
         for i = 1:ntr
             v_f = exp(-((xtr(i)-x_f)^2)/(2*K_sq_x));
             v_g = exp(-((ytr(i)-y_g)^2)/(2*K_sq_y));
             sum = sum + (v_f*v_g);
         end
         V_hat(f,g) = sum/ntr;
    end
end

alpha_t = transpose(alpha);
u_hat_t = transpose(u_hat);
V_hat_t = transpose(V_hat);
delta = 0.1;
U_delta_I = inv(U+delta*eye(n));
J_alpha = 0.5 * ( alpha_t * V_hat_t * (U_delta_I\V_hat) * alpha - ...
                  u_hat_t * (U_delta_I\V_hat) * alpha );

              
ro  = 0.1;
I_n = ones(ntr,ntr);
Q   = V_hat_t * (U_delta_I\V_hat) + ro*I_n;
f_t = - u_hat_t * (U_delta_I\V_hat);

phi_y = zeros(ntr,1);
for i = 1:ntr
    sum = 0;
    for j = 1:ntr
        sum = sum + exp(-(ytr(i) - ytr(j))^2 / (2*K_sq_y));
    end
    phi_y(i) = sum / ntr; 
end


H = Q;
f = f_t;
A = [];
b = [];
lb = zeros(ntr,1);
Aeq = transpose(phi_y);
beq = 1;

[alpha,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,[]);

phi_temp = zeros(ntr,ntr);
for i = 1:ntr
    for j = 1:ntr
        phi_temp(i,j)  = exp(-(ytr(j) - ytr(i))^2 / (2*K_sq_y));
    end    
end
   
w = transpose(alpha)*phi_temp;

[IWLS_model0]=L2IWE_train(xtr,ytr,[],xte,[],[],0);
disp(sprintf('IWLS: sigma = %g, lambda = %g, gamma = %g'...
             ,IWLS_model0.sigma,IWLS_model0.lambda,IWLS_model0.gamma))
[ydisph0]=L2IWE_test(xdisp,IWLS_model0);

[IWLS_model]=L2IWE_train(xtr,ytr,w,xte);
disp(sprintf('IWLS: sigma = %g, lambda = %g, gamma = %g'...
             ,IWLS_model.sigma,IWLS_model.lambda,IWLS_model.gamma))
[ydisph]=L2IWE_test(xdisp,IWLS_model);

figure(1);clf;hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(ydisp,ptr_y1disp,'co','LineWidth',2, 'MarkerSize',2)
plot(ydisp,ptr_y2disp,'go','LineWidth',2, 'MarkerSize',4)
plot(ydisp,pte_ydisp,'r-','LineWidth',2)
plot(ydisp,w_ytrdisp,'k-','LineWidth',2, 'MarkerSize',1)
plot(ytr,w,'bo','LineWidth',2)
legend('p_{tr}(y)','p_{tr}(y)','p_{te}(y)','w(ytr)', 'L2IWE-w(ytr)')
xlabel('y')
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);
print('-dpng','IW_estimation')

%%%%%%%%%%%%%%%%%%%%%%%%% Estimating Regression %%%%%%%%%%%%%%%%%%%%%%%%%

figure(2);clf;hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(xdisp,ydisph,'r-','LineWidth',2)
plot(xdisp,ydisph0,'k-','LineWidth',2)
plot(xtr,ytr,'bo','LineWidth',1)
plot(xte,yte,'gx','LineWidth',1)
legend('L2IWE-f(x)','UW-f(x)','Training','Test')
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);
print('-dpng','regression')

