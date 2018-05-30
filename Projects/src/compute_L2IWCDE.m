clear all

rand('state',0);
randn('state',0);

ntrain = 300;
xtrain = rand(1,ntrain)*2-1;
noise  = randn(1,ntrain);
ytrain = sinc(2*pi*xtrain) + exp(1-xtrain).*noise / 8;

xtest0     = [-0.5 0. 0.5];
ytest0     = linspace(-3,3,300);
axis_limit = [-1 1 -3 3];
% 
xtest1 = repmat(xtest0,length(ytest0),1);
ytest1 = repmat(ytest0',1,length(xtest0));
xtest  = xtest1(:)';
ytest  = ytest1(:)';
ntest  = length(xtest);


%normalization
xscale = std(xtrain,0);
yscale = std(ytrain,0);
xmean  = mean(xtrain);
ymean  = mean(ytrain);

xtrain_normalized = (xtrain-xmean)./repmat(xscale,[1 ntrain]);
ytrain_normalized = (ytrain-ymean)./repmat(yscale,[1 ntrain]);
xtest_normalized  = (xtest-xmean)./repmat(xscale,[1 ntest]);
ytest_normalized  = (ytest-ymean)./repmat(yscale,[1 ntest]);

%True conditional density for artificial data
ptest=pdf_Gaussian_CDF(ytest,sinc(2*pi*xtest),exp(1-xtest)/8);

%%%%%%%%%%%%%% Importance Weight Calculation %%%%%%%%%%%%
xtr = xtrain_normalized;
ytr = ytrain_normalized;
yte = ytest_normalized;
xte = xtest_normalized;
ntr = ntrain;
nte = ntest;
x = cat(2, xtr, xte);
n = ntr+nte;
U = zeros(n);
u_hat = zeros(ntr,1);
V_hat = zeros(n,ntr);
alpha = zeros(ntr,1);
K_x=0.5;
K_y=0.5;
K_sq_x=K_x*K_x;
K_sq_y=K_y*K_y;
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
 
ro = 0.1;
I_n = ones(ntr,ntr);
Q = V_hat_t * (U_delta_I\V_hat) + ro*I_n;
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

%%%%%%%%%%%%%%%%%%%%%%%%% Estimating conditional density %%%%%%%%%%%%%%%%%%%%%%%%%

ph_uw=LSCDE(xtrain_normalized,ytrain_normalized,...
         [],xtest_normalized,ytest_normalized,'SQ');

ph=LSCDE(xtrain_normalized,ytrain_normalized,...
         w,xtest_normalized,ytest_normalized,'SQ');

figure(1)
clf
hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(xtrain,ytrain,'ko','LineWidth',1,'MarkerSize',6)

xtest_unique = unique(xtest);
for xtest_index = 1:length(xtest_unique)
  x=xtest_unique(xtest_index);

  cdf_scale = (xtest0(2)-xtest0(1))*0.8/max(max(ptest(xtest==x)),max(ph(xtest==x)/yscale));
  plot(xtest(xtest==x)+ptest(xtest==x)*cdf_scale,...
     ytest(xtest==x),'b--','LineWidth',2);
  plot(xtest(xtest==x)+ph(xtest==x)*cdf_scale/yscale,...
     ytest(xtest==x),'r-','LineWidth',2);  
 
  cdf_scale_uw=(xtest0(2)-xtest0(1))*0.8/max(max(ptest(xtest==x)),max(ph_uw(xtest==x)/yscale));
  plot(xtest(xtest==x)+ph_uw(xtest==x)*cdf_scale_uw/yscale,...
     ytest(xtest==x),'k-','LineWidth',2); 
end


ntest_dist = 300;
xtest_dist = rand(1,ntest_dist)*2-1;
mu_n1 = -1;
sigma_n1 = 4/9;
mu_n2 = 1;
sigma_n2 = 4/9;
noise_dist = cat(2, mu_n1+sigma_n1*rand(1,ntest_dist/2), mu_n2+sigma_n2*rand(1,ntest_dist/2));

ytest_dist = sinc(2*pi*xtest_dist)+exp(1-xtest_dist).*noise_dist/8;

plot(xtest_dist,ytest_dist,'gx','LineWidth',1,'MarkerSize',6);
legend('Train','True','L2IWCDE-Estimated','UW-Estimated');
title('Conditional Density Estimation');
axis(axis_limit);
xlabel('x');
ylabel('y');
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);
print('-dpng',sprintf('conditional-density'));

