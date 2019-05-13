clear all
% load data 
load('count_data.mat');
% use 5 regression methods 
%_________________________________Training_________________________________
% 1. least square
phix=trainx;
[LS_yhat,LS_theta]=LS(trainx,trainy,phix);
% 2. regularized LS
lamd=0.5;
[RLS_yhat,RLS_theta]=RLS(trainx,trainy,lamd,phix);
% 3. LASSO
[LASSO_yhat,LASSO_theta]=LASSO(trainx,trainy,lamd,phix);
% 4. robust regression
[RR_yhat,RR_theta]=RR(trainx,trainy,phix);
% 5. bayesian regression(BR)
gamma=5;
sigma=5;
[mean_theta,cov_theta]=BR(gamma,sigma,trainx,trainy,phix);
BR_mean=phix'*mean_theta;
BR_var=phix'*cov_theta*phix;
% BR_yhat=normrnd(BR_mean,mean(BR_var,2));
BR_yhat = BR_mean;
%_________________________________Testing__________________________________
tst_x=testx;
tst_y=testy;
tst_phix=tst_x;
% 1. LS prediction
LS_pre=tst_phix'*LS_theta;
LS_rms=norm(round(LS_pre)-tst_y)/sqrt(length(tst_y));
figure(1)
subplot(3,2,1),plot(LS_pre +15.0525,'b')
hold on
plot(tst_y+15.0525,'r')
title('test results of LS')
xlabel('x')
ylabel('y')
hold off

% 2. regularized LS prediction
RLS_pre=tst_phix'*RLS_theta;
RLS_rms=norm(round(RLS_pre)-tst_y)/sqrt(length(tst_y));
figure(1)
subplot(3,2,2),plot(round(RLS_pre),'b')
hold on
plot(tst_y,'r')
title('test results of RLS')
xlabel('x')
ylabel('y')
hold off

% 3. LASSO prediction
LASSO_pre=tst_phix'*(LASSO_theta(1:size(trainx,1),:)-LASSO_theta(size(trainx,1)+1:2*size(trainx,1),:));
LASSO_rms=norm(round(LASSO_pre)-tst_y)/sqrt(length(tst_y));
figure(1)
subplot(3,2,3),plot(LASSO_pre +15.0525,'b')
hold on
plot(tst_y+15.0525,'r')
title('test results of LASSO')
xlabel('x')
ylabel('y')
hold off

% 4. RR prediction
RR_pre=tst_phix'*RR_theta;
RR_rms=norm(RR_pre-tst_y)/sqrt(length(tst_y));
figure(1)
subplot(3,2,4),plot(RR_pre+15.0525,'b')
hold on
plot(tst_y++15.0525,'r')
title('test results of RR')
xlabel('x')
ylabel('y')
hold off

% 5. Baysian prediction
BR_mean=tst_phix'*mean_theta;
BR_var=tst_phix'*cov_theta*tst_phix;
BR_pre=normrnd(BR_mean,mean(BR_var,2));
BR_rms=norm(BR_mean-tst_y)/sqrt(length(tst_y));

gprMdl = fitrgp(trainx',trainy,'Basis','linear','KernelFunction','squaredexponential',...
      'FitMethod','exact','PredictMethod','exact');
 ypred = predict(gprMdl,testx');
L = loss(gprMdl,testx',testy);
sqrt(L)

figure(1)
subplot(3,2,5),plot(BR_mean+15.0525,'b')
hold on
plot(tst_y+15.0525,'r')
title('test results of BR')
xlabel('x')
ylabel('y')
hold off

figure(1)
subplot(3,2,6),plot(ypred+15.0525,'b')
hold on
plot(tst_y+15.0525,'r')
title('test results of GP')
xlabel('x')
ylabel('y')
hold off









