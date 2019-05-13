function [LASSO_yhat,LASSO_theta]=LASSO(x,y,lamd,phix)
H=[phix*(phix)', -(phix*phix'); -(phix*phix'), phix*phix'];
f=lamd-[phix*y;-(phix*y)];
A=-1*eye(length(f));
b=zeros(length(f),1);
LASSO_theta=quadprog(H,f,A,b);
LASSO_yhat=phix'*(LASSO_theta(1:length(f)/2,:)-LASSO_theta(length(f)/2+1:length(f),:));
end