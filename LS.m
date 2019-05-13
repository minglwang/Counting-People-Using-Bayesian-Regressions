function [yhat,LS_theta]=LS(x,y,phix)
LS_theta=(inv(phix*phix')*phix)*y;
yhat=phix'*LS_theta;
end