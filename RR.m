function [RR_yhat,RR_theta]=RR(x,y,phix)
n=size(phix,2);
d=size(phix,1);
f=[zeros(1,d),ones(1,n)];
A=[-phix',-ones(n);phix',-ones(n)];
b=[-(y);y];
tmp=linprog(f,A,b);
RR_theta=tmp(1:d,:);
RR_yhat=phix'*RR_theta;
end