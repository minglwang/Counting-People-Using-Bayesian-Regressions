function [mean_theta,cov_theta]=BR(gamma,sigma,x,y,phix)
cov_theta=inv(1/gamma+1/sigma*phix*phix');
mean_theta=1/sigma*cov_theta*phix*y;
end