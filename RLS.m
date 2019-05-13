function [RLS_yhat,RLS_theta]=RLS(x,y,lamd,phix)
RLS_theta=(inv(phix*phix'+lamd * eye(size(phix,1)))*phix)*y;
RLS_yhat=phix'*RLS_theta;
end