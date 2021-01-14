function [ Xnxt, w_est ] = halfspace_query_synthesis( Xold, Yold, k )
%Active Learning of Halfspaces via Query Synthesis (Paper published at AAAI
%   2015), direct any questions to ibrahim.alabdulmohsin@kaust.edu.sa
%
%   DESCRIPTION:
%       This routine synthesizes near-optimal k queries to learn a halfspace 
%       given the old membership queries. 
%   INPUT:  
%       k is the batch size (i.e. the number of new queries).
%       Xold are the instances of the old queries. Each row of Xold corresponds to one instance.
%           >>>>> You MUST include at least one positive and one negative
%           instance <<<<<
%       Yold are the binary labels {-1, +1} that correspond to Xold
%   OUTPUT:
%       Xnxt is a (k x d) matrix of next queries
%       w_est is the estimated coefficient vector for the halfspace (see
%       the full paper for details)
%   EXAMPLE:
%       If we know that x1=(1,1) is in the halfspace, x2=(-1,-1) is outside the halfspace, 
%       and x3=(1,2) is in the halfspace, and we want to select two queries next then:
%           Xold=  [1,1; 
%                   -1,-1; 
%                   1,2]
%           Yold=  [1;-1;1]
%           k = 2
%       To show the exponential decay, suppose w is fixed, and X contains one positive and one negative instance. 
%       Then, we can estimate w using the following loop: 
%             ERR=[]; k=5; 
%             Y=sign(X*w);
%             for i=1:100,
%               [Xnxt, w_est ] = select_next_queries(X,Y,k);
%               X=[X; Xnxt];
%               Y=sign(X*w);
%               err = norm(w-w_est)
%               ERR = [ERR err];
%             end
%             plot(ERR)
%   DEPENDENCIES: You need to install CVX (http://cvxr.com/cvx/download/)
    
d=size(Xold,2); 
m=size(Xold,1);

%solve the maximum ellipsoid problem
%it is assumed that at least one positive and one negative instance exist
cvx_begin quiet
    try
        cvx_solver mosek
    catch 
        cvx_solver SDPT3
    end
    variable s(d)
    variable u(d)

    maximize (geo_mean(s))
    subject to
        diag(Yold)*(Xold*u)>= norms(Xold.*(ones(m,1)*s'),2,2);
        norm(u)<=1
cvx_end

disp(u);
disp(s);

w_est = u;  
S=diag(s);  %the square root of the covariance matrix 
N=null(u');
B = S*N; 
B=B'*B;
[alph,~] = eigs(B,k);
Xnxt = (N*alph)';

end

