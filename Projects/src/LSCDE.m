function [ph,ph_error]=LSCDE(x_train,y_train,w, x_test,y_test,loss_type)
%
% Least-Squares Conditional Density Estimation
%
% Estimating conditional density p(y|x) from samples {(x_i,y_i)}_{i=1}^n
%
% Usage:
%       [ph,ph_error]=LSCDE(x_train,y_train,x_test,y_test,loss_type)
%
% Input:
%    x_train:      d_x by n training sample matrix
%    y_train:      d_y by n training sample matrix
%    x_test:       d_x by n_test sample matrix
%    y_test:       d_y by n_test sample matrix
%    loss_type:    (OPTIONAL) 'SQ':squared-loss, 'KL':KL-loss (default: 'KL')
%
% Output:
%    ph:          estimates of conditional density at (xtest,ytest)
%    ph_error:       estimation error

% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/LSCDE/

if nargin<6 || isempty(loss_type)
  loss_type='KL';
end


if nargin<3 || isempty(w)
    [d_y,n]=size(y_train);
    w=ones(1, n);
elseif sum(w<0)>0
    error('Importance weights must be non-negative')
end

sigma_list=logspace(-1.5,1.5,9); % Candidates of Gaussian width
lambda_list=logspace(-3,1,9); % Candidates of regularization parameter
cc=eps;
[d_x,n]=size(x_train);
[d_y,n]=size(y_train);
fold=5; % Number of folds of cross-validation
b=min(100,n); % Number of kernel bases

  %%%%%%%%%%%%%%%% Choose Gaussian kernel centers w=(u,v) for z=(x,y)
  rand_index=randperm(n);
  u=x_train(:,rand_index(1:b)); 
  v=y_train(:,rand_index(1:b));
  w_train=w(:,rand_index(1:b));

  cv_fold=[1:fold];
  cv_index=randperm(n);
  cv_split=floor([0:n-1]*fold./n)+1;
  
  v2=sum(v.^2,1);
  xu_dist2=repmat(sum(x_train.^2,1),[b 1])+repmat(sum(u.^2,1)',[1 n])-2*u'*x_train;
  yv_dist2=repmat(sum(y_train.^2,1),[b 1])+repmat(sum(v2,1)',[1 n])-2*v'*y_train;
  vv_dist2=repmat(v2,[b 1])+repmat(v2',[1 b])-2*v'*v;
  score_cv=zeros(length(sigma_list),length(lambda_list));
  w_i=repmat(w_train,[b 1]);
  
  for sigma_index=1:length(sigma_list)
    sigma=sigma_list(sigma_index);
    
    phi_xu=exp(-xu_dist2/(2*sigma^2));
    phi_yv=exp(-yv_dist2/(2*sigma^2));
    phi_zw=phi_xu.*phi_yv;
    phi_vv=exp(-vv_dist2/(4*sigma^2));
    for k=1:fold
      tmp=phi_xu(:,cv_index(cv_split==k));
      disp(size(w));
      disp(size(w_i));
      disp(size(phi_vv.*(tmp*tmp')));
      Phibar_cv(:,:,k)=(sqrt(pi)*sigma)^d_y*phi_vv.*(tmp*tmp');
    end % for fold
    
    for lambda_index=1:length(lambda_list)
      lambda=lambda_list(lambda_index);
      
      score_tmp=zeros(1,fold);
      for k=1:fold
        alphat_cv=mylinsolve(sum(Phibar_cv(:,:,cv_fold~=k),3)/sum(cv_split~=k)+lambda*eye(b), ...
                             mean(phi_zw(:,cv_index(cv_split~=k)),2));
        alphah_cv=max(0,alphat_cv);
        normalization_cv=max(cc,(sqrt(2*pi)*sigma)^d_y*alphah_cv'*phi_xu(:,cv_index(cv_split==k)));
        ph_cv=alphah_cv'*phi_zw(:,cv_index(cv_split==k))./normalization_cv;

        switch loss_type
         case 'KL'
          score_tmp(k)=-mean(log(ph_cv+cc));
         case 'SQ'
          tmp=phi_xu(:,cv_index(cv_split==k))./repmat(normalization_cv,[b,1]);
          tmp2=(sqrt(pi)*sigma)^d_y*phi_vv.*(tmp*tmp');
          score_tmp(k)=alphah_cv'*tmp2*alphah_cv/sum(cv_split==k)/2-mean(ph_cv);
        end
      end % for fold
      
      score_cv(sigma_index,lambda_index)=mean(score_tmp);

    end % for lambda_index
  end % for sigma_index
  
  [score_cv_tmp,lambda_index]=min(score_cv,[],2);
  [score,sigma_index]=min(score_cv_tmp);
  lambda=lambda_list(lambda_index(sigma_index));
  sigma=sigma_list(sigma_index);

%  disp(sprintf('sigma = %g, lambda = %g',sigma,lambda))

  %%%%%%%%%%%%%%%% Computing the final solution `ph'
  phi_xu=exp(-xu_dist2/(2*sigma^2));
  phi_yv=exp(-yv_dist2/(2*sigma^2));
  phi_zw=phi_xu.*phi_yv;
  phi_vv=exp(-vv_dist2/(4*sigma^2));
  Phibar=(sqrt(pi)*sigma)^d_y*phi_vv.*(phi_xu*phi_xu');
  alphat=mylinsolve(Phibar/n+lambda*eye(b),mean(phi_zw,2));
  alphah=max(0,alphat);

  [dummy,n_test]=size(y_test);
  phi_xu_test=kernel_gaussian(x_test,u,sigma)';
  phi_yv_test=kernel_gaussian(y_test,v,sigma)';
  phi_zw_test=phi_xu_test.*phi_yv_test;
  normalization=max(cc,(sqrt(2*pi)*sigma)^d_y*alphah'*phi_xu_test);
  ph=(alphah'*phi_zw_test)./normalization;
  
  if nargout<2
    ph_error=nan;
  else
    switch loss_type
     case 'KL'
      ph_error=-mean(log(ph+cc));
     case 'SQ'
      tmp=phi_xu_test./repmat(normalization,[b 1]);
      Phibar_test=(sqrt(pi)*sigma)^d_y*phi_vv.*(tmp*tmp');
      ph_error=alphah'*Phibar_test*alphah/n_test/2-mean(ph);
    end
  end %nargout
  
