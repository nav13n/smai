function px=pdf_Gaussian_CDF(x,mu,sigma)
  
  px=(1./sqrt(2*pi*sigma.^2)).*exp(-((x-mu).^2)./(2*sigma.^2));
