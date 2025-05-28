from sideKick import BayOpt_SideKick # Help functions
import numpy as np
import scipy.linalg as la # just import the functions that we use
from scipy.linalg import cholesky as chol
from scipy.linalg import cho_factor, cho_solve,solve_triangular
from scipy.optimize import minimize 
from scipy.stats import norm
from sys import float_info as floatinfo  # biggest number that can be represented 
import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.special import erfc, erfcx
import sys
from scipy.stats import qmc
floating_epsilon = sys.float_info.epsilon

class BayesianOptimization_light:

    def __init__(self,TS,CS,AS,x,y,Nmax):

        ### Target function settings
        #===========================
        self.d = TS['dim'] # dimension of the domain
        self.domain = TS['domain'] # domain bounds in the form [(lower, upper),(lower,upper),...], use None in place of -inf or +inf
        
        ### Covariance Settings
        #=======================
        self.fixed_param = CS['fixed_param'].astype(bool).copy() # a binary list indicating which parameters are to be fixed
        self.init_param = CS['init_param'].copy() # an array of initial parameters
        self.param = CS['init_param'].copy()      # set the current parameters to be equal to the initial parameters
        self.param_optimizer = CS['optimizer'] # Covariance parameter optimizer: {'Nelder-Mead','Powell','L-BFGS-B','Newton-CG',...}
        ### pick covariance function 
        #============================
        self.reconstruction_point = self.point_reconstruction
        if CS['Cf'] == "Matern05":
            self.Cf = self.Matern05 # self.Cf calculates the cov at distance (d)
            self.Cf_g = self.Matern05_with_gradient # self.Cf_g calculates the cov at distance(d) as well as the gradient w.r.t. (d)
            self.Cf_with_par_g = self.Matern05_param_gradient
            self.reconstruction_point = self.point_reconstruction_exp
        elif CS['Cf'] == "Matern15":
            self.Cf = self.Matern15
            self.Cf_g = self.Matern15_with_gradient
            self.Cf_with_par_g = self.Matern15_param_gradient
        elif CS['Cf'] == "Matern25":
            self.Cf = self.Matern25
            self.Cf_g = self.Matern25_with_gradient
            self.Cf_with_par_g = self.Matern25_param_gradient
        elif CS['Cf'] == "Gaussian":
            self.Cf = self.Gaussian_cov
            self.Cf_g = self.Gaussian_cov_with_gradient
            self.Cf_with_par_g = self.Gaussian_cov_param_gradient
        else:
           raise ValueError("Invalid covariance function.")

        self.AF_optimization = self.AF_optimization_sampling
        ### Pick Acquisition Function 
        #=========================
        
        if AS['Af'] == "LCB":
            self.Af = self.LCB # Af computes the acquisition at a given array
            self.Af_g = self.LCB_with_gradient #Af computes the acquisition and the gradient at a single point
        elif AS['Af'] == "EI":
            self.Af = self.EI
            self.Af_g = self.EI_with_gradient
        elif AS['Af'] == "PI":
            self.Af =self.PI
            self.Af_g = self.PI_with_gradient
        elif AS['Af'] == "logPI":
            self.Af = self.logPI
            self.Af_g = self.logPI_with_gradient
        elif AS['Af'] == 'logEI':
            self.Af = self.logEI
            self.Af_g = self.logEI_with_gradient 
        elif AS['Af'] == 'uniform':
            self.Af = self.logPI
            self.AF_optimization = self.AF_optimization_random
        elif AS['Af']== 'contextual EI':
            self.Af = self.EI_contextual
            self.Af_g = self.EI_with_gradient
        elif AS['Af'] == 'contextual logEI':
            self.Af = self.logEI_contextual
            self.Af_g = self.logEI_with_gradient

        else:
           raise ValueError("Invalid acquisition function.")

        ### Acquisition Settings 
        self.Af_optimizer = AS['optimizer'] # Acquisition optimizer: {'Nelder-Mead','Powell','L-BFGS-B','Newton-CG',...}
        self.Af_param = AS['Af_param'] # Acquisition parameter ( exploration-exploitation parameter)
        
        # initial evaluations
        self.ind = len(x) # index to keep track of what iteration we are at (how many evaluations we currently have)
        self.x = np.zeros([Nmax+ self.ind , self.d]) # allocate memory for x and y
        self.y_evals = np.zeros([Nmax+self.ind , 1])
        # save initial evaluations
        self.x[:self.ind] = x 
        self.y_evals[:self.ind] = y # save observations

        self.m = np.mean(self.y_evals[:self.ind]) # calculate mean estimate
        self.sd = np.std(self.y_evals[:self.ind]) # calculate standard deviation
        self.y = (self.y_evals - self.m)/self.sd # save standardized data for modelling 

        # save the best 10 points for optimization
        self.y_best_val = np.zeros((1,10))
        self.x_best_val = np.zeros((self.d,10))
    
        I = np.argsort(self.y_evals[0:self.ind],axis= 0).T[0]
        self.y_best_val = self.y_evals[I[:10]]
        self.x_best_val = self.x[I[:10]]
        self.ymin = self.y_best_val[0]
   
        # distances and covariances are updated sequentially whenever the gaussian parameters are not changed
        # initiate these objects
        self.D = np.zeros((self.d,Nmax + self.ind,Nmax+self.ind)) # difference tensor (v_x-u_x)^2 between all observations
        self.D[:,:self.ind,:self.ind] = BayOpt_SideKick.difference_matrices_self(self.x[:self.ind,:]) # distance matrix  = sqrt(sum(D),ax=0)
        self.Skk = np.zeros((self.ind+Nmax,self.ind+Nmax)) # covariance between observations
        self.iSY = np.zeros((self.ind,1)) # S_kk^(-1)Y
        self.L = np.zeros((self.ind, self.ind)) # cholesky factor

        # GP priors, calculating prior parameters TODO allow user to choose prior constants
        domain = BayOpt_SideKick.reshape_domain_for_grid(self.domain)
        self.len = domain[1,:]- domain[0,:]
        if CS['priors']=='uniform':
            self.nll_priors = self.nll_priors_uniform
        elif CS['priors']== 'fraiche':
            self.lambda_s = np.abs(np.log(CS['prior_alpha_std']))/(1)
            self.a_rho = 0.1
            self.lambda_rho = np.abs(np.log(self.a_rho))*(self.len)**(self.d/2)
            self.lambda_s = np.abs(np.log(0.1))/(1)
            self.nll_priors = self.n_ll_priors_fraiche
        elif CS['priors']== 'logNormal':
            self.lambda_s = np.abs(np.log(CS['prior_alpha_std']))/(1)
            self.prior_mu = np.sqrt(2) + np.log(np.sqrt(self.d)) + np.log(self.len)
            self.prior_sig2 = 3
            self.nll_priors = self.n_ll_priors_log_normal
  
### update Gaussian process parameters 
    def update_GP_g(self,n_max): # with exact gradients
        # update standardization
        self.m = np.mean(self.y_evals[:self.ind])
        self.sd = np.std(self.y_evals[:self.ind])
        self.y = (self.y_evals -self.m)/self.sd 
        p_init_log = np.log(self.param[~self.fixed_param])
        result = minimize(self.n_loglik_cov_wrapper_with_gradient, p_init_log,jac = True ,method=self.param_optimizer,options={'maxiter': n_max}) # the parameters are exp transformed in the wrapper to make sure parameters are positive
        if result.success == False:  
            print(f'Parm Estimation Failed in update_GP scipy.minimize, status:{result.success}')
        else:
            self.param[~self.fixed_param] = np.exp(result.x) 
        return result

    def update_GP(self,n_max): # with exact gradients
        # update standardization
        self.m = np.mean(self.y_evals[:self.ind])
        self.sd = np.std(self.y_evals[:self.ind])
        self.y = (self.y_evals -self.m)/self.sd 
        p_init_log = np.log(self.param[~self.fixed_param])
        result = minimize(self.n_loglik_cov_wrapper, p_init_log,jac = False ,method=self.param_optimizer,options={'maxiter': n_max}) # the parameters are exp transformed in the wrapper to make sure parameters are positive
        if result.success == False:  
            print(f'Parm Estimation Failed in update_GP scipy.minimize, status:{result.success}')
        else:
            self.param[~self.fixed_param] = np.exp(result.x) 
        return result


    def update_GP_SGD(self,nr_iter, learn_rate,nr_stc_v_Hutch_tr,tolerance = 1e-2,adams_patience = 5): # with approximate gradients and SGD
        # update standardization
        self.m = np.mean(self.y_evals[:self.ind])
        self.sd = np.std(self.y_evals[:self.ind])
        self.y = (self.y_evals -self.m)/self.sd  #TODO --MEAN--
        init_log_params = np.log(self.param[~self.fixed_param])
        final_params,grad_norm, nr_iter,loss = BayOpt_SideKick.Adam_SGD(self.torch_loglike_wrapper, num_iterations=nr_iter, init_params_np=init_log_params, learning_rate=learn_rate,arg = nr_stc_v_Hutch_tr,tol = tolerance,patience = adams_patience)
        self.param[~self.fixed_param] = np.exp(final_params)
        return (final_params, grad_norm, nr_iter, loss)

# Acquisition Optimization ####################################################################################

    def AF_optimization_sampling(self,nr_sobol_samples_base2 = 4,grad = True): 
        """
        Method: Evaluate the Acquisition on a uniformly sampled set of n points and pick the minimum as an initial point for a gradient based optimizer
        """
        
        # precompute necessary matrices ( for efficient for optimization) 
        Skk,invSY = self.Skk[:self.ind,:self.ind], self.iSY[:self.ind]
        # sample sobol points in d dimensions and scale to our domain
        sampler = qmc.Sobol(d=self.d, scramble=True)
        x_sample = sampler.random_base2(m=nr_sobol_samples_base2)
        x_sample = qmc.scale(x_sample, self.domain[:,0], self.domain[:,1])
        Af_sample = self.Af(x_sample,Skk,invSY) # evaluate on the samples
        Af_best_sample = self.Af(self.x_best_val,Skk,invSY)

        # initial point for optimization
        Imin = np.argmin(Af_sample) # use the minimum eval on the sample 
        x_init = x_sample[Imin,:]
        Imin_best = np.argmin(Af_best_sample) # use the minimum eval on the sample 
        x_best = self.x_best_val[Imin_best,:]
       
        # Acquisition optimization 
        result = minimize(self.Af_g, x_init, bounds=self.domain,jac=grad, method=self.Af_optimizer ,args=(Skk,invSY)) # what if optimization fails 
        result2 = minimize(self.Af_g, x_best, bounds=self.domain,jac=grad, method=self.Af_optimizer ,args=(Skk,invSY)) # what if optimization fails
        #print(f'iteration: {self.ind}, Acquisition optimization: {result.success}')
        if result.fun < result2.fun:
            return result.x,result
        else:
            return result2.x,result2
        

    def AF_optimization_sampling2(self,nr_sobol_samples_base2 = 4, nr_mvn_sample = 10, grad = True): 
        """
        Method: Evaluate the Acquisition on a uniformly sampled set of n points and pick the minimum as an initial point for a gradient based optimizer
        """
        
        # precompute necessary matrices ( for efficient for optimization) 
        Skk,invSY = self.Skk[:self.ind,:self.ind], self.iSY[:self.ind]
        # sample sobol points in d dimensions and scale to our domain
        sampler = qmc.Sobol(d=self.d, scramble=True)
        x_sample = sampler.random_base2(m=nr_sobol_samples_base2)
        x_sample = qmc.scale(x_sample, self.domain[:,0], self.domain[:,1])
        Af_sample = self.Af(x_sample,Skk,invSY) # evaluate on the samples
        Af_best_sample = self.Af(self.x_best_val,Skk,invSY)

        # initial point for optimization
        Imin = np.argmin(Af_sample) # use the minimum eval on the sample 
        x_init = x_sample[Imin,:]
        Imin_best = np.argmin(Af_best_sample) # use the minimum eval on the sample 
        x_best = self.x_best_val[Imin_best,:]

        # Acquisition optimization 
        result = minimize(self.Af_g, x_init, bounds=self.domain,jac=grad, method=self.Af_optimizer ,args=(Skk,invSY)) 
        result2 = minimize(self.Af_g, x_best, bounds=self.domain,jac=grad, method=self.Af_optimizer ,args=(Skk,invSY)) 
        
        X = np.random.multivariate_normal(x_best, np.diag(self.param[:self.d]), size=nr_mvn_sample)
        Bounds = self.domain
        mask = np.all(
            (X >= Bounds[:, 0]) &
            (X <= Bounds[:, 1]), axis=1)
        if np.sum(mask) > 0:
                    X = X[mask]
                    sample_mvn = self.Af(X,Skk,invSY) 
                    I_min = np.argmin(sample_mvn)
                    result3 = minimize(self.Af_g, X[I_min], bounds=self.domain,jac=grad, method=self.Af_optimizer ,args=(Skk,invSY))
                    result3f = result3.fun
        else: 
            result3f = result.fun +10
        

        #print(f'iteration: {self.ind}, Acquisition optimization: {result.success}')
        
        ii = np.argmin(np.array([result.fun,result2.fun,result3f]))
        if ii == 2:
            return result3.x,result3 
        elif ii== 1:
            return result2.x,result2
        else:
            return result.x,result 

        


# Save new evaluations ########################################################################################
    def save_eval(self,x, y): #saves the new evaluation point (x,y) and updates the iteration index 
        self.x[self.ind] = x
        self.y_evals[self.ind] = y
        self.y[self.ind] = (y- self.m)/self.sd # save the demeaned observation  

        # update the difference matrix self.D, the covariance matrix self.Skk, the cholesky factor self.L, and S_kk^(-1)Y
        self.update_Skk_iSY(x,(y-self.m)/self.sd,self.ind)

        self.ind += 1
        self.y_best_val[-1] = self.y_evals[self.ind-1] # put the new point as the last point in ymin
        self.x_best_val[-1] = x
        I = np.argsort(self.y_best_val,axis= 0).T[0] # sort ascending
        self.y_best_val = self.y_best_val[I]
        self.x_best_val = self.x_best_val[I]
        self.ymin = (self.y_best_val[0]-self.m)/self.sd
        #if self.y[self.ind-1] <= self.ymin: # update the best minimum if the current evaluation is lower 
        #    self.ymin = self.y[self.ind-1]
        #    self.xmin = self.x[self.ind-1]
        
        return None
    





# Covariance parameters loss & estimation ####################################################

    def n_loglik_with_approx_gradient(self,par_full,Hutchinson_nr_vec):
        
        rho = par_full[0:self.d]    # extract rho
        dist = np.sqrt(np.sum(self.D[:,:self.ind,:self.ind] / (rho[:, None, None]**2), axis=0)) # compute distance

        r,grad = self.Cf_with_par_g(dist,rho,par_full[-3],par_full[-2]) # evaluate the covariance function with gradient # TODO change to self.Cf_with_p_grad

        r =r + par_full[-1]*np.eye(*dist.shape) # add the nugget term

        try: 
            L,lower = cho_factor(r,lower= True) # Cholesky factor for the covariance function
            
            tr = BayOpt_SideKick.Hutchinsons_trace(L.T,grad,Hutchinson_nr_vec)
        
            # precompute Cov_kk^(-1)Y
            invSY = cho_solve((L,lower) ,self.y[:self.ind])
            
            nlog_p = np.sum(np.log(np.diag(L))) +0.5* np.dot(invSY.T,self.y[:self.ind]) # negative loglike loss

            g = 0.5*tr.flatten() - (0.5*invSY.T @ grad @ invSY).flatten() # full gradient with respect to the parameters

        except la.LinAlgError:
            print('Cholesky factor failed in n_log_lik_cov_param, gradient needs fixing')
            nlog_p = np.array([floatinfo.max/2])
            g = np.zeros(len(par_full)-1)
        
        nll_prior , grad_prior = self.nll_priors(par_full)
        return nlog_p + nll_prior, g+ grad_prior

    def n_loglik_with_gradient(self,par_full):
        
        rho = par_full[0:self.d]    # extract rho
        dist = np.sqrt(np.sum(self.D[:,:self.ind,:self.ind] / (rho[:, None, None]**2), axis=0)) # compute distance

        r,grad = self.Cf_with_par_g(dist,rho,par_full[-3],par_full[-2]) # evaluate the covariance function with gradient 

        r =r + par_full[-1]*np.eye(*dist.shape) # add the nugget term


        try: 
            L,lower = cho_factor(r,lower= True) # Cholesky factor for the covariance function
            

            tr = np.zeros(len(par_full)-1)
            for i in range(grad.shape[0]): # compute gradients with respect to the first term of the nloglik loss
                tr[i] = 0.5*np.trace(cho_solve((L,lower),grad[i]))
        
            # precompute Cov_kk^(-1)Y
            invSY = cho_solve((L,lower) ,self.y[:self.ind])
            nlog_p = np.sum(np.log(np.diag(L))) +0.5* np.dot(invSY.T,self.y[:self.ind]) # negative loglike loss
    
            g = tr.flatten() - (0.5*invSY.T @ grad @ invSY).flatten() # full gradient with respect to the parameters

        except la.LinAlgError:
            print('Cholesky factor failed in n_log_lik_cov_param, gradient needs fixing')
            nlog_p = np.array([floatinfo.max/2])
            g = np.zeros(len(par_full)-1)
        nll_prior , grad_prior = self.nll_priors(par_full)
        return nlog_p + nll_prior, g+ grad_prior
#################
    def n_loglik_cov_wrapper(self,par_free):
        full_params = self.init_param.copy()
        full_params[~self.fixed_param] = np.exp(par_free)
        value = self.n_loglik(full_params) 
        return value

    def n_loglik(self,par_full):
            
            rho = par_full[0:self.d]    # extract rho
            dist = np.sqrt(np.sum(self.D[:,:self.ind,:self.ind] / (rho[:, None, None]**2), axis=0)) # compute distance

            r,grad = self.Cf_with_par_g(dist,rho,par_full[-3],par_full[-2]) # evaluate the covariance function with gradient 

            r =r + par_full[-1]*np.eye(*dist.shape) # add the nugget term

            try: 
                L,lower = cho_factor(r,lower= True) # Cholesky factor for the covariance function
                # precompute Cov_kk^(-1)Y
                invSY = cho_solve((L,lower) ,self.y[:self.ind])
                nlog_p = np.sum(np.log(np.diag(L))) +0.5* np.dot(invSY.T,self.y[:self.ind]) # negative loglike loss

            except la.LinAlgError:
                print('Cholesky factor failed in n_log_lik_cov_param, gradient needs fixing')
                nlog_p = np.array([floatinfo.max/2])

            return nlog_p 
##############

    def n_loglik_cov_wrapper_with_gradient(self,par_free):  ## tested
        """"
        Takes only the free parameters and evaluates the n_loglike of the cov parameters
        """
        full_params = self.init_param.copy()
        full_params[~self.fixed_param] = np.exp(par_free)
        value, grad = self.n_loglik_with_gradient(full_params) 
        return value[0], grad[~self.fixed_param[0:-1]]*np.exp(par_free) # multiply by inner derivative

    def torch_loglike_wrapper(self,params,arg):

        # Detach the tensor so that no gradient history is recorded, and convert to NumPy
        param_np = params.detach().numpy()
    
        # Call the external function that returns loss and gradient as NumPy arrays.
        # (Assume optimizer.n_loglik_cov_wrapper_with_gradient is defined elsewhere.)
        
        full_params = self.init_param.copy()
        full_params[~self.fixed_param] = np.exp(param_np)
        value, grad = self.n_loglik_with_approx_gradient(full_params,arg) 
        v,g = value, grad[~self.fixed_param[0:-1]]*np.exp(param_np)
  

        # Convert the loss (v) to a PyTorch tensor; convert the gradient (g) and ensure it is of type float.
        return torch.from_numpy(v), torch.from_numpy(g)



## Acquisition Functions (Afs) ##################################################################

    def LCB(self,x,Skk,invSY): # lower confidence bound acquisition, expects x to be 2D array (columns = different dimensions)
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY)
        return y_rec-self.Af_param*np.sqrt(y_var)
    
    def LCB_with_gradient(self,x,Skk,invSY): # ( Lower confidence bound)
        # compute reconstruction on x and precompute some other expressions
        y_rec,y_var,Duk,Suk,Grad_Suk,Skk_inv_SukT = self.reconstruction_point(x,Skk,invSY)
        # gradient of LCB with respect to x
        g= Grad_Suk@(invSY) +self.Af_param* (Grad_Suk@(2*Skk_inv_SukT))/(2*np.sqrt(y_var)) 
        return y_rec-self.Af_param*np.sqrt(y_var), g
    #--------------------------------------------------------------------------
    def EI(self,x,Skk,invSY): # Expected improvement acquisition, expects x \in R^d, 2D array with (columns = different dimensions)
        # compute reconstruction mean and variance
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY) 
        y_sig = np.sqrt(y_var) # standard devation
        impr = (self.ymin)- y_rec-self.Af_param
        z= impr/(y_sig)     # scaled improvement                                                   
        # negative EI(x) 
        l = -(impr*norm.cdf(z)+ y_sig * norm.pdf(z))
        return l
    def EI_contextual(self,x,Skk,invSY): # Expected improvement acquisition, expects x \in R^d, 2D array with (columns = different dimensions)
        # compute reconstruction mean and variance
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY) 
        self.Af_param = np.mean(y_var)/np.abs(self.ymin) 
        y_sig = np.sqrt(y_var) # standard devation
        impr = (self.ymin)- y_rec-self.Af_param
        z= impr/(y_sig)     # scaled improvement                                                   
        # negative EI(x) 
        l = -(impr*norm.cdf(z)+ y_sig * norm.pdf(z))
        return l

    def EI_with_gradient(self,x,Skk,invSY): # Expected improvement acquisition, works only for one point in R^d, expects a 2D array (row vector)
        # compute reconstruction on x and precompute some other expressions
        y_rec,y_var,Duk,Suk,Grad_Suk,Skk_inv_SukT = self.reconstruction_point(x,Skk,invSY)
        y_sig = np.sqrt(y_var)
        # compute EI(x)
        impr = (self.ymin)- y_rec-self.Af_param
        z= impr/(y_sig)     # scaled improvement      
        l =-(impr*norm.cdf(z)+ y_sig * norm.pdf(z)) # negative EI(x)

        # compute the gradient for optimization # nabla sigma^2 = -2 gSuk*S_kk_inv*Suk^T/sigma
        G_Suk_S_kk_inv_SukT = -Grad_Suk@(Skk_inv_SukT)
        g= (Grad_Suk@(invSY))*norm.cdf(z) - norm.pdf(z)*G_Suk_S_kk_inv_SukT/(y_var )   
        return l,g
    #--------------------------------------------------------------------------
    def PI(self,x,Skk,invSY): # Probability of improvement acquisition, expects 2D array 
        # compute reconstruction mean and variance
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY)
        # scaled improvement
        z= (self.ymin- y_rec-self.Af_param)/(np.sqrt(y_var))                  
        return - norm.cdf(z)

    def PI_with_gradient(self,x,Skk,invSY):
        # compute reconstruction mean and variance
        y_rec,y_var,Duk,Suk,Grad_Suk,Skk_inv_SukT = self.reconstruction_point(x,Skk,invSY)
        sig = np.sqrt(y_var)
        z= (self.ymin- y_rec-self.Af_param)/(sig)    # scaled improvement            
        # gradient for optimization
        G_Suk_S_kk_inv_SukT = -Grad_Suk@(Skk_inv_SukT)
        g= (Grad_Suk@(invSY)/sig  + z*(G_Suk_S_kk_inv_SukT)/(y_var))*norm.pdf(z)   
        return - norm.cdf(z),g
    #--------------------------------------------------------------------------
    


    def logPI(self,x,Skk,invSY):
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY)
        
        z= (self.ymin- y_rec-self.Af_param)/(np.sqrt(y_var))                    
        z2 = - z/np.sqrt(2)
        neg_z2 = (z2 <= 0)
        neglogpi = np.ones(z2.shape)
        neglogpi[neg_z2] = -np.log(erfc(z2[neg_z2])) +np.log(2)
        neglogpi[~neg_z2] =  -np.log(erfcx(z2[~neg_z2]))+z2[~neg_z2]**2 +np.log(2)
        return neglogpi


    def logPI_with_gradient(self,x,Skk,invSY):
        y_rec,y_var,Duk,Suk,Grad_Suk,Skk_inv_SukT = self.reconstruction_point(x,Skk,invSY)
        y_var = y_var 
        z= (self.ymin- y_rec-self.Af_param)/(np.sqrt(y_var))                    #
        z2 = - z/np.sqrt(2)

        G_Suk_S_kk_inv_SukT = -Grad_Suk@(Skk_inv_SukT)
        dz= -(Grad_Suk@(invSY)/np.sqrt(y_var)  + z*(G_Suk_S_kk_inv_SukT)/(y_var))       


        if z2 <= 0:
    
            nlogpi = -np.log(erfc(z2)) +np.log(2)
            ngrad = (-np.sqrt(2/np.pi)/erfcx(z2) ) * dz
            
        else:
        
            nlogpi = -np.log(erfcx(z2))+z2**2 +np.log(2)
            ngrad =  (-np.sqrt(2/np.pi)/erfcx(z2) )* dz 

        return nlogpi, ngrad
    #--------------------------------------------------------------------------
    
    def logEI(self,x,Skk,invSY):
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY)
        y_sig = np.sqrt(y_var)   
                       
        impr = (self.ymin)- y_rec-self.Af_param

        z = impr/y_sig
        nlogei = - self.ln_h(z ) - np.log(y_sig) 

        return nlogei
    
    def logEI_contextual(self,x,Skk,invSY):
        y_rec,y_var = self.reconstruction_Skk(x,Skk,invSY)
        y_sig = np.sqrt(y_var)   
        self.Af_param = np.mean(y_var)/np.abs(self.ymin)                
        impr = (self.ymin)- y_rec-self.Af_param

        z = impr/y_sig
        nlogei = - self.ln_h(z ) - np.log(y_sig) 

        return nlogei 
    
    def logEI_with_gradient(self,x,Skk,invSY):
        y_rec,y_var,Duk,Suk,Grad_Suk,Skk_inv_SukT = self.reconstruction_point(x,Skk,invSY)
        y_sig = np.sqrt(y_var)                  
        impr = (self.ymin)- y_rec-self.Af_param
        z = impr/y_sig
        nlogei = - self.ln_h_point(z ) - np.log(y_sig) 

        G_Suk_S_kk_inv_SukT = -Grad_Suk@(Skk_inv_SukT)
        dz = -(Grad_Suk@(invSY)/y_sig  + z*(G_Suk_S_kk_inv_SukT)/(y_var))
        nlogei_grad = - dz*self.d_logh(z) -(-2*Grad_Suk@(Skk_inv_SukT)) /(2*y_var)
        return nlogei, nlogei_grad


    
    def ln_h(self,x):
        invsqrt_eps = 1/np.sqrt(floating_epsilon)
        c1 = np.log(2*np.pi)/2 
        c2 = np.log(np.pi/2)/2
        logh = np.ones(x.shape)
        case1 = (x >= -1)
        case2 = ((x > -invsqrt_eps) & (x < -1))
        case3 = (x <= -invsqrt_eps)
        x1 = x[case1]
        x2= x[case2]
        x3 = x[case3]
        logh[case1] = np.log( norm.pdf(x1) + x1*norm.cdf(x1) )
        logh[case2]= + self.log1mexp(c2 + np.log( -x2 *erfcx(  -x2 /np.sqrt(2) ) ) ) -x2**2/2 -c1 
        logh[case3] =  - 2*np.log(-x3) -x3**2/2 -c1 
      
        return logh
    
    def ln_h_point(self,x):
        invsqrt_eps = 1/np.sqrt(floating_epsilon)
        c1 = np.log(2*np.pi)/2 
        c2 = np.log(np.pi/2)/2
        logh = np.ones(x.size)

        if x>= -1:
            logh = np.log(norm.pdf(x)+x*norm.cdf(x))
        elif x <= -invsqrt_eps:
            logh=  -x**2/2 -c1 - 2*np.log(-x)
        else:
            logh=  -x**2/2 -c1 + self.log1mexp(c2 + np.log( -x *erfcx(  -x /np.sqrt(2) ) ) )

        return logh

    
    def d_logh(self,z):
          
        if z > 5:
            return 1/z
        elif z< -30:
            return (z -1/z + 3/z**3)/(-1+3/z**2)
            #return -z
        else:
            return   norm.cdf(z)/(z*norm.cdf(z)+norm.pdf(z))
    
    def log1mexp(self,x): # see the logEI paper
        lexp = np.ones(x.shape)
        case1 = x > -np.log(2)
        lexp[case1] = np.log(  -np.expm1(x[case1]))
        lexp[~case1] =  np.log1p(-np.exp(x[~case1]))
        return lexp
    

# Reconstruction ################################################################################################
    def add_param(self,param): #  help function (for debugging), manually update the cov parameters
        self.param = param
        return None

    def reconstruction_Skk(self,xu,Skk,invSY):
        """Computes the reconstruction on a new set of points when Skk and invSY are precomputed"""
        # expects xu to be a simple array of length d= nr of dimensions

        rho, cov_par,sig_ep2 = self.extract_param()

        # compute Suk: cross covariance matrix
        Duk = BayOpt_SideKick.difference_matrices(xu,self.x[:self.ind,:]) 
        Duk = np.sqrt(np.sum(Duk / (rho[:, None, None]**2), axis=0))    
        Suk = self.Cf(Duk,*cov_par )

        m_u = Suk@invSY
        v_u = cov_par[0] - np.sum((solve_triangular(self.L,Suk.T,lower= True)**2),axis= 0)
        v_u = np.maximum(v_u,0)
        v_u += sig_ep2
        return m_u,v_u.reshape(-1,1)
    
    def point_reconstruction(self,xu,Skk,invSY): # recurrent function (internal optimization) = > ensure numerical efficiency and stability
        """Performs kriging for a single point (to be used in Acquisition optimization)"""
        rho, cov_par, sig_ep2 = self.extract_param()

        # compute Suk: cross covariance matrix
        Duk = BayOpt_SideKick.distance_anisotropic(xu,self.x[:self.ind,:],rho) 
        Suk,grad_Suk_div_Duk = self.Cf_g(Duk,*cov_par )
        m_u = Suk@invSY # reconstruction mean
    
        if self.Lf == True:
            Skk_inv_SukT = cho_solve((self.L,True), Suk.T)
        else:
            Skk_inv_SukT = np.linalg.lstsq(Skk, Suk.T,rcond= None)[0]

        v_u = (cov_par[0]) -np.dot(Suk, Skk_inv_SukT)
        v_u = np.maximum(v_u,0) # make sure it is possitive ( in case numerical instability cases v <0)
        v_u += sig_ep2

        # compute gradient of Suk
        rho = self.param[0:self.d]

        Grad_Suk =(grad_Suk_div_Duk)*((xu-self.x[:self.ind])/rho**2).T

        return m_u, v_u, Duk, Suk,Grad_Suk,Skk_inv_SukT
    

    def point_reconstruction_exp(self,xu,Skk,invSY): # recurrent function (internal optimization) = > ensure numerical efficiency and stability
        """Performs kriging for a single point (to be used in Acquisition optimization)"""
        rho, cov_par, sig_ep2 = self.extract_param()

        # compute Suk: cross covariance matrix
        Duk = BayOpt_SideKick.distance_anisotropic(xu,self.x[:self.ind,:],rho) 
        Suk,guk = self.Matern05_with_gradient(Duk,*cov_par )
        m_u = Suk@invSY # reconstruction mean
    
        if self.Lf == True:
            Skk_inv_SukT = cho_solve((self.L,True), Suk.T)
        else:
            Skk_inv_SukT = np.linalg.lstsq(Skk, Suk.T,rcond= None)[0]

        v_u = (cov_par[0]) -np.dot(Suk, Skk_inv_SukT)
        v_u = np.maximum(v_u,0) # make sure it is possitive ( in case numerical instability cases v <0)
        v_u += sig_ep2

        # compute gradient of Suk
        rho = self.param[0:self.d]
        try:
            with np.errstate(divide='raise', invalid='raise'):
                Grad_Suk =(guk/Duk)*((xu-self.x[:self.ind])/rho**2).T
     
        except FloatingPointError:
            Grad_Suk= (guk)*((np.ones((1,self.d)))/rho**2).T

        return m_u, v_u, Duk, Suk,Grad_Suk,Skk_inv_SukT
    
    
    def compute_Skk_iSY(self): 
        """Precomputes some matrices for more efficient calculations, use only after updating cov param, otherwise use update_Skk"""

        rho, cov_par, sig_ep2 = self.extract_param()

        # compute S_kk: covariance matrix for the known locations
        Dkk = np.sqrt(np.sum(self.D[:,:self.ind,:self.ind] / (rho[:, None, None]**2), axis=0)) 
        Skk = self.Cf(Dkk,*cov_par ) + sig_ep2*np.eye(*Dkk.shape)
        self.Skk[:self.ind,:self.ind] = Skk
        # reconstruction mean and variance
        try: # faster routine
            L = la.cholesky(Skk[:self.ind,:self.ind], lower=True)
            self.L[:self.ind,:self.ind] = L
            self.Lf = True
            self.iSY = cho_solve((L,True),self.y[:self.ind])
             
        except la.LinAlgError:
            print('cholesky failed')
            self.Lf = False
            self.iSY = np.linalg.lstsq(Skk, self.y[:self.ind], rcond=None)

        return None

    def update_Skk_iSY(self,x_new,y_new,ind):
        """Update cholesky factor, Skk, iSY"""

        rho, cov_par, sig_ep2 = self.extract_param()

        Dnk = BayOpt_SideKick.difference_matrices(x_new,self.x[:ind])
        self.D[:, ind:ind+1 , :ind] = Dnk
        self.D[:, :ind, ind:ind+1 ] = np.transpose(Dnk, (0, 2, 1))

        Dnk = np.sqrt(np.sum(Dnk / (rho[:, None, None]**2), axis=0)) # convert to distances
        Snk = self.Cf(Dnk,*cov_par) # add sig_ep2 + sig2 on diagonal

        self.Skk[ind:ind+1,:ind] = Snk
        self.Skk[:ind, ind:ind+1] = Snk.T
        self.Skk[ind, ind] = sig_ep2 + cov_par[0]

        # update S_kk^(-1)Y
        iSkSkn = cho_solve((self.L,True),Snk.T)
        dev = (np.dot(Snk,self.iSY)-y_new)*(1/(sig_ep2 + cov_par[0]- np.dot(Snk,iSkSkn)))
        iSY = np.zeros((ind+1,1))
        iSY[:ind] = self.iSY + iSkSkn*dev 
        iSY[ind] = -dev
        self.iSY = iSY

        # update the cholesky factor
        w = solve_triangular(self.L,Snk.T,lower= True).flatten()
        self.L = np.pad(self.L, ((0, 1), (0, 1)), mode='constant')
        self.L[-1, :-1] = w
        self.L[-1, -1] = np.sqrt(sig_ep2 + cov_par[0]- np.dot(w, w))


        return None

# Covariance Functions ##########################################################################

    def Matern05(self,dist, sigma2, kappa):
        d = dist / kappa
        r = sigma2 * np.exp(-d)
        return r
    
    def Matern15(self,dist, sigma2, kappa):
        d = np.sqrt(3) * dist / kappa
        r = sigma2 * (1 + d) * np.exp(-d)
        return r

    def Matern25(self,dist, sigma2, kappa):
        d = np.sqrt(5) * dist / kappa
        r = sigma2 * (1 + d + (d ** 2) / 3) * np.exp(-d)
        return r

    def Gaussian_cov(self,dist, sigma2, kappa):
        r = sigma2 * np.exp(-2 * (dist / kappa) ** 2)
        return r
    

### Covariance functions that also compute the gradients, (for gradient based optimization)
# note: to handle the limit where d--> correctly for grad Suk we return (dr/d dist)/dist (the additional distance comes from the cov derivative)
# except in the case of the exponential (matern05) then we return the gradient 
    def Matern05_with_gradient(self,dist, sigma2, kappa): # gradient checked
        d = dist / kappa
        r = sigma2 * np.exp(-d)
        g = -(1/kappa)*r
        return r,g
    
    def Matern15_with_gradient(self,dist, sigma2, kappa): # gradient checked
        c = np.sqrt(3)/kappa
        d = dist*c
        r = sigma2 * (1 + d) * np.exp(-d)
        g_div_dist = -c**2*sigma2*np.exp(-d) # = -(c)*r + sigma2*(c)*np.exp(-d)
        return r,g_div_dist

    def Matern25_with_gradient(self,dist, sigma2, kappa): # gradient checked
        c = np.sqrt(5)/kappa  # constant
        d = c* dist 
        exp_d = np.exp(-d)
        r = sigma2 * (1 + d + (d ** 2) / 3) * exp_d 
        g_div_dist = - c**2*sigma2*(1+d)*exp_d/3.0 #g = -c*r + sigma2*(c + 2*c*d/3)*np.exp(-d)  #
        return r,g_div_dist 

    def Gaussian_cov_with_gradient(self,dist, sigma2, kappa):  # gradient checked
        r = sigma2 * np.exp(-2 * (dist / kappa) ** 2)
        g_div_dist = -(4/kappa**2)*r
        return r,g_div_dist


    def extract_param(self):
        rho= self.param[0:self.d]       # axis-anisotropy
        cov_par = self.param[self.d:self.d+2]       # covariance parameters
        sig_ep2 = self.param[-1]        # nugget variance
        return rho, cov_par, sig_ep2

###################################################

    def Matern05_param_gradient(self,dist, rho,sigma2, kappa):

        d_s =  dist / kappa # scaled distance
        exp_d_s = np.exp(-d_s) # = dr/(d sigma2)
        r = sigma2 * exp_d_s # cov matrix
        d_kappa =   (dist / kappa**2)*r  # derivative in kappa
        dist = dist + np.eye(*dist.shape) # off set the diagonal in dist
        # Initialize d_rho with zeros where dist is zero
        d_rho = np.zeros((rho.shape[0], dist.shape[0], dist.shape[1]))
        denom = dist[None, :, :] * kappa
        # Use np.divide with the 'where' parameter to avoid division where denom==0.
        d_rho=np.divide(r[None, :, :], denom, out=d_rho, where=(denom != 0)) 
        d_rho *= (self.D[:,:self.ind,:self.ind] / (rho[:, None, None] ** 3)) # multiply by D/rho  

        return r, np.concatenate((d_rho, exp_d_s[None,:,:], d_kappa[None,:,:]),axis= 0)

    def Matern15_param_gradient(self,dist, rho,sigma2, kappa):
        d_s = np.sqrt(3) * dist / kappa # scaled distance
        exp_d_s = np.exp(-d_s)
        d_sig2 = (1 + d_s) * exp_d_s # derivative in sig2
        r = sigma2 * d_sig2 # cov matrix
        d_kappa =  3 * (sigma2 * dist**2*exp_d_s )/ kappa**3  # derivative in kappa
        d_rho = (3*sigma2/kappa**2)* (self.D[:,:self.ind,:self.ind]*exp_d_s[None, :, :] / (rho[:, None, None]**3 )) 

        return r, np.concatenate((d_rho, d_sig2[None,:,:], d_kappa[None,:,:]),axis= 0)

    def Matern25_param_gradient(self,dist, rho,sigma2, kappa):

        d_s = np.sqrt(5) * dist / kappa # scaled distance
        exp_d_s = np.exp(-d_s)
        d_sig2 = (1 + d_s + d_s**2 /3) * exp_d_s # derivative in sig2
        r = sigma2 * d_sig2 # cov matrix
        d_kappa =   sigma2*d_s**2 *(1+d_s)*exp_d_s/(3*kappa) # derivative in kappa
        d_rho =(sigma2*5 *(1+ d_s)*exp_d_s/(3*kappa**2))[None, :,:]* (self.D[:,:self.ind,:self.ind] / (rho[:, None, None]**3 )) 

        return r, np.concatenate((d_rho, d_sig2[None,:,:], d_kappa[None,:,:]),axis= 0)

    def Gaussian_cov_param_gradient(self,dist, rho,sigma2, kappa):
        d_s = dist / kappa
        exp_d_s = np.exp(-2*d_s**2)
        r = sigma2 * exp_d_s
        d_kappa = 4*dist**2/kappa**3 *r
        d_rho = (4*r/kappa**2)[None,:,:]* (self.D[:,:self.ind,:self.ind] / (rho[:, None, None]**3 )) 
        return r, np.concatenate((d_rho, exp_d_s[None,:,:], d_kappa[None,:,:]),axis= 0)

############################################## priors for the GP parameters
    def n_ll_priors_fraiche(self,par_full): # fraiche
        d = self.d
        grad = np.zeros(d+2)
        nll_s2 = self.lambda_s*np.sqrt(par_full[-3])+ (1/2)*np.log(par_full[-3])  # prior for the 
        grad[-2] = self.lambda_s/(2*np.sqrt(par_full[-3])) + 1/(2*par_full[-3])

        nll_rho = np.sum((d/2 +1)* np.log(par_full[:d]) + self.lambda_rho* par_full[:d]**(-d/2))
        grad[:d] = (d/2 +1)/par_full[:d] - (d/2)*self.lambda_rho*par_full[:d]**(-d/2 -1)
        return nll_rho+nll_s2 , grad

    def n_ll_priors_log_normal(self,par_full): # log normal

        d = self.d
        grad = np.zeros(d+2)

        nll_s2 = self.lambda_s*np.sqrt(par_full[-3])+ (1/2)*np.log(par_full[-3])  # prior for the 
        grad[-2] = self.lambda_s/(2*np.sqrt(par_full[-3])) + 1/(2*par_full[-3])

        c = 1- self.prior_mu/self.prior_sig2
        log_rho = np.log(par_full[:d])

        nll_rho = np.sum(log_rho*c+ (log_rho)**2/(2*self.prior_sig2)) # log normal priors
        grad[:d] = c/par_full[:d] + log_rho/(self.prior_sig2*par_full[:d])
        return nll_rho + nll_s2, grad

    def nll_priors_uniform(self,par_full):
        return 0,0

#######################################

    def print_init_report(self): # print some info after initiation
        print('Domain and initial points info:')
        print('---------------------------------')
        print(f' Dimension : {self.d}')
        print(f' Domain : {self.domain}')
        print(f' Input dimensions X: {self.x.shape}, Y: {self.y.shape}')
        print(f' Number of initial points : {self.ind}')
        print('Other:')
        print('-----------------------------------')
        print(f' Acquisition parameter : {self.Af_param}')
        print(f' Initial Cov param : {self.init_param}')
        print(f' Fixed Cov param : {self.fixed_param}')
        print(f'X initial {self.x}')
        print(f'y initial {self.y_evals}')
        print('-----------------------------------')
        print('-----------------------------------')

##########################################################################


# TODO symmetric matrices, redundant computation?
# TODO for unbounded optimization scipy optimize can take bounds [ (None , None), (other bounds),...] but then be careful with the grid generation

# TODO make a help function that does initial estimation of cov parameters

# Acquisition optimization
# TODO there is a problem with PI and EI after sufficient number of evaluations the acquisition function is very flat which means that the BO will repeatedly pick the first point in the domain
# TODO knowledge Gradient




# Sequential update and batching 

# TODO consider sequential update for distance matrices and for Skk if the parameters are fixed (batch approach)

# Hyperparameter estimation



