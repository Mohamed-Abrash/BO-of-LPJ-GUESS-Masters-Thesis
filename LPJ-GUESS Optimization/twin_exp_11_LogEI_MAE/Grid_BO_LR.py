import time
t1 = time.time()
import numpy as np
import os
os.chdir('/home/er5728de/Code')
from bayesian_optimization_with_regression import BayesianOptimization_regression as BO
from sideKick import BayOpt_SideKick
import OptimizationTestFunctions  as OptTF
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc
import subprocess
from tqdm import tqdm

'''
df_loss = pd.read_csv("/home/er5728de/Master_thesis/methane_assimilation2/build/param_loss_3dim.csv")
acrotelm_por = df_loss['acrotelm_por'].to_numpy()
Fgas = df_loss['Fgas'].to_numpy()
decay_length = df_loss['decay_length'].to_numpy()
Loss = np.sqrt(df_loss['loss'].to_numpy())
loss_std = np.sqrt(np.var(Loss))
Loss_stan = Loss/loss_std # standardize the variance
'''


def read_parameters(file_path):
    """
    Reads parameters from a text file with comments and returns a dictionary.

    Args:
        file_path (str): The path to the text file.

    Returns:
        dict: A dictionary containing the parameter names as keys and their values as integers.
    """

    parameters = {} # params dict
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('!') or not line:  # Skip comments and blank lines
                continue
            try:
                key, value = line.split()
                parameters[key] = float(value)  # Convert value to integer
            except ValueError:
                print(f"Warning: Could not parse line: {line}")
    return parameters

def write_parameters(file_path, parameters):
    """
    Writes parameters to a text file with comments.

    Args:
        file_path (str): The path to the text file.
        parameters (dict): A dictionary containing the parameter names as keys and their values as integers.
    """
    with open(file_path, 'w') as f:
        f.write("! parameters\n")
        for key, value in parameters.items():
            f.write(f"{key} {value}\n\n")  # Add an extra newline for better readability

filename = "BO_state_regression.txt"

def status_update(status,file):
    with open(filename, "a") as file:
        # Write the line and add a newline character at the end
        file.write(status + "\n")

def loss_function(y, y_hat):
    #return np.mean((y_hat-y)**2)
    return np.mean(np.abs(y_hat-y))

# Get the bounds of the grid
#Params: acrotelm_por, Fgas, decay_length
#domain = np.array([[0.1, 0, 0.1],  # Lower bounds
#                   [1.25, 0.03, 26]]) # Upper bounds

domain = np.array([(0.01, 1), (0.01, 0.3), (0.35, 1.1), (0.15, 1.25), (0, 0.015), 
                   (0.1, 1.25), (0.1, 1.25), (0, 0.03), (0, 0.1), (0.1, 26)])
wd = '/home/er5728de/Master_thesis/methane_assimilation2/build'
os.chdir(wd)
param_path = '/home/er5728de/Master_thesis/methane_assimilation2/build/guess_Params.ins'  # Replace with your file path
with open(filename, "w") as file:
    pass

df = pd.read_csv("/home/er5728de/Code/test_data.csv")
#df = pd.read_csv("/home/er5728de/Code/sii1.csv")
Value = df['CH4'].to_numpy()

def TF(x):
    # Achley
    #return 10 * x.shape[-1] + np.sum(x**2 - 10*np.cos(2*np.pi*x),axis= x.ndim-1)
    # Spherical
    #return np.sum(x**2,axis= x.ndim-1)

    params = read_parameters(param_path)
    params['RMOIST'] = x[0]
    params['CH4toCO2_peat'] = x[1]
    params['oxid_frac'] = x[2]
    params['tiller_por'] = x[3]
    params['tiller_radius'] = x[4]
    params['catotelm_por'] = x[5]
    params['acrotelm_por'] = x[6]
    params['Fgas'] = x[7]
    params['RMOIST_ANAEROBIC'] = x[8]
    params['decay_length'] = x[9]
    write_parameters(param_path, params)
    # Define the command as a single string
    command = '/home/er5728de/Master_thesis/methane_assimilation2/build/guess guess_2.ins'
    # Access the output and return code
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    out_path = '/home/er5728de/Master_thesis/methane_assimilation2/build/dch4.out'
    df_est = pd.read_fwf(out_path)
    df_est = df_est[(df_est['Year'] >= 2005) & (df_est['Year'] <= 2015)]
    Value_est = df_est['Value'].to_numpy()
    mse = loss_function(Value, Value_est)
    #loss = np.sqrt(mse)
    loss = mse
    
    return loss


def Plot_Value(x):
    # Rastrigin
    #return 10 * x.shape[-1] + np.sum(x**2 - 10*np.cos(2*np.pi*x),axis= x.ndim-1)
    # Spherical
    #return np.sum(x**2,axis= x.ndim-1)

    params = read_parameters(param_path)
    params['RMOIST'] = x[0]
    params['CH4toCO2_peat'] = x[1]
    params['oxid_frac'] = x[2]
    params['tiller_por'] = x[3]
    params['tiller_radius'] = x[4]
    params['catotelm_por'] = x[5]
    params['acrotelm_por'] = x[6]
    params['Fgas'] = x[7]
    params['RMOIST_ANAEROBIC'] = x[8]
    params['decay_length'] = x[9]
    write_parameters(param_path, params)
    # Define the command as a single string
    command = '/home/er5728de/Master_thesis/methane_assimilation2/build/guess guess_2.ins'
    # Access the output and return code
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    out_path = '/home/er5728de/Master_thesis/methane_assimilation2/build/dch4.out'
    df_est = pd.read_fwf(out_path)
    df_est = df_est[(df_est['Year'] >= 2005) & (df_est['Year'] <= 2015)]
    Value_est = df_est['Value'].to_numpy()
    return Value_est



#np.random.seed(1) # random seed
d= 10 # dimension of the domain
#N_init = 20 # number of initial points

#Initial Points using Sobol sampler
#Create Sobol sampler for 2 dimensions
sampler = qmc.Sobol(d, scramble=True) #d dimensions
samples = sampler.random_base2(m = 7) #Generates 2^m points
l_bounds = domain[:, 0]
u_bounds = domain[:, 1]
X_init = qmc.scale(samples, l_bounds, u_bounds)

def Init_Points(d, X):
    Y = np.array([TF(row) for row in X])
    Y = Y.reshape(-1, 1)
    return Y

Y_init = Init_Points(d, X_init)


#Initial Points using Uniform Dsitribution
#X_init = np.column_stack([
#    np.random.uniform(x_min, x_max, N_init),  # First parameter range
#    np.random.uniform(y_min, y_max, N_init)   # Second parameter range
#])
#X_init = np.random.uniform(0,26,2*N_init).reshape(N_init,2) # initial points
#Y_init = TF(X_init).reshape(-1,1) # Evaluations of the Target function on the initial points
#================================================================== cov settings: param = [rho1 rho2 ... rhod sig2 kappa sig_ep2 ]
rho_init = (domain[:,1]-domain[:, 0]) # initial ranges
CS = {'Cf': 'Matern25', # Covariance function: Matern05, Matern15, Matern25, Gaussian
      'init_param': np.array([*rho_init, 1, 1 ,10**(-5)]), # initial hyperparameters for the GP covariance function
      'fixed_param': np.array([*np.zeros(d), 0,1,1]), # fixed parameters: 0 free, 1 fixed parameter
      'optimizer': 'L-BFGS-B',
      'priors': 'logNormal',
      'prior_alpha_std': 0.1, # P(std > 1) = alpha_std (exponential prior on the standard deviation)
      'beta_linear_mask': np.array([False for i in range(d)]),
      'beta_quadratic_mask': np.array([False for i in range(d)])
      }
#================================================================== 
# Target function settings
TS ={'Tf': TF, # test function
    'dim': d, # number of dimensions
    'domain': domain} # domain bounds (use None for infinity)
#==================================================================
# acquisition function
AS = {'Af': 'logEI', # Acquisition function: EI, PI, LCB
    'Af_param': 2, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }
#==================================================================

Nmax = 5000
n_update = 250 # reestimate cov parameters every n_update iterations

optimizer = BO(TS,CS,AS,X_init,Y_init,Nmax)
#xgrid = BayOpt_SideKick.generate_grid(BayOpt_SideKick.reshape_domain_for_grid(optimizer.domain),100) # grid for plotting

#--------------------------------------------------
# Initial values
result = optimizer.update_GP_SGD(nr_iter=100,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=5) # updates the GP parameter estimates, does not need to be updated every iteration
optimizer.compute_Skk_iSY() # compute new cov matrix and cholesky factor
x_,res = optimizer.AF_optimization(10) # returns the next evaluation point


#m,v = optimizer.reconstruction_Skk(xgrid,optimizer.Skk,optimizer.iSY)
#Af_grid = optimizer.Af(xgrid,optimizer.Skk,optimizer.iSY)
y_ = TF(x_) # evaluate the new point

#---------------------------------------------------

optimizer.save_eval(x_,y_) # save the new evaluation
y_best = np.min(Y_init)
tolerance = 0.0001
R = np.zeros(Nmax)
r = 0

for k in range(1,Nmax):
    
    # BO iteration
    if k == 300:
        optimizer.Af_param = 0
        #optimizer.set_regression_structure(np.array([True for i in range(optimizer.d)]), np.array([True for i in range(optimizer.d)]))
        #result = optimizer.update_GP_SGD(nr_iter=30,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=5) # updates the GP parameter estimates, does not need to be updated every iteration
        #optimizer.compute_Skk_iSY() # compute new cov matrix and cholesky factor
        #r = 0

    if k == 900:
        optimizer.set_regression_structure(np.array([True for i in range(optimizer.d)]), np.array([True for i in range(optimizer.d)]))
        result = optimizer.update_GP_SGD(nr_iter=30,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=5) # updates the GP parameter estimates, does not need to be updated every iteration
        optimizer.compute_Skk_iSY() # compute new cov matrix and cholesky factor
        status2 =f"Iteration: {k},{result[2]} Adams steps, gradient norm {result[1]}, current minimum {y_best}, current minimum location {optimizer.x_best_val[0]}, current parameters {optimizer.param}, current beta {optimizer.beta}"
        #status_update(status1,filename)
        status_update(status2, filename)
        r = 0


    if (k%n_update == 0) or (k <= 300): # update hyperparameters every 100 iterations
        result = optimizer.update_GP_SGD(nr_iter=30,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=5) # updates the GP parameter estimates, does not need to be updated every iteration
        optimizer.compute_Skk_iSY() # compute new cov matrix and cholesky factor
        #status1 =f"Iteration: {k},{result[2]} Adams steps, gradient norm {result[1]}, current minimum {optimizer.y_best_val[0]}, current minimum location {optimizer.x_best_val[0]}"
        status2 =f"Iteration: {k},{result[2]} Adams steps, gradient norm {result[1]}, current minimum {y_best}, current minimum location {optimizer.x_best_val[0]}, current parameters {optimizer.param}, current beta {optimizer.beta}"
        #status_update(status1,filename)
        status_update(status2, filename)
        r = 0

    if (k%100 == 0) and (k%n_update != 0):
        optimizer.compute_Skk_iSY()

    
    if (k == 1000) or (k == 2000): #full restart at 1000 and compare with current result
        current_params = optimizer.param
        optimizer.add_param(optimizer.init_param)
        result2 = optimizer.update_GP_SGD(nr_iter=100,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=5)
        r = 0

        if result[-1] < result2[-1]:
            optimizer.add_param(current_params)
            
        optimizer.compute_Skk_iSY()
    


     

    x_,res= optimizer.AF_optimization(10) # returns the next evaluation point
    #ind = optimizer.ind
    #m,v = optimizer.reconstruction_Skk(xgrid,optimizer.Skk,optimizer.iSY[:ind])    # reconstruction on the grid
    #m = optimizer.sd*m.ravel()+optimizer.m
    #v = optimizer.sd*np.sqrt(v.ravel())
    #Af_grid = optimizer.Af(xgrid,optimizer.Skk[:ind,:ind],optimizer.iSY[:ind])  # acquisition on the grid
    x_new = BayOpt_SideKick.design_matrix(x_, optimizer.beta_quadratic_mask)
    mu,v=optimizer.reconstruction_Skk(x_new,optimizer.Skk[:optimizer.ind,:optimizer.ind],optimizer.iSr)
    y_ = TF(x_) # evaluate the new point
    y_std = (y_)/optimizer.sd
    r += (mu-y_std)**2/v
    R[k] = r


    if y_ < y_best:
        y_best = y_
    
    optimizer.save_eval(x_,y_) # save the new evaluation and update cov matrix, cholesky factor
    
    
    if y_best < tolerance:
        status_terminate = f"Loop terminated at interation{k} with tolerance {tolerance}"
        status_update(status_terminate, filename)
        break
    

#min_ind = df_loss['loss'].idxmin()
with open("Grid_BO_10dim_twin_exp_LR.txt", "w") as file:
    #file.write(f"Minimum grid {Loss[min_ind]} at x = [{acrotelm_por[min_ind]}, {Fgas[min_ind]}, {decay_length[min_ind]}]\n")
    file.write(f"real params RMOIST 0.130854227, CH4toCO2_peat 0.07529059359999998, oxid_frac 0.943677555, tiller_por 0.67723274, tiller_radius 0.008360541779999998, catotelm_por 0.9333307890000001, acrotelm_por 0.862018508, Fgas 0.00694513492, RMOIST_ANAEROBIC 0.0117290403, decay_length 10.2991547\n")
    file.write(f'Minimum estimate {y_best} at x = {optimizer.x_best_val[0]}')

np.save('y_evals_LR',optimizer.y_evals)
np.save('x_evals_LR',optimizer.x[:, :optimizer.d])
np.save('Chi_Square_resid_LR', R)

I = np.minimum.accumulate(optimizer.y_evals)
plt.plot(I)
plt.title('Improvement')
plt.xlabel('Number of iterations')
plt.axvline(x=len(X_init)-1, color='grey', linestyle='--', linewidth=2)
plt.savefig("Improvement_10dim_twin_exp_LR.png")
plt.close('all')

Value_recon = Plot_Value(optimizer.x_best_val[0])
years = np.linspace(2005, 2016, len(Value))
plt.plot(years, Value, label = "True Data")
plt.plot(years, Value_recon, label = "Reconstruction")
plt.title('True Values vs Reconstruction')
plt.xlabel('Year')
plt.ylabel('CH4')
plt.legend()
plt.savefig("Time_Series_10dim_twin_exp_LR.png")
plt.close('all')

residuals = np.abs(Value-Value_recon)
plt.plot(years, residuals)
plt.title('Residual Plot')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.savefig("Residuals_10dim_twin_exp_LR.png")
plt.close('all')

'''
def BO_loop_wrapper(TS,CS,AS,Xinit,Nmax,Testfunction,n_update):
    Y_init = Init_Points(d, X_init)
    opt = BO(TS,CS,AS,Xinit,Y_init,Nmax)
    
    result = opt.update_GP_SGD(nr_iter=50,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=20) # updates the GP parameter estimates, does not need to be updated every iteration
    opt.compute_Skk_iSY() # compute new cov matrix and cholesky factor
    x_,res = opt.AF_optimization(8) # returns the next evaluation point
    y_ = Testfunction(x_) # evaluate the new point
    opt.save_eval(x_,y_) # save the new evaluation

    for k in tqdm(range(1,Nmax)):
        
        # BO iteration
        if k%n_update == 0: # update hyperparameters every 5 iterations
            result = opt.update_GP_SGD(nr_iter=20,learn_rate=0.01,nr_stc_v_Hutch_tr=100,tolerance=1e-2, adams_patience=5) # updates the GP parameter estimates, does not need to be updated every iteration
            opt.compute_Skk_iSY() # compute new cov matrix and cholesky factor

        x_,res= opt.AF_optimization(8) # returns the next evaluation point
        y_ = Testfunction(x_) # evaluate the new point
        opt.save_eval(x_,y_) 
    I = np.minimum.accumulate(opt.y_evals)
    return opt, I

ASP = {'Af': 'PI', # Acquisition function: EI, PI, LCB
    'Af_param': 0, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }
ASE = {'Af': 'EI', # Acquisition function: EI, PI, LCB
    'Af_param': 0, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }
ASL = {'Af': 'LCB', # Acquisition function: EI, PI, LCB
    'Af_param': 2, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }
ASlogE = {'Af': 'logEI', # Acquisition function: EI, PI, LCB
    'Af_param': 0, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }
ASlogP = {'Af': 'logPI', # Acquisition function: EI, PI, LCB
    'Af_param': 0, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }
ASrand = {'Af': 'uniform', # Acquisition function: EI, PI, LCB
    'Af_param': 0, # Exploration parameter
    'optimizer':'L-BFGS-B' # Optimizer choice of acquisition 
    }

optE, IE = BO_loop_wrapper(TS,CS,ASE,X_init,Nmax= 1000,Testfunction= TF,n_update= 100)
optP, IP = BO_loop_wrapper(TS,CS,ASP,X_init,Nmax= 1000,Testfunction= TF,n_update= 100)
optlogE, IlogE = BO_loop_wrapper(TS,CS,ASlogE,X_init,Nmax= 1000,Testfunction= TF,n_update= 100)
optlogP, IlogP = BO_loop_wrapper(TS,CS,ASlogP,X_init,Nmax= 1000,Testfunction= TF,n_update= 100)
optL, IL = BO_loop_wrapper(TS,CS,ASL,X_init,Nmax= 1000,Testfunction= TF,n_update= 100)

plt.plot(IE,'-.',label='EI',color= 'blue')
plt.plot(IlogE,':',label='logEI',color='blue')

plt.plot(IP,'-.',label='PI',color='red')
plt.plot(IlogP,':',label='logPI',color='red')

plt.plot(IL,'-.',label='LCB',color= 'green')


#ax[0].plot(Improvement1_uniform,'-.',label='random',color='black')
plt.title(f' Bayesian Optimization')

plt.ylabel('Regret')
plt.xlabel('Iterations')
plt.axvline(x=len(X_init)-1, color='grey', linestyle='--', linewidth=2)
plt.legend()

plt.tight_layout()
plt.savefig('BO_opt_10dim.png')
'''
dt = time.time() - t1  # Execution time in seconds
dt_min = dt / 60  # Execution time in minutes
dt_hr = dt / 3600  # Execution time in hours

with open("BO_execution_time_10dim_twin_exp_LR.txt", "w") as file:
    file.write(f"Execution time: {dt:.6f} seconds\n")
    file.write(f"Execution time: {dt_min:.6f} minutes\n")
    file.write(f"Execution time: {dt_hr:.6f} hours\n")