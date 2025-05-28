import numpy as np
from scipy.linalg import cho_factor, solve_triangular
import torch
import torch.nn as nn
import torch.optim as optim
class ParameterModel(nn.Module):
    def __init__(self, init_params):
        super(ParameterModel, self).__init__()
        # We wrap the initial parameters in nn.Parameter so that PyTorch tracks them.
        # To use a custom initial point, simply pass in your desired tensor.
        self.params = nn.Parameter(init_params.clone())
    
    def forward(self):
        # The forward method returns the current parameter tensor.
        return self.params

    

class BayOpt_SideKick:

    def difference_matrices(X,Y):
        # Tested
        # compute the difference matrix
        # result is a three dim tensor in each dim : D[k]_ij = (Xk_i -Yk_j)^2 where Xk_i is the kth coordinate of the i row of X
        # if you sum D element wise along the first coordinate you get the distance matrix squared :: distance_matrix(X,Y) = sqrt(D)
        
        if X.ndim ==1:
            N= 1
            d = len(X)
            X = X[None,:] # TODO there is a better way
        else:
            N,d = X.shape

        Ny, d = Y.shape
        D = np.empty((d, N, Ny))
        for i in range(d):
            D[i] = (X[:, i][:, None] - Y[:, i][None, :])**2
        return D

    def difference_matrices_self(X):
        # Tested
        N, d = X.shape
        D = np.empty((d, N, N))
        for i in range(d):
            D[i] = (X[:, i][:, None] - X[:, i][None, :])**2
        return D

    def distance_anisotropic(X,Y,rho):

        if X.ndim ==1:
            N= 1
            d = len(X)
            X = X[None,:] 
        else:
            N,d = X.shape

        Ny, d = Y.shape
        D = np.zeros(( N, Ny))
        for i in range(d):
            D += ((X[:, i][:, None] - Y[:, i][None, :])/rho[i])**2
        return np.sqrt(D)




    def compute_diag_cholesky(S_uk, S_kk):  # tested
        
        """
        Compute the diagonal of S_uk * inv(S_kk) * S_uk^T using Cholesky decomposition.
        
        Parameters:
            S_uk (ndarray): An (m x n) matrix.
            S_kk (ndarray): An (n x n) positive definite matrix.
        
        Returns:
            diag_elements (ndarray): A length-m vector with the diagonal entries.
        """
        # Compute the Cholesky factorization: S_kk = L * L^T
        L, lower = cho_factor(S_kk, lower=True)
        
        # Solve L * Y = S_uk^T (forward substitution)
        Y = solve_triangular(L, S_uk.T, lower=True)
        
        # Solve L^T * X^T = Y (back substitution)
        X = solve_triangular(L.T, Y, lower=False).T  # X is S_uk * inv(S_kk)
        
        # Compute the diagonal elements as the row-wise dot product of X and S_uk
        diag_elements = np.sum(X * S_uk, axis=1)
        return diag_elements



    def generate_grid(domain, num_points=10):
        """
        Generate a grid of points for an arbitrary-dimensional domain.

        Parameters:
            domain (np.ndarray): A (2, d) array where the first row is the lower bounds
                                and the second row is the upper bounds for each dimension.
            num_points (int): Number of points to generate along each dimension.

        Returns:
            np.ndarray: An array of shape (num_points**d, d) where each row is a point in the grid.
        """
        # Number of dimensions
        d = domain.shape[1]
        
        # Create a list of 1D arrays for each dimension
        axes = [np.linspace(domain[0, i], domain[1, i], num_points) for i in range(d)]
        
        # Generate the full grid using meshgrid with 'ij' indexing for proper multidimensional ordering
        mesh = np.meshgrid(*axes, indexing='ij')
        
        # Stack and reshape the grid to a list of points (each row is a point in d-dimensional space)
        grid_points = np.stack(mesh, axis=-1).reshape(-1, d)
        return grid_points

    def reshape_domain_for_grid(bounds):


        bounds_array = np.array([[b[0] if b[0] is not None else np.NINF for b in bounds],  # Lower bounds
                        [b[1] if b[1] is not None else np.PINF for b in bounds]]) # Upper bounds
        return bounds_array

    def design_matrix(x, beta_q_mask):
        x= np.atleast_2d(x) # enforce 2D array 
        n,d= x.shape
        X = np.zeros((n,1 + d + sum(beta_q_mask)))
        X[:,:d] = x
        X[:,d] = 1
        X[:,d+1:] = x[:,beta_q_mask]**2
        return X
    
    def Hutchinsons_trace(R,gradient,m): # takes in Upper chol factor, gradient tensor, number of stochastic vectors
        # returns the hutchinson trac estimate of R^-1

        n = R.shape[0]
        V = np.random.choice([-1, 1], size=(n, m)) 
        W = solve_triangular(R,V,lower= False) # R^-1 V = W <=> solve for R W = V
        tr_est = np.zeros(gradient.shape[0])
        for i in range(gradient.shape[0]):
            tr_est[i] = np.sum(W*(gradient[i]@W))
        return tr_est/m

    
    def Adam_SGD(objective_func, num_iterations, init_params_np, learning_rate=0.01,arg = 100,tol = 1e-2,patience = 5):
        """
        Optimizes a given objective function using the Adam optimizer.
        
        Args:
            objective_func: A function that takes a torch.Tensor of parameters as input
                            and returns a tuple (loss, gradient), where 'loss' is a scalar
                            tensor and 'gradient' is a tensor of the same shape as the input.
            num_iterations: The number of optimization iterations to perform.
            init_params: A torch.Tensor containing the initial parameters.
            learning_rate: The learning rate for the Adam optimizer (default 0.01).
        
        Returns:
            final_params: A torch.Tensor containing the optimized parameters.
        """
        # Create an instance of the model holding our parameters.
        model = ParameterModel(torch.from_numpy(init_params_np))
        
        # Set up the Adam optimizer to update the model's parameters.
        adam_opt = optim.Adam(model.parameters(), lr=learning_rate)
        prev_loss = float('inf')
        best_value = float("inf")
        no_improvement_count = 0
        nr_steps = num_iterations
        # Run the optimization loop.
        for i in range(num_iterations):
            # Clear any previously accumulated gradients.
            adam_opt.zero_grad()
            
            # Compute the loss and gradient using your custom objective function.
            # model() calls the forward() method, which returns the parameter tensor.
            loss, grad = objective_func(model(),arg)
             # Manually assign the computed gradient to the model's parameter.
            # This tells PyTorch what gradient to use for the update.

            model.params.grad = grad



            if loss.item() < best_value - tol:
                best_value = loss.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        
            if no_improvement_count >= patience:
                nr_steps = i
                break
            if torch.norm(model.params.grad).item() < tol:
                nr_steps = i
                break

            prev_loss = loss.item()
           
            
            # Perform a single optimization step (update the parameters).
            adam_opt.step()
            

        
        # Return the optimized parameters (detached from the computation graph).
        return model.params.data.numpy(),torch.norm(model.params.grad).item(), nr_steps, loss.item()


