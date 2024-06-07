#!/usr/bin/env python

"""canonicalRegressor.py: quantum machine learning algorithm for regression with canonical variables."""

__author__      = "J. Fuentes"
__copyright__   = "Copyright 2024, University of Luxembourg"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iontrap import QuantumEvolution

from sklearn.metrics import r2_score
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

import warnings
warnings.filterwarnings("ignore")

class IonRegression:
    def __init__(self, features, target_function, initial_conditions, N=3, bounds=[-2,2]):
        # Declare the feature space (time domain)
        self.t = features
        # Declare target function (output variable)
        self.Y = target_function
        # Set initial conditions (observables)
        self.Q0 = initial_conditions
        # Number of coefficients
        self.N = N
        # These coefficients should be updated during the optimisation process
        self.qe = QuantumEvolution([0.0]*N)
        # Define bounds on the parameter scanning
        self.bounds = torch.tensor([[bounds[0]]*N, [bounds[1]]*N], dtype=torch.float64)

    def objective(self, C):
        # Update QuantumEvolution instance with new coefficients
        self.qe.C = C
        # -----------------------------------------------------------------
        # Perform the evolution based on initial conditions and time array
        q_t, _ = self.qe.evolution(self.Q0, self.t)
        # Compute the RMSE loss
        loss = torch.sqrt(torch.mean((q_t - self.Y) ** 2))
        # -----------------------------------------------------------------
        # Compute penalties given the constraints:
        # Call theta below as the coefficients are updated in this module
        theta_t = self.qe.theta(self.t)
        # Compute the penalty: not required if the ansatz is healthy as in the paper
        # penalty = self.constraints(theta_t)
        penalty = 0
        # -----------------------------------------------------------------
        # Add the penalty to the loss
        loss = loss + penalty
        # Return the negative loss for maximisation
        return -loss
    
    def constraints(self, theta):
        self.penalty_scale = 0.0
        # Calculate the derivative of theta
        dtheta_dt = torch.autograd.grad(theta.sum(), self.t, create_graph=True)[0]
        # Initialise penalty
        penalty = 0
        # Loop through theta and its derivative
        # Exclude the last value, diff is one element shorter
        for i, theta_val in enumerate(theta[:-1]):
            if theta_val != 0 and not (dtheta_dt[i] == 1 or dtheta_dt[i] == -1):
                # If the constraint is violated, add to the penalty
                penalty += self.penalty_scale * (1 - torch.abs(dtheta_dt[i]))**2
        return penalty

    def fit(self, num_iterations=50, num_init_guesses=10, early_stopping=20):
        # These initial conditions are for the parameters' initial guess
        train_X = torch.rand(num_init_guesses, self.N) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        train_Y = torch.tensor([self.objective(C) for C in train_X], dtype=torch.float64).unsqueeze(-1)

        # Model and likelihood
        gp_model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        
        # Variables for early stopping
        best_rmse = float('inf')
        best_iteration = 0

        # -----------------------------------------------------------------
        # Bayesian optimisation loop
        for iteration in range(num_iterations):
            # Perform model fitting
            fit_gpytorch_model(mll)

            # Acquisition function
            EI = ExpectedImprovement(model=gp_model, best_f=train_Y.max())
            
            # Optimise the acquisition function
            new_point, _ = optimize_acqf(
                acq_function=EI,
                bounds=self.bounds,
                q=1,
                num_restarts=5,
                raw_samples=50,
                return_best_only=True,
                sequential=True,
            )
            
            # Evaluate the objective function at the preprocessed new point
            new_obj = self.objective(new_point.squeeze()).unsqueeze(-1).detach()

            # Update the training points and outcomes
            train_X = torch.cat([train_X, new_point])
            
            # Ensure both tensors have the same number of dimensions
            if train_Y.dim() == 1:
                train_Y = train_Y.unsqueeze(-1)
            if new_obj.dim() == 1:
                new_obj = new_obj.unsqueeze(-1)
            
            # Now, concatenate the tensors
            train_Y = torch.cat([train_Y, new_obj], dim=0)
            
            # Re-initialise the model and likelihood with updated data
            gp_model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

            # Calculate RMSE and convert back to positive to get RMSE
            rmse = -new_obj.item()

            # Show verbosity
            if (iteration + 1) % 10 == 0:
                print(f"Iteration = {iteration + 1}/{num_iterations}, RMSE = {rmse:.4f}")

            # Early stopping check
            if rmse < best_rmse:
                best_rmse = rmse
                best_iteration = iteration
            elif iteration - best_iteration >= early_stopping:
                print(f"Early stopping at iteration {iteration + 1}, best RMSE = {best_rmse:.4f}")
                break
        # -----------------------------------------------------------------

        # Extract optimal parameters
        self.C_star = train_X[train_Y.argmax(), :]

    def get_train_test_data(self, t_test, Y_test):
        # Get optimal parameters
        self.qe.C = self.C_star
        # -----------------------------------------------------------------
        # Perform time evolution to get predictions for training set
        q_train, _ = self.qe.evolution(self.Q0, self.t)
        q_train = q_train.detach().numpy()
        # -----------------------------------------------------------------
        # Convert torch tensors into numpy
        t_train = self.t.detach().numpy()
        Y_train = self.Y.detach().numpy()
        # -----------------------------------------------------------------
        # Perform time evolution to get predictions for testing set
        q_test, _ = self.qe.evolution(self.Q0, t_test)
        q_test = q_test.detach().numpy()
        # -----------------------------------------------------------------
        # Convert torch tensors into numpy
        t_test = t_test.detach().numpy()
        Y_test = Y_test.detach().numpy()

        return t_train, q_train, Y_train, t_test, q_test, Y_test
    
    def plot_predictions(self, T_test, X_test, P=[-55/48, 5/96, -1/480]):

        # Get back trained data to contrast with unseen data
        t_train, x_train, X_train, t_test, x_test, X_test = self.get_train_test_data(T_test, X_test)

        # Introduce the exact parameters to plot the synthetic data
        plots = PlotRegression(Q0=self.Q0, P=P)

        # Get the plots
        plots.get_plots(t_train, x_train, X_train, t_test, x_test, X_test)

    
class PlotRegression:
    def __init__(self, Q0, P=None):
        self.P = P
        self.Q0 = Q0
        # //////////////////////
        # Plot's parameters
        self.lw = 1
        self.fs = 16
        self.figsize = (15, 10)
        self.custom = '#999999'

    def get_true_trajectories(self):
        # Predefined time interval for all experiments
        time = np.linspace(-np.pi, np.pi, 200)
        # Get true synthetic trajectories
        if self.P:
            self.qe = QuantumEvolution(self.P)
            x_t, p_t = self.qe.evolution(self.Q0, time)
        else:
            self.qe = QuantumEvolution(np.array([-49/48, -1/96, 1/96]))
            x_t, p_t = self.qe.evolution(self.Q0, time)
        return time, x_t, p_t

    def set_plot_aesthetics(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.custom)
        ax.spines['left'].set_linewidth(self.lw)
        ax.spines['bottom'].set_color(self.custom)
        ax.spines['bottom'].set_linewidth(self.lw)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(axis='x', colors=self.custom, labelsize=self.fs)
        ax.tick_params(axis='y', colors=self.custom, labelsize=self.fs)
    
    def plot_evaluations(self, feature, y_true, y_pred, label='Data'):
        plt.scatter(feature, y_true, s=20, marker='^', color='#B2E4ED', label=label)
        plt.scatter(feature, y_pred, s=20, alpha=0.75, color='#9292C2', label='Prediction')
        # Compute R2 score
        r2_train = r2_score(y_true, y_pred)
        # Add annotations and format
        plt.annotate(f'$R^2$ = {r2_train:.2f}', xy=(0.8, 0.95), 
                     xycoords='axes fraction', fontsize=self.fs, color=self.custom, verticalalignment='top')
        plt.xlabel(r'time', fontsize=self.fs, color=self.custom)
        plt.ylabel(r'trajectory at time t', fontsize=self.fs, color=self.custom)
        plt.xticks(fontsize=self.fs, color=self.custom)
        plt.yticks(fontsize=self.fs, color=self.custom)
        plt.legend(loc='upper center', fontsize=self.fs, 
                   labelcolor=self.custom, frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.15))

    # Module to synthesise plotting instructions
    def plot_observables(self, t, x, p, lw=4):
        # Plot skeleton 
        plt.plot(t, x, lw=lw, color='#B2E4ED', label=r'$\hat{x}$')
        plt.plot(t, p, lw=lw, color='#DBA9CE', label=r'$\hat{p}$')
        # Plot ornaments
        plt.xlabel(r'time', fontsize=self.fs, color=self.custom)
        plt.ylabel(r'canonical variables', fontsize=self.fs, color=self.custom)
        plt.xticks(fontsize=self.fs, color=self.custom)
        plt.yticks(fontsize=self.fs, color=self.custom)
        plt.legend(loc='upper center', fontsize=self.fs, 
                   labelcolor=self.custom, frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.15))

    def plot_phase(self, x, p, lw=4):
        # Plot skeleton 
        plt.plot(x, p, lw=lw, color='#BCE9D1')
        # Plot ornaments
        plt.xlabel(r'$\hat{x}$', fontsize=self.fs, color=self.custom)
        plt.ylabel(r'$\hat{p}$', fontsize=self.fs, color=self.custom)
        plt.xticks(fontsize=self.fs, color=self.custom)
        plt.yticks(fontsize=self.fs, color=self.custom)

    def get_plots(self, t_train, q_train, Y_train, t_test, q_test, Y_test, pdf_name=None):
        # Import plot routines for the canonical variables
        from iontrap import plots

        # Define figure environment
        plt.figure(figsize=self.figsize)

        # List of labels for the subplots
        labels = ['A', 'B', 'C', 'D']

        # Remove spines from plots
        for subplot_idx in range(1, 5):
            plt.subplot(2, 2, subplot_idx)
            ax = plt.gca()  # Get current axes
            # Apply custom settings to axes
            self.set_plot_aesthetics(ax)
            # Add label to the upper left corner
            ax.annotate(labels[subplot_idx - 1], xy=(-0.1, 1.1), xycoords='axes fraction', 
                        fontsize=self.fs, color=self.custom, weight='bold', ha='left', va='top')
        
        # /////////////////////////////////////////////////////
        # Remove these lines from the published code
        time, x_t, p_t = self.get_true_trajectories()

        # Upper left subplot: canonical variables
        plt.subplot(2, 2, 1)
        self.plot_observables(time, x_t.detach().numpy(), p_t.detach().numpy())

        # Upper right subplot: phase portrait
        plt.subplot(2, 2, 2)
        self.plot_phase(x_t.detach().numpy(), p_t.detach().numpy())
        # /////////////////////////////////////////////////////
        
        # Bottom left subplot: Training data and predictions
        plt.subplot(2, 2, 3)
        self.plot_evaluations(t_train, Y_train, q_train, 'Training Data')

        # Bottom right subplot: Testing data and predictions
        plt.subplot(2, 2, 4)
        self.plot_evaluations(t_test, Y_test, q_test, 'Testing Data')

        # Adjust layout to prevent overlap and display plot
        plt.tight_layout()

        from sklearn.metrics import r2_score, mean_squared_error

        # Calculate R2 scores
        r2_train = r2_score(Y_train, q_train)
        r2_test = r2_score(Y_test, q_test)

        # Calculate RMSE
        rmse_train = np.sqrt(mean_squared_error(Y_train, q_train))
        rmse_test = np.sqrt(mean_squared_error(Y_test, q_test))

        # Print the R2 and RMSE metrics
        print(f'Training R²: {r2_train:.4f}, Training RMSE: {rmse_train:.4f}')
        print(f'Testing R²: {r2_test:.4f}, Testing RMSE: {rmse_test:.4f}')

        # Optional
        if pdf_name:
            plt.savefig(pdf_name, format='pdf', dpi=300)
        # Show plot
        plt.show()

class DataHandler:
    def __init__(self, filename):
        self.filename = filename
        self.T, self.X, self.P = self.import_data()

    def import_data(self):
        # Read the data from CSV
        data = pd.read_csv(self.filename, header=None, names=['t', 'x', 'p'])

        # Convert to torch tensors
        T = torch.tensor(data['t'].values, dtype=torch.float64)
        X = torch.tensor(data['x'].values, dtype=torch.float64)
        P = torch.tensor(data['p'].values, dtype=torch.float64)

        # Add 10% noise to X and P
        noise_X = X * 0.1 * torch.randn_like(X)
        noise_P = P * 0.1 * torch.randn_like(P)
        X = X + noise_X
        P = P + noise_P

        return T, X, P

class TrainTestDataHandler(DataHandler):
    def __init__(self, filename, train_ratio=0.8):
        super().__init__(filename)
        self.train_ratio = train_ratio
        self.train_indices, self.test_indices = self.split_data()

    def split_data(self):
        num_train = int(len(self.T) * self.train_ratio)
        indices = torch.randperm(len(self.T))
        return indices[:num_train], indices[num_train:]

    def get_train_data(self):
        return self.T[self.train_indices], self.X[self.train_indices], self.P[self.train_indices]

    def get_test_data(self):
        return self.T[self.test_indices], self.X[self.test_indices], self.P[self.test_indices]