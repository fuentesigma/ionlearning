#!/usr/bin/env python

"""canonicalClassifier.py: quantum machine learning algorithm for binary classification with canonical variables."""

__author__      = "J. Fuentes"
__copyright__   = "Copyright 2024, University of Luxembourg"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iontrap import QuantumEvolution

from sklearn.metrics import roc_curve, auc, confusion_matrix

import torch
import torch.nn.functional as F

from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

import warnings
warnings.filterwarnings("ignore")

class IonClassifier:
    def __init__(self, features, target_function, initial_conditions, N=3, bounds=[-2,1]):
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
        # -----------------------------------------------------------------
        # Add placeholders for mu and sigma of q_t
        self.mu_qt = None
        self.sigma_qt = None

    def mu_sigma(self, q_t_samples):
        """Calculate the mean and standard deviation of q_t based on provided samples"""
        self.mu_qt = torch.mean(q_t_samples)
        self.sigma_qt = torch.std(q_t_samples)

    def objective(self, C):
        # Update QuantumEvolution instance with new coefficients
        self.qe.C = C
        # -----------------------------------------------------------------
        # Processing of q_t
        # Perform time evolution based on initial conditions and time array
        q_t, _ = self.qe.evolution(self.Q0, self.t)
        # Compute the meand and std dev of q_t
        self.mu_sigma(q_t)
        # Ensure mu_qt and sigma_qt are set
        if self.mu_qt is None or self.sigma_qt is None:
            raise ValueError("mu_qt and sigma_qt need to be set before calling objective.")
        # Normalise q_t
        q_t_norm = (q_t - self.mu_qt) / self.sigma_qt
        # Apply sigmoid to map q_t to probabilities
        # q_t_probs = torch.sigmoid(q_t_norm)
        q_t_probs = torch.sigmoid(q_t)
        # -----------------------------------------------------------------
        # Copy the target values from the external target function
        Y_t = self.Y
        # -----------------------------------------------------------------
        # Compute the BCE loss
        q_t_probs = q_t_probs.float()   # Convert model outputs to Float
        Y_t = Y_t.float()               # Convert target tensor to Float
        if not torch.all((0 <= q_t_probs) & (q_t_probs <= 1)):
            raise ValueError("q_t_probs contains values outside the [0, 1] range.")
        loss = F.binary_cross_entropy(q_t_probs, Y_t, reduction='mean')
        # -----------------------------------------------------------------
        # Compute penalties given the constraints:
        # This is not necessary if the ansatz is exact
        # theta_t = self.qe.theta(self.t)
        # Compute the penalty or fix it at zero otherwise
        penalty = 0
        # -----------------------------------------------------------------
        # Add the penalty to the loss
        loss += penalty
        # Return the negative loss for maximisation (adapt as needed for your optimisation setup)
        return -loss
    
    def constraints(self, theta):
        self.penalty_scale = 1e-2
        # Calculate the derivative of theta
        dtheta_dt = torch.autograd.grad(theta.sum(), self.t, create_graph=True)[0]
        # Initialise penalty
        penalty = 0
        # Loop through theta and its derivative
        for i, theta_val in enumerate(theta[:-1]):  # Exclude the last value because diff is one element shorter
            if theta_val != 0 and not (dtheta_dt[i] == 1 or dtheta_dt[i] == -1):
                # If the constraint is violated, add to the penalty
                penalty += self.penalty_scale * (1 - torch.abs(dtheta_dt[i]))**2
        return penalty

    def fit(self, num_iterations=50, num_init_guesses=10, early_stopping=20):
        # These initial conditions are for the parameters' initial guesses
        train_X = torch.rand(num_init_guesses, self.N) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        train_Y = torch.tensor([self.objective(C) for C in train_X], dtype=torch.float64).unsqueeze(-1)

        # Model and likelihood
        gp_model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        # Variables for early stopping
        best_score = float('inf')
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
            score = -new_obj.item()

            # Show verbosity
            if (iteration + 1) % 10 == 0:
                print(f"Iteration = {iteration + 1}/{num_iterations}, Score = {score:.4f}")

            # Early stopping check
            if score < best_score:
                best_score = score
                best_iteration = iteration
            elif iteration - best_iteration >= early_stopping:
                print(f"Early stopping at iteration {iteration + 1}, best Score = {best_score:.4f}")
                break
        # -----------------------------------------------------------------

        # Extract optimal parameters
        self.C_star = train_X[train_Y.argmax(), :]

    def plot_predictions(self, t_test, X_test, X_test_labels, X_train, pdf_name=None):
        # Plot options
        self.fs = 16
        self.custom = '#999999'
        self.figsize = (15, 10)
        # Get optimal parameters
        self.qe.C = self.C_star

        # -----------------------------------------------------------------
        # Perform time evolution to get predictions for training set
        q_train, _ = self.qe.evolution(self.Q0, self.t)
        q_train_probs = torch.sigmoid(q_train).detach().numpy() # predictions
        # -----------------------------------------------------------------
        # Convert torch tensors into numpy
        t_train = self.t.detach().numpy()       # time domain
        Y_train = self.Y.detach().numpy()       # target labels
        X_train = X_train.detach().numpy()      # target observable (unlabelled)
        # # Get labels for c0 and c1
        c0, c1 = Y_train == 0, Y_train == 1
        # Call plots
        # Define figure environment
        plt.figure(figsize=self.figsize)
        # List of labels for the subplots
        labels = ['A', 'B', 'C', 'D']
        # Remove spines from plots
        for subplot_idx in range(1, 5):
            plt.subplot(2, 2, subplot_idx)
            ax = plt.gca()  # Get current axes
            # Add label to the upper left corner
            ax.annotate(labels[subplot_idx - 1], xy=(-0.1, 1.2), xycoords='axes fraction', 
                        fontsize=self.fs, color=self.custom, weight='bold', ha='left', va='top')

        plt.subplot(2, 2, 1)
        self._plot_data(t_train, X_train, c0, c1, self.fs, self.custom)

        plt.subplot(2, 2, 2)
        self._plot_predicted_probabilities(q_train_probs, Y_train, self.fs, self.custom)
        
        plt.subplot(2, 2, 3)
        self._plot_roc_curve(Y_train, q_train_probs, self.fs, self.custom)

        plt.subplot(2, 2, 4)
        self._plot_confusion_matrix(Y_train, q_train_probs, self.fs, self.custom)

        # Save file as required
        if pdf_name:
            plt.savefig(pdf_name + '_train.pdf', format='pdf', dpi=300)
        plt.show()

        # -----------------------------------------------------------------
        # Perform time evolution to get predictions for testing set
        q_test, _ = self.qe.evolution(self.Q0, t_test)
        q_test_probs = torch.sigmoid(q_test).detach().numpy() # predictions
        # -----------------------------------------------------------------
        # Convert torch tensors into numpy
        t_test = t_test.detach().numpy()           # time domain
        Y_test = X_test_labels.detach().numpy()    # target labels
        X_test = X_test.detach().numpy()           # target observable (unlabelled)
        # # Get labels for c0 and c1
        c0, c1 = Y_test == 0, Y_test == 1
        # Call plots
        # Define figure environment
        plt.figure(figsize=self.figsize)
        # List of labels for the subplots
        labels = ['A', 'B', 'C', 'D']
        # Remove spines from plots
        for subplot_idx in range(1, 5):
            plt.subplot(2, 2, subplot_idx)
            ax = plt.gca()  # Get current axes
            # Add label to the upper left corner
            ax.annotate(labels[subplot_idx - 1], xy=(-0.1, 1.2), xycoords='axes fraction', 
                        fontsize=self.fs, color=self.custom, weight='bold', ha='left', va='top')

        plt.subplot(2, 2, 1)
        self._plot_data(t_test, X_test, c0, c1, self.fs, self.custom)

        plt.subplot(2, 2, 2)
        self._plot_predicted_probabilities(q_test_probs, Y_test, self.fs, self.custom)

        plt.subplot(2, 2, 3)
        self._plot_roc_curve(Y_test, q_test_probs, self.fs, self.custom)

        plt.subplot(2, 2, 4)
        self._plot_confusion_matrix(Y_test, q_test_probs, self.fs, self.custom)
        # Save file as required
        if pdf_name:
            plt.savefig(pdf_name + '_test.pdf', format='pdf', dpi=300)
        plt.show()

    def _plot_data(self, tau, target, c0, c1, fs, custom):
        plt.scatter(tau[c0], target[c0], color='#B2E4ED', alpha=0.8, label='Class 0')
        plt.scatter(tau[c1], target[c1], color='#DBA9CE', alpha=0.8, label='Class 1')
        plt.xlabel('Time', fontsize=fs, color=custom)
        plt.ylabel('Data Points', fontsize=fs, color=custom)
        self._set_plot_style(fs, custom)
        self._add_legend('upper center', fs, custom, ncol=2, bbox_to_anchor=(0.5, 1.3))

    def _plot_predicted_probabilities(self, q_t_probs, Y_t, fs, custom):
        plt.hist(q_t_probs[Y_t == 0], bins=50, color='#B2E4ED', alpha=0.8, label='Class 0')
        plt.hist(q_t_probs[Y_t == 1], bins=50, color='#DBA9CE', alpha=0.8, label='Class 1')
        plt.xlabel('Predicted probability', fontsize=fs, color=custom)
        plt.ylabel('Frequency', fontsize=fs, color=custom)
        self._set_plot_style(fs, custom)
        self._add_legend('upper center', fs, custom, ncol=2, bbox_to_anchor=(0.5, 1.3))

    def _plot_roc_curve(self, Y_t, q_t_probs, fs, custom):
        fpr, tpr, _ = roc_curve(Y_t, q_t_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='#BCE9D1', lw=4, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='#F58CD0', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=fs, color=custom)
        plt.ylabel('True Positive Rate', fontsize=fs, color=custom)
        self._set_plot_style(fs, custom)
        self._add_legend('lower right', fs, custom)

    def _plot_confusion_matrix(self, Y_t, q_t_probs, fs, custom):
        Y_pred = (q_t_probs > 0.5).astype(int)
        cm = confusion_matrix(Y_t, Y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Pastel2', cbar=False, annot_kws={"size": fs, "color": custom})
        plt.xlabel('Predicted Class', fontsize=fs, color=custom)
        plt.ylabel('True Class', fontsize=fs, color=custom)
        plt.xticks([0.5, 1.5], ['Class 0', 'Class 1'], fontsize=fs, color=custom)
        plt.yticks([0.5, 1.5], ['Class 0', 'Class 1'], fontsize=fs, color=custom, va='center')
        plt.tight_layout()

    def _set_plot_style(self, fs, custom):
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(custom)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_color(custom)
        ax.spines['bottom'].set_linewidth(1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(axis='x', colors=custom, labelsize=fs)
        ax.tick_params(axis='y', colors=custom, labelsize=fs)

    def _add_legend(self, loc, fs, custom, **kwargs):
        l = plt.legend(loc=loc, fontsize=fs, frameon=False, **kwargs)
        for text in l.get_texts():
            text.set_color(custom)

class DataHandler:
    def __init__(self, filename):
        self.filename = filename
        self.T, self.X, self.P, self.labels_X, self.labels_P = self.import_data()

    def import_data(self):
        # Read the data from CSV
        data = pd.read_csv(self.filename, header=None, names=['t', 'x', 'p'])

        # Convert to torch tensors
        T = torch.tensor(data['t'].values, dtype=torch.float64)
        X = torch.tensor(data['x'].values, dtype=torch.float64)
        P = torch.tensor(data['p'].values, dtype=torch.float64)

        # Add 10% noise to X and P
        noise_X = 0.3 * torch.randn_like(X)
        noise_P = 0.3 * torch.randn_like(P)
        X = X + noise_X
        P = P + noise_P

        # Calculate medians
        median_X = torch.median(X)
        median_P = torch.median(P)

        # Label data based on median
        labels_X = (X > median_X).to(torch.int64)
        labels_P = (P > median_P).to(torch.int64)

        return T, X, P, labels_X, labels_P

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
        return (self.T[self.train_indices], 
                self.X[self.train_indices], 
                self.P[self.train_indices], 
                self.labels_X[self.train_indices], 
                self.labels_P[self.train_indices])

    def get_test_data(self):
        return (self.T[self.test_indices], 
                self.X[self.test_indices], 
                self.P[self.test_indices], 
                self.labels_X[self.test_indices], 
                self.labels_P[self.test_indices])
