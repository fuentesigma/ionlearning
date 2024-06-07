#!/usr/bin/env python

"""iontrap.py: quantum machinery for ion trap learning."""

__author__      = "J. Fuentes"
__copyright__   = "Copyright 2024, University of Luxembourg"

# Basic computing modules
import torch
import numpy as np
# Graphics and plots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Additional tools
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore")

# Module to synthesise plotting instructions
def plots(x, y, lw, color, fs, x_axis, y_axis, title=None):
    # Custom spine colour
    custom = '#999999'
    # Plot skeleton 
    plt.plot(x, y, lw=lw, color=color)
    # Plot ornaments
    plt.title(title, fontsize=fs, color=custom)
    plt.xlabel(x_axis, fontsize=fs, color=custom)
    plt.ylabel(y_axis, fontsize=fs, color=custom)
    plt.xticks(fontsize=fs, color=custom)
    plt.yticks(fontsize=fs, color=custom)
    # Customising the axes and frame
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # -----
    ax.spines['left'].set_color(custom)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_color(custom)
    ax.spines['bottom'].set_linewidth(lw)
    # Limit the number of ticks on each axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    # Axes ticks
    ax.tick_params(axis='x', colors=custom)
    ax.tick_params(axis='y', colors=custom)

# Fundamental class to compute quantum evolution on observables
class QuantumEvolution:
    def __init__(self, C):
        """
        Initialise the QuantumEvolution model with a single array of coefficients.
        :param C: Tensor of coefficients
        :param N: Number of pairs of coefficients
        """
        # Sanity checks: C must be a torch tensor
        if isinstance(C, torch.Tensor):
            self.C = C
        else:
            self.C = torch.tensor(C, dtype=torch.float64, requires_grad=True)
        # Period of the data
        self.period = 1
        self.epsilon = 1e-3
        self.betadelta = 1e-4
        # Plot's parameters
        self.lw = 2
        self.fs = 16

    def theta(self, t):
        # Ensure t is a torch tensor with requires_grad enabled
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        else:
            t_tensor = t.requires_grad_(True)
        # Initialise the series
        series = torch.zeros_like(t_tensor)
        # Calculate the series with n terms
        for i, c in enumerate(self.C):
            series += c * torch.sin((2*i + 1) * t_tensor)
        return series

    def beta(self, t):
        # Ensure t is a tensor with requires_grad=True for differentiation
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        else:
            t_tensor = t.requires_grad_(True)

        # Use theta(t) directly
        theta_t = self.theta(t_tensor)

        # Compute the first derivative of theta
        d_theta, = torch.autograd.grad(
            theta_t, t_tensor, grad_outputs=torch.ones_like(theta_t), create_graph=True
            )
        
        # Compute the second derivative of theta
        dd_theta, = torch.autograd.grad(
            d_theta, t_tensor, grad_outputs=torch.ones_like(d_theta), create_graph=True
            )
        
        # Identify positions where theta is zero
        zero_indices = torch.abs(theta_t) < self.betadelta

        # Modify u11 and u22 at indices where u12 is zero
        d_theta[zero_indices] = 1.0

        # Compute beta field
        num = -(2 * dd_theta * theta_t - d_theta**2 + 1)

        # Avoid division by very small numbers
        if torch.abs(theta_t) < self.epsilon:
            beta_t = torch.zeros_like(theta_t)
        else:
            beta_t = num / theta_t**2

        # Remove nan's from the elastic field
        beta_t[zero_indices] = 1e-3

        # Convert the result back to a NumPy array
        return beta_t.detach().numpy()
    
    def umatrix(self, t):
        # Ensure t is a torch tensor with requires_grad enabled
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        else:
            t_tensor = t.requires_grad_(True)

        # Prepare theta for differentiation
        theta_t = self.theta(t_tensor)
        d_theta = torch.autograd.grad(theta_t.sum(), t_tensor, create_graph=True)[0]

        # -------------------------------------------------
        # Compute matrix elements independently 
        u11 = d_theta
        u22 = u11.clone()
        u12 = theta_t

        # Identify positions where u12 is zero
        zero_indices = torch.abs(u12) < self.epsilon

        # Modify u11 and u22 at indices where u12 is zero
        u11[zero_indices] = 1.0
        u22[zero_indices] = 1.0

        # Compute u21
        u21 = (u11**2 - 1) / u12

        # Set u21 to zero at the same indices
        u21[zero_indices] = 0.0
        # -------------------------------------------------
        
        # Compose the evolution matrix u
        u = torch.stack([u11, u12, u21, u22])

        # Convert to numpy for compatibility with other operations
        return u.detach().numpy()
    
    def evolution(self, Q, t):
        # Ensure t is a torch tensor with requires_grad enabled
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        else:
            t_tensor = t.requires_grad_(True)

        # Retrieve evolution matrix
        u = self.umatrix(t_tensor)
        q_t = Q[0]*u[0,:] + Q[1]*u[1,:]
        p_t = Q[0]*u[2,:] + Q[1]*u[3,:]

        # Transform into torch tensors
        q_tensor = torch.tensor(q_t, dtype=torch.float64, requires_grad=True)
        p_tensor = torch.tensor(p_t, dtype=torch.float64, requires_grad=True)

        # Output
        return q_tensor, p_tensor
    
    def gaussianwavepacket(self, x, t, Q):
        # Ensure t is a torch tensor with requires_grad enabled
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        else:
            t_tensor = t.requires_grad_(True)

        # Retrieve evolution matrix
        u = self.umatrix(t_tensor)

        # Uncertainty shadow
        DeltaSquare = u[0,:]**2 + u[1,:]**2

        # Convert DeltaSquare to a PyTorch tensor
        DeltaSquare = torch.tensor(DeltaSquare, dtype=torch.float64)

        # Gaussian wave packet: square modulus
        return torch.tensor(
            (np.pi * (DeltaSquare + 1) )**(-1/2) * torch.exp(-(x - 0)**2/(DeltaSquare + 1)), 
            dtype=torch.float64,
            requires_grad=True)
    
    def plot_fields(self, t, peak_detection=False, pdf_name=None):
        # Convert t to a tensor if it's not already, without requiring gradients
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor(t, dtype=torch.float64)
        else:
            t_tensor = t

        # Compute theta and beta over the range of t
        theta_t = np.array([self.theta(T).detach().numpy() for T in t_tensor])
        beta_t = np.array([self.beta(T) for T in t_tensor])
        
        # Create a figure
        plt.figure(figsize=(12, 5))
        
        # Plot theta(t)
        plt.subplot(1, 2, 1)
        plots(t, theta_t, self.lw, '#F73B64', self.fs, r'time', r'$\theta(t)$')
        
        # Plot beta(t)
        plt.subplot(1, 2, 2)
        plots(t, beta_t, self.lw, '#F285D0', self.fs, r'time', r'$\beta(t)$')

        # Detect peaks
        if peak_detection:
            peaks, _ = find_peaks(beta_t)
            if len(peaks) > 1:
                # Find the height of the smallest peak
                smallest_height = np.min(beta_t[peaks])
                # Draw horizontal lines and annotate distances
                for i in range(1, len(peaks)):
                    # Distance between peaks
                    distance = t[peaks[i]] - t[peaks[i-1]]
                    # Draw horizontal line at the smallest peak's height
                    plt.hlines(smallest_height, t[peaks[i-1]], t[peaks[i]], color='#BBBBBB', linestyle='--', alpha=0.5)
                    # Annotate distance
                    mid_point_x = (t[peaks[i]] + t[peaks[i-1]]) / 2
                    plt.annotate(
                        f'{distance:.2f}', 
                        (mid_point_x, smallest_height), 
                        textcoords="offset points",
                        color='#BBBBBB',
                        xytext=(0,10),
                        ha='center',
                        size=self.fs)
                    
        # Adjust layout to prevent overlap and display plot
        plt.tight_layout()
        # Optional
        if pdf_name:
            plt.savefig(pdf_name, format='pdf', dpi=300)
        # Show plot
        plt.show()
    
    def plot_trajectories(self, Q, t, pdf_name=None):
        # Retrieve canonical variables
        q, p = self.evolution(Q, t)

        # Define figure environment
        plt.figure(figsize=(12, 5))

        # Plot left
        plt.subplot(1, 2, 1)
        plots(t, q.detach().numpy(), self.lw, '#8063EB', self.fs, r'time', r'position at time t')

        # Plot centre/right
        plt.subplot(1, 2, 2)
        plots(t, p.detach().numpy(), self.lw, '#F7C23B', self.fs, r'time', r'momentum at time t')
        
        # Adjust layout to prevent overlap and display plot
        plt.tight_layout()
        # Optional
        if pdf_name:
            plt.savefig(pdf_name, format='pdf', dpi=300)
        # Show plot
        plt.show()