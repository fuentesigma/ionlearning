# Quantum Machine Learning With Canonical Variables

This repository contains the code that accompanies the paper "Quantum Machine Learning With Canonical Variables" by J. Fuentes.

## Libraries Required

To run the code, the following libraries are required:

### Basic Computing Modules
- `torch`
- `numpy`

### Graphics and Plots
- `matplotlib`

### Additional Tools
- `scipy`

### Data Handling and Visualisation
- `pandas`
- `seaborn`

### Machine Learning and Optimisation
- `scikit-learn`
- `botorch`
- `gpytorch`

## Installation

Make sure you have Python 3.8 installed. You can install the required dependencies using `pip`. It is recommended to create a virtual environment to manage the dependencies:

```bash
# Create a virtual environment
python3.8 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install torch numpy pandas seaborn matplotlib scipy scikit-learn botorch gpytorch
