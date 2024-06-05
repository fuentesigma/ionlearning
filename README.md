# Quantum Machine Learning With Canonical Variables

This repository contains the code that accompanies the paper _"Quantum Machine Learning With Canonical Variables"_ by <span style="color:#6a9fb5;">J. Fuentes</span>.

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

## Files Included

- **iontrap.py**: This file simulates the equations of motion within an ion trap.
- **canonicalClassifier.py**: This file implements a binary regressor for the canonical variables.
- **canonicalRegressor.py**: This file implements a regressor for the canonical variables.
- **controlPanel.ipynb**: This Jupyter notebook helps to reproduce all the examples in the paper and demonstrates how the framework can be implemented for simulations.

## Data

The `data` folder contains datasets used for training the models. Ensure that this folder is present in the same directory as the code files.

## Installation

Make sure you have Python 3.8 installed. You can install the required dependencies using `pip`. It is recommended to create a virtual environment to manage the dependencies:

```bash
# Create a virtual environment
python3.8 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install torch numpy pandas seaborn matplotlib scipy scikit-learn botorch gpytorch

```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/fuentesigma/ionlearning.git
    cd ionlearning
    ```

2. Run the Jupyter notebook:
    ```bash
    jupyter notebook controlPanel.ipynb
    ```

Follow the instructions in `controlPanel.ipynb` to reproduce the experiments and simulations from the paper.
