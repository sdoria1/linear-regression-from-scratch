import numpy as np
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.utils.logger import set_up_logger
from src.linear_regression import LinearRegression
import matplotlib.pyplot as plt
plt.set_loglevel("critical")
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = set_up_logger()

def load_csv_data(path: str, delimiter: str = ',') -> tuple[np.ndarray, np.ndarray]:
    """Loads a CSV and returns X, Y as NumPy arrays. Assumes last column is target.
    Rows with missing values are dropped.

    Args:
        path (str): Path to the CSV file
        delimiter (str): CSV delimiter

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and target vector Y
    """
    raw_data = np.genfromtxt(path, delimiter=delimiter, skip_header=1)
    # Remove rows with NaNs
    clean_data = raw_data[~np.isnan(raw_data).any(axis=1)]
    X = clean_data[:, :-1]
    Y = clean_data[:, -1]
    logger.info(f"Original rows: {raw_data.shape[0]}, Cleaned rows: {clean_data.shape[0]}")
    return X, Y


def plot_predictions(Y_true, Y_pred):
    """Plots predictions vs true Y values

    Args:
        Y_true (ndarray): A (n_samples, 1) array of true values
        Y_pred (_type_): A (n_samples, 1) array of predicted values
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_true, Y_pred, alpha=0.6)
    plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--')  # identity line
    plt.xlabel("Actual Median Home Value")
    plt.ylabel("Predicted Median Home Value")
    plt.title("Predicted Median Home Value vs Actual Median Home Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def closed_form(X: np.ndarray, Y: np.ndarray):
    """Runs a closed form regression

    Args:
        X (np.ndarray): The feature array
        Y (np.ndarray): The target vector
    Returns:
        float: The MSE of the method
        ndarray: The weights
        float: The bias
    """
    closed_form_model = LinearRegression()
    w, b = closed_form_model.closed_form_fit(X, Y, strict=True)
    
    # Predict and evaluate
    Y_pred_closed = closed_form_model.predict(X)
    mse_closed = closed_form_model.mse(Y, Y_pred_closed)
    normalized_mse_closed = mse_closed / np.mean(Y**2)
    logger.info(f"MSE Closed Form Model: {normalized_mse_closed}")
    logger.info(f"Weights: {w}")
    logger.info(f"Bias: {b}")
    logger.info("Plotting...")
    plot_predictions(Y, Y_pred_closed)
    return mse_closed, w, b
    
def grad_descent(X: np.ndarray, Y: np.ndarray, MSE_cf = None):
    """Runs a closed form regression

    Args:
        X (np.ndarray): The feature array
        Y (np.ndarray): The target vector
        MSE_cf (float): The loss calculated in MSE, for plotting against loss history
    Returns:
        float: The MSE of the method
        ndarray: The weights
        float: The bias
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = X_normalized = (X - X_mean) / X_std
    gradient_descent_model = LinearRegression()
    w, b = gradient_descent_model.gradient_descent_fit(X_normalized, Y, alpha_0=0.1, decay_rate = 0.01, batch_size = None, epochs=500, verbose=True)
    w = w / X_std
    b = b - np.sum((X_mean / X_std) * w)

    # Predict and evaluate
    Y_pred_grad = gradient_descent_model.predict(X_normalized)
    mse_grad = gradient_descent_model.mse(Y, Y_pred_grad)
    normalized_mse_grad = mse_grad / np.mean(Y**2)
    logger.info(f"Normalized MSE Gradient Descent Model: {normalized_mse_grad}")
    logger.info(f"Weights: {w}")
    logger.info(f"Bias: {b}")
    logger.info("Plotting...")
    plot_predictions(Y, Y_pred_grad)
    plot_loss_history(gradient_descent_model.loss_history, MSE_cf)
    return mse_grad, w, b
    
def plot_loss_history(loss_history, cf_mse):
    print(len(loss_history))
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Gradient Descent MSE", linewidth=2)
    if cf_mse is not None:
        plt.axhline(cf_mse, color="red", linestyle="--", label="Closed Form MSE")
    
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def main():
    
    # Path to your CSV file
    data_path = os.path.join("data", "HousingData.csv")
    
    # Load and prepare data
    X, Y = load_csv_data(data_path)
    # Initialize and fit model
    MSE_closed, _, _ = closed_form(X, Y)
    grad_descent(X, Y, MSE_closed)

    
if __name__ == "__main__":
    main()