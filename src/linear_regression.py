from utils.logger import set_up_logger
import warnings
from utils.regression_math import calculate_coefs
import numpy as np
from typing import Optional

logger = set_up_logger()

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.bias = None
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_w = None
        self.best_b = None
        self.best_epoch = None
    def closed_form_fit(self, X: np.ndarray, Y: np.ndarray, strict: bool):
        """
        Fits the linear regression model using a closed-form QR-based solution.
        Struggles with features that are highly co-linear

        Args:
            X (np.ndarray): Training feature matrix (n_samples, n_features).
            Y (np.ndarray): Training target vector (n_samples,).
            strict (bool): Whether to raise or warn on numerical validation failure.

        Returns:
            Tuple[np.ndarray, float]: Coefficients and bias term.
        """
        # Validate the shape of the arrays
        try:
            self.validate_inputs(X, Y, True)
        except ValueError as e:
            logger.error(f"Validation failed: {e}")
            raise 
        logger.info("Inputs validated successfully.")
        
        # Apply formula to calculate coefficients
        b, w = calculate_coefs(X, Y, strict)
        
        # Update model
        self.coefficients = w
        self.bias = b
        
        # Return fit to the user
        return w, b
    
    def gradient_descent_fit(self, X: np.ndarray, Y: np.ndarray,
        alpha_0: float = 0.01,
        decay_rate: float = 0,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        verbose: bool = False) -> None:
        """_summary_

        Args:
            X (np.ndarray): The training array
            Y (np.ndarray): The target array
            alpha_0 (float, optional): the learning rate. Defaults to 0.01.
            decay_rate (float, optional): the rate of learning decay, should add numerical stability. Defaults to 0 (no decay)
            epochs (int, optional): number of learning epochs. Defaults to 1000.
            batch_size (Optional[int], optional): The batch size for training. Defaults to None.
            verbose (bool, optional): Reports epochs if true. Defaults to False.
        """
        n_samples, n_features = X.shape
        batch_size = batch_size or n_samples # default to full batch
        
        # Naive initial guess
        self.coefficients = np.zeros(n_features)
        self.bias = 0.0
        for epoch in range(epochs):
            # Shuffle dataset
            indicies = np.random.permutation(n_samples)
            X_shuffled = X[indicies]
            Y_shuffled = Y[indicies]
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                
                dw, db, y_pred = self.compute_gradients(X_batch, Y_batch)
                
                #Updating
                alpha = alpha_0 / (1 + decay_rate * epoch)
                self.coefficients -= alpha * dw
                self.bias -= alpha * db
            loss = self.mse(Y_batch, y_pred)
            if loss < self.best_loss:
                self.best_w = self.coefficients
                self.best_b = self.bias
                self.best_epoch = epoch
                self.loss_history.append(loss)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:4d} | MSE: {loss:.4f}")
        self.coefficients = self.best_w
        self.bias = self.best_b
        
        return self.coefficients, self.bias

    def compute_gradients(self, X: np.ndarray, Y: np.ndarray):
        """Computes dw and db for the training epoch

        Args:
            X (np.ndarray): Training epoch training set
            Y (np.ndarray): Training epoch target set

        Returns:
            _type_: the change in w and b based on the epoch
        """
        y_pred = self.predict(X)
        residuals = y_pred - Y
        n = len(X)
        dw = (X.T @ residuals / n)
        db = (np.sum(residuals) / n)
        
        return dw, db, y_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values using the learned coefficients and bias.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
        """
        if self.coefficients is None or self.bias is None:
            raise ValueError("Model has not been fit yet.")

        return X @ self.coefficients + self.bias
    def mse(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        """Computes mean squared error between true and predicted values.

        Args:
            Y (np.ndarray): True target values, shape (n_samples,)
            Y_pred (np.ndarray): Predicted target values, shape (n_samples,)

        Returns:
            float: Mean squared error
        """
        return np.mean((Y - Y_pred) ** 2)

    def validate_inputs(self, X: np.ndarray, Y: np.ndarray, closed_form: bool):
        """Validates the shape of the inputs and validates whether or not training will be possible.

        Args:
            X (np.ndarray): The training feature set
            Y (np.ndarray): The training target set
            closed_form (bool): Whether or not we are solving using a closed form solution
        
        Return:
            True if valid inputs are given
        """
        # Get relevant dimensions
        X_rows = X.shape[0]
        Y_rows = Y.shape[0]
        X_columns = X.shape[1]
        # Check that the number of rows in all arrays are equal
        if not (X_rows == Y_rows):
            raise ValueError(f"Mismatch in the number of rows across train & test sets.\nX train: {X_rows}\nY train: {Y_rows}")
        if X_columns > 1000 and closed_form: # O(np^2) where p = number of features. Warn user if this is computationally complex
            logger.warning(f"Large feature set may cause matrix inversion to be slow or unstable. Feature count: {X_columns}")
            warnings.warn(
            f"Large feature set may cause matrix inversion to be slow or unstable. Feature count: {X_columns}",
            RuntimeWarning
        )
        return True

"""
# Seed for reproducibility
np.random.seed(42)

# Generate 100 samples with 2 features
X = np.random.rand(100, 2)

# True weights: w = [3, 5], bias = 7
true_w = np.array([3, 5])
true_b = 7

# Generate Y with a bit of noise
Y = X @ true_w + true_b + np.random.randn(100) * 0.1  # small noise

model = LinearRegression()
model.gradient_descent_fit(X, Y, alpha_0=0.1, decay_rate = 0.01, epochs=500, batch_size=25, verbose=True)

print("Estimated coefficients:", model.coefficients)
print("Estimated bias:", model.bias)
"""