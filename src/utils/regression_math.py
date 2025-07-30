from utils.logger import set_up_logger
import numpy as np
from typing import Tuple
logger = set_up_logger()

def calculate_coefs(X: np.ndarray, Y: np.ndarray, strict: bool) -> Tuple[float, np.ndarray]:
    """
    Computes both the bias and coefficients using QR decomposition on an
    augmented feature matrix [1 | X].

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        Y (np.ndarray): Target vector of shape (n_samples,) or (n_samples, 1)
        strict (bool): Whether to raise on validation failure

    Returns:
        Tuple[float, np.ndarray]: (bias, coefficients)
    """
    # MGS calculation and validation
    n = X.shape[0]
    X_aug = np.hstack((np.ones((n, 1)), X))  # shape: (n_samples, n_features + 1)
    Q, R = MGS(X_aug)
    if validate_MGS(Q, R, X_aug, strict):
        logger.info("Successfully validated MGS")
    else:
        logger.warning("Falling back to NumPy QR decomposition via Householder.")
        Q, R = np.linalg.qr(X_aug) # Fall back to Householder, usually occurs when vectors are nearly colinear
    
    # Implementing backwards substitution to calculate w
    Qt_y = Q.T @ Y
    w_full = backward_substitution(R, Qt_y)
    
    b = w_full[0]
    w = w_full[1:]

    return b, w
    

def MGS(X: np.ndarray):
    """Modified Gram-Schmidt method for QR decomposition.

    Args:
        X (np.ndarray): A matrix shape (n_samples, n_features)
    
    Returns:
        ndarray: Q An orthonormal matrix of shape (n_samples, n_features), where the columns form an orthonomral basis for X
        ndarray: R An upper triangular matrix (n_features, n_features) such that X is approximately Q @ R
    """
    # Getting the dimensions
    n, p = X.shape
    
    # Setting up each array
    Q = np.zeros((n, p))
    R = np.zeros((p, p))
    
    for j in range(p):
        v = X[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

def validate_MGS(Q: np.ndarray, R: np.ndarray, X: np.ndarray, strict: bool):
    """
Validates numerical stability of the Modified Gram-Schmidt decomposition.

Checks that:
- Q is orthonormal (Qᵀ Q ≈ I)
- R is upper triangular (R ≈ triu(R))
- Q @ R is approximately equal to X

Logs warnings or raises errors depending on the `strict` flag.

Args:
    Q (ndarray): Orthonormal matrix (n_samples, n_features)
    R (ndarray): Upper triangular matrix (n_features, n_features)
    X (ndarray): Original matrix (n_samples, n_features)
    strict (bool): Whether to raise on failure or log a warning

Returns:
    bool: True if all validations pass
"""

    # Determine orthonormality (Is I approximately Q.T @ Q)
    I = np.eye(Q.shape[1])
    is_orthonormal = np.allclose(Q.T @ Q, I)
    if not is_orthonormal:
        msg = "Q is not orthonormal"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    # Determine if Q @ R is approximately X
    is_accurate = np.allclose(Q @ R, X, atol=1e-8)
    if not is_accurate:
        msg = "Q @ R is not close to X"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    is_triangular = np.allclose(R, np.triu(R))
    if not is_triangular:
        msg = "R is not triangular"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    return is_accurate and is_orthonormal and is_triangular

def backward_substitution(R: np.ndarray, b: np.ndarray):
    """Implements backward substitution to calculate a vector (w) such that RV = b

    Args:
        R (ndarray): A upper triangular matrix (p x p)
        b (ndarray): A vector (p)

    Returns:
        ndarray: w (p) such that R @ w = b
    """
    p = R.shape[0]
    w = np.zeros_like(b, dtype=float)

    for i in reversed(range(p)):
        # b[i] - sum of known terms to the right
        sum_terms = np.dot(R[i, i+1:], w[i+1:])
        w[i] = (b[i] - sum_terms) / R[i, i]

    return w

