import numpy as np
import sys
import os
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from linear_regression import LinearRegression  # adjust import path as needed

def test_closed_form():
    X = np.array([
        [1, 0, 2],
        [2, 1, 1],
        [0, 3, 3],
        [4, 2, 0],
        [1, 1, 1]
    ])
    Y = 2 * X[:, 0] - 1.5 * X[:, 1] + 4 * X[:, 2] + 3  # true model

    model = LinearRegression()
    w, b = model.closed_form_fit(X, Y, strict=True)

    assert np.allclose(w, [2.0, -1.5, 4.0], atol=1e-8), f"w incorrect: {w}"
    assert np.isclose(b, 3.0, atol=1e-8), f"b incorrect: {b}"
    print("✅ Example 3 passed.")
    
def test_validate_inputs():
    model = LinearRegression()

    # ✅ Valid input
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([5, 6])
    assert model.validate_inputs(X, Y, closed_form=True) is True
    print("✅ Valid input passed")

    # ❌ Mismatched number of rows
    Y_bad = np.array([5])
    try:
        model.validate_inputs(X, Y_bad, closed_form=True)
        print("❌ Failed to raise on row mismatch")
    except ValueError as e:
        print(f"✅ Caught mismatch error: {e}")

    # ⚠️ Large number of features should log a warning (not raise)
    X_wide = np.ones((10, 1500))
    Y_wide = np.ones(10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.validate_inputs(X_wide, Y_wide, closed_form=True)
        print("✅ Large feature warning triggered" if w else "❌ No warning for large features")

    # ❌ X is not 2D (shape[1] will raise)
    X_flat = np.array([1, 2, 3])
    Y_flat = np.array([1, 2, 3])
    try:
        model.validate_inputs(X_flat, Y_flat, closed_form=True)
        print("❌ Failed to raise on 1D X")
    except IndexError as e:
        print(f"✅ Caught dimensionality issue in X: {e}")
def test_gradient_descent():
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

    assert np.allclose(model.coefficients, [3.0, 5.0], atol=1e-1), f"w incorrect: {model.coefficients}"
    assert np.isclose(model.bias, 7.0, atol=1e-1), f"b incorrect: {model.bias}"
    print("✅ Gradient descent example passed.")

test_closed_form()
test_validate_inputs()
test_gradient_descent()