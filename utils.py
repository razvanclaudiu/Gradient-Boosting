import numpy as np
import matplotlib.pyplot as plt
import time

from enum import Enum
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score


class ModelType(Enum):

    MY_GBM = "my_gbm"
    My_GBM_CUSTOM_TREE = "my_gbm_custom_tree"
    SKLEARN_GBM_SKLEARN_TREE = "sklearn_gbm_sklearn_tree"
    ALL = "all"

def visualize_results(y_true, y_pred, fig_path=None):

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    if fig_path:
        plt.savefig(fig_path)
    plt.show()

def visualize_dataset(X, y, fig_path=None):
    for i in range(X.shape[1]):
        plt.scatter(X[:, i], y, alpha=0.5)
        plt.xlabel(f'Feature {i}')
        plt.ylabel('Target y')
        if fig_path:
            plt.savefig(fig_path+f'_feature_{i}.png')
        plt.show()

def inject_outliers(y, fraction=0.1, magnitude=10):
    y_out = y.copy()
    n_outliers = int(len(y) * fraction)
    indices = np.random.choice(len(y), n_outliers, replace=False)
    # Add extreme noise to selected indices
    noise = np.random.normal(0, np.std(y) * magnitude, n_outliers)
    y_out[indices] += noise
    return y_out

def process_models(model_dict, category_name, X_train, X_test, y_train, y_test, history):
    for loss_name, model in model_dict.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        history[category_name][loss_name].append(mae)


def models_fit_and_prediction(models: dict, X_train, X_test, y_train, y_test, visualize_dataset: bool = True, 
                              model_type = ModelType.MY_GBM, fig_path_prefix: str = "figs/"):
    
    best_config = None
    min_mae = float('inf')

    for name, model in models.items():
        start = time.time()
            
        model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        # Metric Calculations
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n[{model_type.name}] Loss: {name}")
        print(f"  MAE: {mae:.4f} | MSE: {mse:.4f} | MedAE: {medae:.4f} | R2: {r2:.4f}")
        print(f"  Time: {elapsed:.4f}s")

        # Track the winner based on MAE (more robust for comparing different losses)
        if mae < min_mae:
            min_mae = mae
            best_config = {
                'model_type': model_type.name,
                'loss_name': name,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'depth': getattr(model, 'max_depth', 'N/A')
            }

        if visualize_dataset:
            visualize_results(y_test, y_pred, f"{fig_path_prefix}{model_type.value}_{name}.png")

    return best_config