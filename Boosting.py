from decision_tree import MyDecisionTree

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
# Use of sklearn.tree for the *Weak Learner* is allowed
from sklearn.tree import DecisionTreeRegressor 

from enum import Enum

class LossType(Enum):
    L2 = 1
    L1 = 2
    HUBER = 3
    
class GBRegressor(BaseEstimator, RegressorMixin):
    """
    Custom Gradient Boosting Regressor.
    Uses L2, L1 and Huber Losses
    Weak learner is constrained to a shallow Regression Tree (Decision Stump).
    """

    # --------------------------------------------------------------------------------
    # Task 1: Weak Learner Initialization. 
    # --------------------------------------------------------------------------------
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=None, loss_type=LossType.L2, delta=None, decision_stump_type=DecisionTreeRegressor):
        """
        Initializes the Gradient Boosting Regressor.
        
        Parameters:
        n_estimators (int): The number of boosting stages (trees) to perform.
        learning_rate (float): Shrinks the contribution of each tree.
        max_depth (int): The maximum depth of the base regression estimator (weak learner).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estmiators_ = []  # Stores the trained base learners
        self.initial_prediction_ = None  # Initial constant prediction
        self.loss_type = loss_type
        self.decision_stump_type = decision_stump_type
        if loss_type == LossType.HUBER and delta is None:
            raise ValueError("Delta must be provided for Huber loss.")
        self.delta = delta # Huber threshold

    # --------------------------------------------------------------------------------
    # TASK 1: Weak Learner Implementation 
    # --------------------------------------------------------------------------------
    def _make_weak_learner(self):
        """
        Creates and returns an instance of the weak learner, i.e., the Decision Stump.
        
        Parameters:
        decision_stump_type: The type of decision stump to create, default is DecisionTreeRegressor.
        Returns:
        weak_learner: An instance of the weak learner.
        """

        if self.decision_stump_type == DecisionTreeRegressor:
            weak_learner = DecisionTreeRegressor(max_depth=self.max_depth)
        else:
            weak_learner = MyDecisionTree(max_depth=self.max_depth) 
        return weak_learner


    # --------------------------------------------------------------------------------
    # TASK 3: The negative gradient for L2, L1 and Huber loss
    # --------------------------------------------------------------------------------

    def _get_negative_gradient(self, y, prediction):
        """
        Computes the negative gradient according to the loss_type configuration parameter selected. 
        For LossType.L2, that is the residuals.
        For LossType.L1, that is the sign of the residuals.
        For LossType.Huber, that is the piecewise function combining the sensitivity of L2 loss for small errors with the robustness
        of L1 loss for large errors.

        Parameters:
        y : True label
        prediction: current prediction

        Returns:
        ngrad : the negative gradient
        """

        # Compute the residuals (difference between true labels and predictions)
        residual = y - prediction

        # Negative gradient for L2 loss is simply the residuals
        if self.loss_type == LossType.L2:
            return residual

        # Negative gradient for L1 loss is the sign of the residuals
        elif self.loss_type == LossType.L1:
            # np.sign returns -1 for negative, 0 for zero, 1 for positive
            return np.sign(residual)

        elif self.loss_type == LossType.HUBER:
            delta = self.delta
            
            # For small residuals (|residual| <= delta), use L2-like gradient (residual itself)
            # For large residuals (|residual| > delta), use L1-like gradient scaled by delta
            return np.where(
                np.abs(residual) <= delta,
                residual,
                delta * np.sign(residual)
            )

        else:
            raise ValueError("Unsupported loss type")
        

    # --------------------------------------------------------------------------------
    # TASK 4: The Core Boosting Algorithm
    # --------------------------------------------------------------------------------        
    def fit(self, X, y):
        """
        Builds the additive model in a forward stage-wise fashion.

        Parameters:
        X (np.ndarray): The training input samples.
        y (np.ndarray): The target values.
        """
        # Convert X and y to NumPy arrays for consistent handling
        X, y = np.array(X), np.array(y)
        
        # 1. Initialize the model with a constant value (e.g. mean of y)
        if self.loss_type == LossType.L2:
            self.initial_prediction_ = np.mean(y)
        else:
            self.initial_prediction_ = np.median(y)
        
        # Initialize the current ensemble prediction F(x) to F_0(x)
        current_prediction = np.full_like(y, self.initial_prediction_, dtype=float)
        
        for _ in range(self.n_estimators):
            
            # 2. Compute the negative gradient (pseudo-residuals)
            residuals = self._get_negative_gradient(y, current_prediction)

            # 3. Fit the base learner (h_m) to the pseudo-residuals
            h_m = self._make_weak_learner()
            h_m.fit(X, residuals)

            if self.loss_type in [LossType.L1, LossType.HUBER]:
                
                raw_residuals = y - current_prediction

                if self.decision_stump_type == DecisionTreeRegressor:
                    leaf_indices = h_m.apply(X)
                    unique_leaves = np.unique(leaf_indices)
                    for leaf_idx in unique_leaves:
                        mask = (leaf_indices == leaf_idx)
                        leaf_val = np.median(raw_residuals[mask])
                        h_m.tree_.value[leaf_idx, 0, 0] = leaf_val
                
                else: 
                    leaf_nodes = h_m.apply(X)
                    
                    leaf_map = {}
                    for i, node in enumerate(leaf_nodes):
                        if node not in leaf_map:
                            leaf_map[node] = []
                        leaf_map[node].append(i)
                    
                    for node, indices in leaf_map.items():
                        node.value = np.median(raw_residuals[indices])
                        
            # Store the weak learner
            self.estmiators_.append(h_m)
            
            # 4. Update the ensemble model
            h_m_prediction = h_m.predict(X)

            current_prediction = current_prediction + self.learning_rate * h_m_prediction
            
        return self


    # --------------------------------------------------------------------------------
    # TASK 4: Make Predictions
    # --------------------------------------------------------------------------------
    def predict(self, X):
        """
        Predicts target values for new data points.

        Parameters:
        X (np.ndarray): The input samples.

        Returns:
        ensemble_prediction (np.ndarray): The predicted target values.
        """
        if not self.estmiators_:
            # If the model hasn't been fitted, return the initial prediction
            return np.full(X.shape[0], self.initial_prediction_)
            
        # Initialize prediction with the initial constant prediction F_0(x)
        ensemble_prediction = np.full(X.shape[0], self.initial_prediction_, dtype=float)
        
        # Sum up the predictions of all weak learners, weighted by the learning rate
        for h_m in self.estmiators_:
            ensemble_prediction += self.learning_rate * h_m.predict(X)
        
        return ensemble_prediction