import numpy as np

class MyDecisionTree:
    """
    Simple CART-style regression tree implemented from scratch.
    Intended as a weak learner for Gradient Boosting.
    """

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # prediction at leaf

    def __init__(self, max_depth=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    # -----------------------------------------------------
    # Apply method to retrieve leaf nodes
    # -----------------------------------------------------
    def apply(self, X):
        """
        Returns the leaf node object for each sample in X.
        Useful for inspecting or updating leaves after fitting.
        """
        X = np.asarray(X)
        # Return a list of the actual Node objects
        return [self._apply_one(x, self.root) for x in X]

    def _apply_one(self, x, node):
        # If it's a leaf (has a value), return the node itself
        if node.value is not None:
            return node
        
        # Otherwise, traverse down
        feature_val = x[node.feature]
        if feature_val <= node.threshold:
            return self._apply_one(x, node.left)
        else:
            return self._apply_one(x, node.right)
    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    # -----------------------------------------------------
    # Build tree recursively
    # -----------------------------------------------------
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return self.Node(value=np.mean(y))

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return self.Node(value=np.mean(y))

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return self.Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    # -----------------------------------------------------
    # Find best split using MSE
    # -----------------------------------------------------
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                mse = (
                    len(y_left) * np.var(y_left)
                    + len(y_right) * np.var(y_right)
                ) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------
    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
