import numpy as np

from main import MCMetrics



class CrossValidation:
    
    def __init__(self):
        pass

    def k_fold_cv(self, k, cm):
        """
        Performs k-fold cross-validation on the confusion matrix.

        Parameters
        ----------
        k : int
            Number of folds for cross-validation.
        cm : np.ndarray
            Confusion matrix to be used for cross-validation.

        Returns
        -------
        list
            List of MCMetrics instances for each fold.
        """
        
        # Ensure the confusion matrix is a numpy array
        cm = np.array(cm)
        
        # Get the number of classes
        num_classes = cm.shape[0]
        
        # Initialize list to hold MCMetrics instances for each fold
        cv_results = []
        
        # Perform k-fold cross-validation
        for fold in range(k):
            # Create a new confusion matrix for this fold
            fold_cm = np.zeros_like(cm)
            
            # Distribute the counts in the confusion matrix across the folds
            for i in range(num_classes):
                for j in range(num_classes):
                    count = cm[i, j]
                    fold_count = count // k
                    remainder = count % k
                    
                    # Assign the base count to all folds
                    fold_cm[i, j] += fold_count
                    
                    # Distribute the remainder across the first 'remainder' folds
                    if fold < remainder:
                        fold_cm[i, j] += 1
            
            # Create an MCMetrics instance for this fold and add it to the results list
            mc_metrics_instance = MCMetrics(cm=fold_cm)
            cv_results.append(mc_metrics_instance)
        
        return cv_results