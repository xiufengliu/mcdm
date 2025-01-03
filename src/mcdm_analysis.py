import numpy as np
from scipy.stats import gmean

class MCDMAnalysis:
    def __init__(self):
        # Constants for AHP
        self.RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    def calculate_ahp_weights(self, matrix):
        """Calculate weights using AHP method"""
        # Convert matrix to numpy array if not already
        matrix = np.array(matrix, dtype=float)
        n = len(matrix)
        
        # Calculate weights using geometric mean method
        weights = gmean(matrix, axis=1)
        weights = weights / weights.sum()
        
        # Calculate consistency
        eigenval = np.max(np.linalg.eigvals(matrix).real)
        ci = (eigenval - n) / (n - 1)
        cr = ci / self.RI[n]
        
        return weights, cr
    
    def topsis(self, decision_matrix, weights, criteria_type):
        """
        Implement TOPSIS method
        decision_matrix: numpy array of values
        weights: criteria weights
        criteria_type: list of 'max' or 'min' for each criterion
        """
        # Normalize decision matrix
        norm_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))
        
        # Weight normalization
        weighted_matrix = norm_matrix * weights
        
        # Ideal solutions
        ideal_best = np.array([max(col) if ctype == 'max' else min(col) 
                             for col, ctype in zip(weighted_matrix.T, criteria_type)])
        ideal_worst = np.array([min(col) if ctype == 'max' else max(col) 
                              for col, ctype in zip(weighted_matrix.T, criteria_type)])
        
        # Distances
        s_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        s_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
        
        # Closeness coefficient
        closeness = s_worst / (s_best + s_worst)
        
        return closeness

    def sensitivity_analysis(self, base_matrix, base_weights, criteria_type, variations=[-0.2, -0.1, 0.1, 0.2]):
        """
        Perform sensitivity analysis by varying weights
        
        Parameters:
        -----------
        base_matrix : numpy.ndarray
            The decision matrix
        base_weights : numpy.ndarray
            The original criteria weights
        criteria_type : list
            List of 'max' or 'min' for each criterion
        variations : list
            List of weight variations to test
        """
        results = {}
        base_ranking = np.argsort(-self.topsis(base_matrix, base_weights, criteria_type))
        
        for var in variations:
            for i in range(len(base_weights)):
                new_weights = base_weights.copy()
                new_weights[i] *= (1 + var)
                new_weights = new_weights / new_weights.sum()
                
                new_ranking = np.argsort(-self.topsis(base_matrix, new_weights, criteria_type))
                results[(i, var)] = new_ranking
                
        return results