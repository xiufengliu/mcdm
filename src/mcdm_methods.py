import numpy as np
from scipy.stats import gmean

class MCDMethods:
    @staticmethod
    def ahp(comparison_matrix):
        """Analytic Hierarchy Process"""
        n = len(comparison_matrix)
        # Calculate eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(np.array(comparison_matrix, dtype=float))
        max_idx = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_idx].real
        weights = weights / weights.sum()
        
        # Calculate consistency ratio
        ci = (eigenvalues[max_idx].real - n) / (n - 1)
        ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
        cr = ci / ri[n] if n >= 3 else 0
        
        return weights, cr

    @staticmethod
    def topsis(decision_matrix, weights, criteria_type):
        """TOPSIS Method"""
        # Normalize decision matrix
        norm_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))
        
        # Calculate weighted normalized matrix
        weighted_matrix = norm_matrix * weights
        
        # Determine ideal and negative-ideal solutions
        ideal = np.max(weighted_matrix, axis=0) * np.array([1 if t == "max" else -1 for t in criteria_type])
        neg_ideal = np.min(weighted_matrix, axis=0) * np.array([1 if t == "min" else -1 for t in criteria_type])
        
        # Calculate distances
        d_pos = np.sqrt(np.sum((weighted_matrix - ideal)**2, axis=1))
        d_neg = np.sqrt(np.sum((weighted_matrix - neg_ideal)**2, axis=1))
        
        # Calculate relative closeness
        closeness = d_neg / (d_pos + d_neg)
        return closeness

    @staticmethod
    def wsm(decision_matrix, weights, criteria_type):
        """Weighted Sum Model"""
        # Normalize the decision matrix
        normalized = np.zeros_like(decision_matrix, dtype=float)
        for j in range(decision_matrix.shape[1]):
            if criteria_type[j] == "max":
                normalized[:, j] = decision_matrix[:, j] / np.max(decision_matrix[:, j])
            else:
                normalized[:, j] = np.min(decision_matrix[:, j]) / decision_matrix[:, j]
        
        # Calculate weighted sum
        return np.sum(normalized * weights, axis=1)

    @staticmethod
    def wpm(decision_matrix, weights, criteria_type):
        """Weighted Product Model"""
        # Convert min criteria to max by taking reciprocal
        modified_matrix = decision_matrix.copy()
        for j, ctype in enumerate(criteria_type):
            if ctype == "min":
                modified_matrix[:, j] = 1 / modified_matrix[:, j]
        
        # Calculate weighted product
        return np.prod(modified_matrix ** weights, axis=1)

    @staticmethod
    def vikor(decision_matrix, weights, criteria_type):
        """VIKOR Method"""
        # Normalize and determine f* and f-
        n_matrix = decision_matrix.copy()
        f_star = np.zeros(len(weights))
        f_minus = np.zeros(len(weights))
        
        for j, ctype in enumerate(criteria_type):
            if ctype == "max":
                f_star[j] = np.max(decision_matrix[:, j])
                f_minus[j] = np.min(decision_matrix[:, j])
            else:
                f_star[j] = np.min(decision_matrix[:, j])
                f_minus[j] = np.max(decision_matrix[:, j])
        
        # Calculate S and R
        S = np.zeros(len(decision_matrix))
        R = np.zeros(len(decision_matrix))
        
        for i in range(len(decision_matrix)):
            for j in range(len(weights)):
                if criteria_type[j] == "max":
                    S[i] += weights[j] * (f_star[j] - decision_matrix[i,j]) / (f_star[j] - f_minus[j])
                    R[i] = max(R[i], weights[j] * (f_star[j] - decision_matrix[i,j]) / (f_star[j] - f_minus[j]))
                else:
                    S[i] += weights[j] * (decision_matrix[i,j] - f_star[j]) / (f_minus[j] - f_star[j])
                    R[i] = max(R[i], weights[j] * (decision_matrix[i,j] - f_star[j]) / (f_minus[j] - f_star[j]))
        
        # Calculate Q
        v = 0.5  # Weight of strategy
        S_star = np.min(S)
        S_minus = np.max(S)
        R_star = np.min(R)
        R_minus = np.max(R)
        
        Q = v * (S - S_star) / (S_minus - S_star) + (1-v) * (R - R_star) / (R_minus - R_star)
        
        return 1 - Q  # Convert to same scale as other methods (higher is better)

    @staticmethod
    def promethee(decision_matrix, weights, criteria_type):
        """PROMETHEE II Method"""
        n_alternatives = len(decision_matrix)
        n_criteria = len(weights)
        
        # Calculate preference matrix for each criterion
        preference = np.zeros((n_alternatives, n_alternatives, n_criteria))
        
        for k in range(n_criteria):
            for i in range(n_alternatives):
                for j in range(n_alternatives):
                    d = decision_matrix[i,k] - decision_matrix[j,k]
                    if criteria_type[k] == "min":
                        d = -d
                    preference[i,j,k] = max(0, d)
        
        # Calculate aggregated preference matrix
        weighted_preference = np.sum(preference * weights, axis=2)
        
        # Calculate positive and negative flows
        positive_flow = np.sum(weighted_preference, axis=1) / (n_alternatives - 1)
        negative_flow = np.sum(weighted_preference, axis=0) / (n_alternatives - 1)
        
        # Calculate net flow
        net_flow = positive_flow - negative_flow
        
        return (net_flow - np.min(net_flow)) / (np.max(net_flow) - np.min(net_flow)) 