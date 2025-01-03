import numpy as np
from scipy import stats
from itertools import combinations, permutations
import warnings

class MethodSensitivityAnalysis:
    def __init__(self, mcdm_methods):
        """Initialize with MCDM methods object"""
        self.methods = mcdm_methods

    def analyze_method_stability(self, decision_matrix, weights, criteria_type, method_name):
        """Analyze stability of specific method under weight variations"""
        variations = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
        n_criteria = len(weights)
        stability_results = {}
        
        method = getattr(self.methods, method_name.lower())
        base_ranking = method(decision_matrix, weights, criteria_type)
        
        for i in range(n_criteria):
            for var in variations:
                if var == 0:
                    continue
                    
                # Modify weights
                mod_weights = weights.copy()
                mod_weights[i] *= (1 + var)
                mod_weights = mod_weights / np.sum(mod_weights)  # Renormalize
                
                # Calculate new ranking
                new_ranking = method(decision_matrix, mod_weights, criteria_type)
                
                # Store results
                stability_results[(i, var)] = new_ranking
                
        return stability_results, base_ranking

    def cross_method_analysis(self, decision_matrix, weights, criteria_type):
        """Compare results across different methods"""
        methods = ['topsis', 'wsm', 'wpm', 'vikor', 'promethee']
        rankings = {}
        
        # Calculate rankings for each method
        for method in methods:
            method_func = getattr(self.methods, method)
            rankings[method] = method_func(decision_matrix, weights, criteria_type)
        
        # Calculate correlation matrix between methods
        n_methods = len(methods)
        correlation_matrix = np.zeros((n_methods, n_methods))
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                correlation_matrix[i,j] = stats.spearmanr(rankings[m1], rankings[m2])[0]
                
        return rankings, correlation_matrix

    def statistical_tests(self, method_rankings):
        """Perform comprehensive statistical tests"""
        test_results = {
            'friedman_test': {},
            'kendall_w': {},
            'kruskal_wallis': {},
            'spearman_correlation': {}
        }
        
        methods = list(method_rankings.keys())
        rankings_array = np.array([method_rankings[m] for m in methods])
        
        # Check if we have enough variation for tests
        if len(np.unique(rankings_array)) <= 1:
            logger.warning("Constant rankings detected - statistical tests may not be meaningful")
            test_results['friedman_test'] = {'statistic': np.nan, 'p_value': np.nan}
            test_results['kendall_w'] = {'w': np.nan}
            test_results['kruskal_wallis'] = {'statistic': np.nan, 'p_value': np.nan}
        else:
            # Friedman test
            try:
                friedman_stat, friedman_p = stats.friedmanchisquare(*rankings_array)
                test_results['friedman_test'] = {'statistic': friedman_stat, 'p_value': friedman_p}
            except Exception as e:
                logger.warning(f"Friedman test failed: {str(e)}")
                test_results['friedman_test'] = {'statistic': np.nan, 'p_value': np.nan}
            
            # Kendall's W
            try:
                rankings_matrix = np.array([stats.rankdata(-r) for r in rankings_array])
                kendall_w = stats.kendalltau(rankings_matrix[0], rankings_matrix[1])[0]
                test_results['kendall_w'] = {'w': kendall_w if not np.isnan(kendall_w) else 0}
            except Exception as e:
                logger.warning(f"Kendall W calculation failed: {str(e)}")
                test_results['kendall_w'] = {'w': np.nan}
            
            # Kruskal-Wallis H-test
            try:
                kruskal_stat, kruskal_p = stats.kruskal(*rankings_array)
                test_results['kruskal_wallis'] = {'statistic': kruskal_stat, 'p_value': kruskal_p}
            except Exception as e:
                logger.warning(f"Kruskal-Wallis test failed: {str(e)}")
                test_results['kruskal_wallis'] = {'statistic': np.nan, 'p_value': np.nan}
        
        # Spearman correlation with error handling
        for m1, m2 in combinations(methods, 2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spearman_corr, spearman_p = stats.spearmanr(
                        method_rankings[m1], 
                        method_rankings[m2]
                    )
                test_results['spearman_correlation'][f'{m1}_vs_{m2}'] = {
                    'correlation': spearman_corr if not np.isnan(spearman_corr) else 0,
                    'p_value': spearman_p if not np.isnan(spearman_p) else 1
                }
            except Exception as e:
                logger.warning(f"Spearman correlation failed for {m1} vs {m2}: {str(e)}")
                test_results['spearman_correlation'][f'{m1}_vs_{m2}'] = {
                    'correlation': np.nan,
                    'p_value': np.nan
                }
        
        return test_results

    def detailed_rank_reversal(self, decision_matrix, weights, criteria_type):
        """Perform detailed rank reversal analysis"""
        methods = ['topsis', 'wsm', 'wpm', 'vikor', 'promethee']
        n_alternatives = len(decision_matrix)
        
        reversal_analysis = {
            'frequency': {m: 0 for m in methods},
            'severity': {m: [] for m in methods},
            'stability': {m: [] for m in methods},
            'patterns': {m: {} for m in methods}
        }
        
        # Analyze all possible subsets
        for size in range(2, n_alternatives + 1):
            for subset in combinations(range(n_alternatives), size):
                subset_matrix = decision_matrix[list(subset)]
                
                for method in methods:
                    method_func = getattr(self.methods, method)
                    orig_ranking = method_func(decision_matrix, weights, criteria_type)
                    sub_ranking = method_func(subset_matrix, weights, criteria_type)
                    
                    # Check for rank reversals
                    orig_order = np.argsort(-orig_ranking)[list(subset)]
                    sub_order = np.argsort(-sub_ranking)
                    
                    # Analyze reversals
                    for i, j in combinations(range(len(subset)), 2):
                        orig_comp = orig_order[i] < orig_order[j]
                        sub_comp = sub_order[i] < sub_order[j]
                        
                        if orig_comp != sub_comp:
                            reversal_analysis['frequency'][method] += 1
                            severity = abs(orig_ranking[subset[i]] - orig_ranking[subset[j]])
                            reversal_analysis['severity'][method].append(severity)
                            
                            # Record pattern
                            pattern = f"A{subset[i]+1}-A{subset[j]+1}"
                            if pattern not in reversal_analysis['patterns'][method]:
                                reversal_analysis['patterns'][method][pattern] = 0
                            reversal_analysis['patterns'][method][pattern] += 1
                    
                    # Calculate stability score
                    stability = 1 - (np.sum(orig_order != sub_order) / len(subset))
                    reversal_analysis['stability'][method].append(stability)
        
        # Calculate average severity and stability
        for method in methods:
            if reversal_analysis['severity'][method]:
                reversal_analysis['severity'][method] = np.mean(reversal_analysis['severity'][method])
            else:
                reversal_analysis['severity'][method] = 0
                
            reversal_analysis['stability'][method] = np.mean(reversal_analysis['stability'][method])
        
        return reversal_analysis 