import numpy as np
from scipy import stats
from itertools import combinations
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import warnings

class MCDMStatistics:
    @staticmethod
    def analyze_rankings(rankings, options):
        """Enhanced ranking analysis with more quantitative metrics"""
        analysis = {
            'basic_stats': {
                'mean': np.mean(rankings),
                'std': np.std(rankings),
                'skewness': stats.skew(rankings),
                'kurtosis': stats.kurtosis(rankings),
                'variance': np.var(rankings),
                'coefficient_of_variation': stats.variation(rankings)
            },
            'confidence_intervals': {
                '95%': stats.t.interval(0.95, len(rankings)-1, loc=np.mean(rankings), scale=stats.sem(rankings)),
                '99%': stats.t.interval(0.99, len(rankings)-1, loc=np.mean(rankings), scale=stats.sem(rankings))
            },
            'ranking_distribution': pd.Series(rankings).value_counts().to_dict()
        }
        
        # Only perform normality test if sample size is sufficient
        if len(rankings) >= 8:  # Minimum recommended sample size
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analysis['normality_test'] = stats.normaltest(rankings)
        else:
            analysis['normality_test'] = None
            
        return analysis

    @staticmethod
    def analyze_sensitivity(sensitivity_results, options):
        """Analyze sensitivity of rankings"""
        stability_scores = {}
        base_ranking = None
        
        # Get base ranking
        for (idx, var), ranking in sensitivity_results.items():
            if var == 0 or base_ranking is None:  # Use first ranking as base if no 0 variation
                base_ranking = ranking
                break
        
        if base_ranking is None:
            return {}
            
        # Calculate stability scores for each option
        for i, option in enumerate(options):
            rank_changes = []
            for ranking in sensitivity_results.values():
                current_rank = np.where(ranking == i)[0][0]
                base_rank = np.where(base_ranking == i)[0][0]
                rank_changes.append(abs(current_rank - base_rank))
            
            # Calculate stability score (1 = most stable, 0 = least stable)
            stability_scores[option] = 1 - (np.mean(rank_changes) / len(options))
            
        return stability_scores

    @staticmethod
    def comparative_analysis(current_results, reference_studies):
        """Compare current results with reference studies"""
        comparisons = {
            'ranking_correlation': {},
            'weight_differences': {},
            'performance_gaps': {},
            'statistical_tests': {}
        }
        
        for study_name, ref_study in reference_studies.items():
            # Spearman rank correlation
            correlation, p_value = stats.spearmanr(current_results['rankings'], ref_study['rankings'])
            comparisons['ranking_correlation'][study_name] = {
                'correlation': correlation,
                'p_value': p_value
            }
            
            # Weight differences analysis
            weight_diff = np.abs(current_results['weights'] - ref_study['weights'])
            comparisons['weight_differences'][study_name] = {
                'mean_diff': np.mean(weight_diff),
                'max_diff': np.max(weight_diff),
                'criteria_specific': dict(zip(ref_study['criteria'], weight_diff))
            }
            
            # Mann-Whitney U test for rankings
            stat, p_val = stats.mannwhitneyu(current_results['rankings'], ref_study['rankings'])
            comparisons['statistical_tests'][study_name] = {
                'mann_whitney_u': {'statistic': stat, 'p_value': p_val}
            }
        
        return comparisons

    @staticmethod
    def sensitivity_meta_analysis(sensitivity_results, weights, criteria):
        """Enhanced sensitivity analysis with meta-analysis"""
        meta_analysis = {
            'weight_elasticity': {},
            'ranking_stability': {},
            'critical_thresholds': {},
            'interaction_effects': {}
        }
        
        # Calculate weight elasticity
        for i, criterion in enumerate(criteria):
            changes = []
            effects = []
            for (idx, var), ranking in sensitivity_results.items():
                if idx == i:
                    changes.append(var)
                    effects.append(np.mean(np.abs(ranking - np.arange(len(ranking)))))
            
            if changes and effects:
                slope, intercept, r_value, p_value, std_err = stats.linregress(changes, effects)
                meta_analysis['weight_elasticity'][criterion] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_error': std_err
                }
        
        return meta_analysis

    @staticmethod
    def cross_validation_analysis(decision_matrix, weights, min_splits=2):
        """Cross-validation analysis for robustness"""
        n_alternatives = decision_matrix.shape[0]
        n_criteria = decision_matrix.shape[1]
        
        # Determine appropriate number of splits based on sample size
        n_splits = min(min_splits, n_criteria - 1)  # Ensure at least one criterion in test set
        
        cv_results = {
            'ranking_stability': [],
            'weight_sensitivity': [],
            'cross_validation_scores': []
        }
        
        if n_criteria < 3:  # Not enough samples for meaningful cross-validation
            return {
                'warning': 'Not enough criteria for cross-validation',
                'n_criteria': n_criteria,
                'min_required': 3
            }
        
        try:
            # K-fold cross validation on criteria subsets
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for train_idx, test_idx in kf.split(decision_matrix.T):
                if len(train_idx) < 2:  # Skip if training set is too small
                    continue
                    
                # Calculate rankings with subset of criteria
                train_matrix = decision_matrix[:, train_idx]
                train_weights = weights[train_idx]
                train_weights = train_weights / np.sum(train_weights)  # Renormalize weights
                
                # Calculate stability metrics
                rankings = np.argsort(-np.dot(train_matrix, train_weights))
                
                try:
                    stability_score = silhouette_score(train_matrix, rankings)
                    cv_results['ranking_stability'].append(stability_score)
                except ValueError as e:
                    cv_results['ranking_stability'].append(None)
                    
                # Store weight sensitivity
                cv_results['weight_sensitivity'].append(
                    np.std(np.dot(train_matrix, train_weights))
                )
                
                # Store cross-validation score (correlation between full and subset rankings)
                full_rankings = np.argsort(-np.dot(decision_matrix, weights))
                cv_results['cross_validation_scores'].append(
                    stats.spearmanr(rankings, full_rankings)[0]
                )
            
            # Calculate average metrics
            cv_results['summary'] = {
                'avg_stability': np.mean([x for x in cv_results['ranking_stability'] if x is not None]),
                'avg_sensitivity': np.mean(cv_results['weight_sensitivity']),
                'avg_cv_score': np.mean(cv_results['cross_validation_scores']),
                'n_splits_used': n_splits,
                'n_criteria': n_criteria
            }
            
        except Exception as e:
            cv_results['error'] = str(e)
            cv_results['status'] = 'failed'
            
        return cv_results

    @staticmethod
    def uncertainty_analysis(decision_matrix, weights, n_simulations=1000):
        """Monte Carlo simulation for uncertainty analysis"""
        n_alternatives, n_criteria = decision_matrix.shape
        
        # Generate random variations
        weight_variations = np.random.normal(weights, 0.05 * weights, (n_simulations, n_criteria))
        weight_variations = weight_variations / weight_variations.sum(axis=1)[:, np.newaxis]
        
        # Run simulations
        simulation_results = []
        for sim_weights in weight_variations:
            sim_ranking = np.argsort(-np.dot(decision_matrix, sim_weights))
            simulation_results.append(sim_ranking)
        
        # Analyze results
        simulation_results = np.array(simulation_results)
        
        # Calculate rank probabilities and confidence
        rank_probabilities = np.zeros((n_alternatives, n_alternatives))
        rank_confidence = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                rank_probabilities[i, j] = np.mean(simulation_results[:, j] == i)
            
            # Use new scipy.stats.mode interface
            mode_result = stats.mode(simulation_results[:, i])
            rank_confidence[i] = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
        
        uncertainty_metrics = {
            'rank_probabilities': rank_probabilities,
            'rank_confidence': rank_confidence,
            'rank_volatility': np.std(simulation_results, axis=0),
            'simulation_summary': {
                'n_simulations': n_simulations,
                'weight_variation_std': 0.05,
                'mean_rank_volatility': np.mean(np.std(simulation_results, axis=0))
            }
        }
        
        return uncertainty_metrics 