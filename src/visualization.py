import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


class MCDMVisualizer:
    def __init__(self):
        # Set style for all plots
        #plt.style.use('seaborn')
        sns.set_style('darkgrid')  # Or another Seaborn style like 'white', 'ticks', etc.

        sns.set_palette("husl")
    
    def plot_weights(self, criteria, weights, title):
        """Plot criteria weights as a bar chart"""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(criteria, weights)
        plt.title(f'Criteria Weights - {title}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Weight')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_rankings(self, options, rankings, title):
        """Plot TOPSIS rankings as a horizontal bar chart"""
        plt.figure(figsize=(10, 6))
        bars = plt.barh(options, rankings)
        plt.title(f'TOPSIS Rankings - {title}')
        plt.xlabel('Score')
        
        # Add value labels at the end of bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_sensitivity(self, sensitivity_data, criteria, title):
        """Plot sensitivity analysis results as a heatmap"""
        variations = sorted(list(set(var for _, var in sensitivity_data.keys())))
        criteria_indices = sorted(list(set(idx for idx, _ in sensitivity_data.keys())))
        
        # Create matrix for heatmap
        matrix = np.zeros((len(criteria_indices), len(variations)))
        for (idx, var), ranking in sensitivity_data.items():
            var_idx = variations.index(var)
            matrix[idx][var_idx] = np.mean(np.abs(ranking))
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, 
                   xticklabels=[f'{v:+.0%}' for v in variations],
                   yticklabels=criteria,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd')
        plt.title(f'Sensitivity Analysis - {title}')
        plt.xlabel('Weight Variation')
        plt.ylabel('Criteria')
        plt.tight_layout()
        return plt.gcf()

    def plot_radar_chart(self, options, criteria, decision_matrix, title):
        """Plot radar chart for comparing options across criteria"""
        angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False)
        
        # Close the plot by appending first value
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, option in enumerate(options):
            values = np.concatenate((decision_matrix[i], [decision_matrix[i][0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=option)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        plt.title(f'Radar Chart - {title}')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        return plt.gcf()

    def plot_correlation_matrix(self, decision_matrix, criteria, title):
        """Plot correlation matrix between criteria"""
        corr_matrix = np.corrcoef(decision_matrix.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=criteria,
                   yticklabels=criteria,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0)
        plt.title(f'Criteria Correlation Matrix - {title}')
        plt.tight_layout()
        return plt.gcf()

    def plot_ranking_distribution(self, sensitivity_results, options, title):
        """Plot distribution of rankings under sensitivity analysis"""
        rankings_dist = []
        labels = []
        
        for option_idx in range(len(options)):
            rankings = []
            for ranking in sensitivity_results.values():
                rankings.append(np.where(ranking == option_idx)[0][0])
            rankings_dist.append(rankings)
            labels.extend([options[option_idx]] * len(rankings))
        
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=rankings_dist)
        plt.xticks(range(len(options)), options, rotation=45)
        plt.title(f'Ranking Distribution Under Sensitivity - {title}')
        plt.ylabel('Rank')
        plt.tight_layout()
        return plt.gcf()

    def plot_method_comparison(self, method_rankings, case_name):
        """Plot comparison of rankings across different methods"""
        methods = list(method_rankings.keys())
        alternatives = range(len(next(iter(method_rankings.values()))))
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(alternatives))
        width = 0.15
        multiplier = 0
        
        for method, rankings in method_rankings.items():
            offset = width * multiplier
            plt.bar(x + offset, rankings, width, label=method)
            multiplier += 1
        
        plt.xlabel('Alternatives')
        plt.ylabel('Score')
        plt.title(f'Method Comparison - {case_name}')
        plt.xticks(x + width * (len(methods) - 1) / 2, [f'A{i+1}' for i in x])
        plt.legend(loc='best')
        plt.tight_layout()
        return plt.gcf()

    def plot_rank_reversals(self, reversal_results, case_name):
        """Plot rank reversal analysis results"""
        methods = list(reversal_results['frequency'].keys())  # Get methods from frequency dict
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Frequency of rank reversals
        frequencies = [reversal_results['frequency'][m] for m in methods]
        ax1.bar(methods, frequencies)
        ax1.set_title('Frequency of Rank Reversals')
        ax1.set_ylabel('Number of Reversals')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Average severity of reversals
        severities = [reversal_results['severity'][m] for m in methods]
        ax2.bar(methods, severities)
        ax2.set_title('Average Severity of Rank Reversals')
        ax2.set_ylabel('Severity Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Stability scores
        stabilities = [reversal_results['stability'][m] for m in methods]
        ax3.bar(methods, stabilities)
        ax3.set_title('Method Stability Scores')
        ax3.set_ylabel('Stability Score')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Rank Reversal Analysis - {case_name}')
        plt.tight_layout()
        return plt.gcf()

    def plot_stability_analysis(self, method_stability, case_name):
        """Plot stability analysis results"""
        methods = list(method_stability.keys())
        fig, axes = plt.subplots(len(methods), 1, figsize=(12, 4*len(methods)))
        if len(methods) == 1:
            axes = [axes]
            
        for ax, method in zip(axes, methods):
            stability = method_stability[method]
            variations = []
            rankings = []
            
            for (criterion, var), ranking in stability['stability_results'].items():
                variations.append(var)
                rankings.append(ranking)
            
            im = ax.imshow(rankings, aspect='auto', cmap='YlOrRd')
            ax.set_title(f'{method} Stability Analysis')
            ax.set_xlabel('Alternatives')
            ax.set_ylabel('Weight Variation')
            plt.colorbar(im, ax=ax)
            
        plt.suptitle(f'Method Stability Analysis - {case_name}')
        plt.tight_layout()
        return plt.gcf()

    def plot_statistical_tests(self, statistical_results, case_name):
        """Plot statistical test results"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots for different test results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: P-values for statistical tests
        test_names = []
        p_values = []
        
        # Collect p-values from tests
        if 'friedman_test' in statistical_results:
            test_names.append('Friedman')
            p_values.append(statistical_results['friedman_test']['p_value'])
            
        if 'kruskal_wallis' in statistical_results:
            test_names.append('Kruskal-Wallis')
            p_values.append(statistical_results['kruskal_wallis']['p_value'])
        
        # Plot p-values
        x = np.arange(len(test_names))
        ax1.bar(x, p_values)
        ax1.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (α=0.05)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_names, rotation=45)
        ax1.set_ylabel('p-value')
        ax1.set_title('Statistical Significance Tests')
        ax1.legend()
        
        # Plot 2: Correlation coefficients
        if 'spearman_correlation' in statistical_results:
            correlations = []
            pair_names = []
            
            for pair, values in statistical_results['spearman_correlation'].items():
                correlations.append(values['correlation'])
                pair_names.append(pair)
            
            x = np.arange(len(pair_names))
            ax2.bar(x, correlations)
            ax2.set_xticks(x)
            ax2.set_xticklabels(pair_names, rotation=45)
            ax2.set_ylabel('Correlation Coefficient')
            ax2.set_title('Spearman Correlations Between Methods')
            
            # Add Kendall's W if available
            if 'kendall_w' in statistical_results:
                ax2.axhline(y=statistical_results['kendall_w']['w'], 
                          color='g', linestyle='--', 
                          label="Kendall's W")
                ax2.legend()
        
        plt.suptitle(f'Statistical Analysis Results - {case_name}')
        plt.tight_layout()
        return fig 