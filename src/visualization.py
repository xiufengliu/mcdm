import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


class MCDMVisualizer:
    def __init__(self):
        # Reset to default style first
        plt.style.use('default')
        
        # Nature style settings
        plt.rcParams['figure.figsize'] = (8, 5)
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.grid'] = True
        
        # Nature color palette
        self.colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
        sns.set_palette(self.colors)
        
        # Global font sizes
        self.SMALL_SIZE = 10
        self.MEDIUM_SIZE = 12
        self.BIGGER_SIZE = 14
        
        # Set font sizes
        plt.rc('font', size=self.MEDIUM_SIZE, family='Arial')
        plt.rc('axes', titlesize=self.BIGGER_SIZE)
        plt.rc('axes', labelsize=self.BIGGER_SIZE)
        plt.rc('xtick', labelsize=self.MEDIUM_SIZE)
        plt.rc('ytick', labelsize=self.MEDIUM_SIZE)
        plt.rc('legend', fontsize=self.MEDIUM_SIZE)
        
        # Set background color to white
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Set DPI for better quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_weights(self, criteria, weights, title):
        """Plot criteria weights as a bar chart"""
        plt.figure(figsize=(8, 5))
        bars = plt.bar(criteria, weights, color=self.colors[0])
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
        """Plot rankings as a horizontal bar chart"""
        plt.figure(figsize=(8, 5))
        bars = plt.barh(options, rankings, color=self.colors[1])
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
        
        plt.figure(figsize=(8, 5))
        x = np.arange(len(alternatives))
        width = 0.15
        multiplier = 0
        
        for i, (method, rankings) in enumerate(method_rankings.items()):
            offset = width * multiplier
            plt.bar(x + offset, rankings, width, label=method, color=self.colors[i % len(self.colors)])
            multiplier += 1
        
        plt.xlabel('Alternatives')
        plt.ylabel('Score')
        plt.xticks(x + width * (len(methods) - 1) / 2, [f'A{i+1}' for i in x])
        plt.legend(loc='best', frameon=True, edgecolor='black')
        plt.tight_layout()
        return plt.gcf()

    def plot_rank_reversals(self, reversal_results, case_name):
        """Plot rank reversal analysis results"""
        methods = list(reversal_results['frequency'].keys())
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        
        # Plot 1: Frequency of rank reversals
        ax1.bar(methods, [reversal_results['frequency'][m] for m in methods], color=self.colors[0])
        ax1.set_ylabel('Number of Reversals')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Average severity
        ax2.bar(methods, [reversal_results['severity'][m] for m in methods], color=self.colors[1])
        ax2.set_ylabel('Severity Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Stability scores
        ax3.bar(methods, [reversal_results['stability'][m] for m in methods], color=self.colors[2])
        ax3.set_ylabel('Stability Score')
        ax3.tick_params(axis='x', rotation=45)
        
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

    def save_figure(self, fig, filename):
        """Save figure in PDF format with Nature-compatible settings"""
        fig.savefig(filename + '.pdf', 
                   format='pdf',
                   dpi=300, 
                   bbox_inches='tight',
                   transparent=True)
        plt.close(fig) 

    def plot_consolidated_weights(self, all_case_weights):
        """Plot consolidated weights for all cases in a single figure
        Args:
            all_case_weights: Dictionary with case_id as key and tuple of (criteria, weights) as value
        """
        plt.figure(figsize=(12, 8))
        
        n_cases = len(all_case_weights)
        bar_width = 0.8 / max(len(weights) for _, (_, weights) in all_case_weights.items())
        
        for case_idx, (case_id, (criteria, weights)) in enumerate(all_case_weights.items()):
            x = np.arange(len(weights))
            plt.bar(x + case_idx * bar_width, weights, 
                   bar_width, 
                   label=f'Case {case_id[-1]}',
                   color=self.colors[case_idx % len(self.colors)])
        
        plt.xlabel('Criteria Type')
        plt.ylabel('Weight')
        
        # Create common criteria categories
        criteria_categories = {
            'Economic': ['Investment Cost', 'Operational Cost', 'NPV', 'Payback Period', 
                        'Economic Synergy', 'Cost-Effectiveness', 'Consumer Benefits'],
            'Environmental': ['CO2 Reduction', 'Environmental Impact', 'Energy Efficiency', 
                            'External Energy Dependence'],
            'Social': ['Social Acceptance', 'Public Acceptance', 'Community Engagement', 
                      'Stakeholder Alignment'],
            'Technical': ['Land Use Compatibility', 'Ease of Implementation', 'Grid Dependence',
                         'Platform Adaptability', 'Contract Feasibility', 'Scalability',
                         'Regulatory Compliance']
        }
        
        # Set x-ticks at category positions
        plt.xticks(np.arange(4), list(criteria_categories.keys()), rotation=0)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf() 

    def plot_cross_method_correlation(self, method_rankings):
        """Plot correlation matrix between different MCDM methods
        Args:
            method_rankings: Dictionary with method names as keys and ranking arrays as values
        """
        # Convert rankings to numpy arrays
        methods = list(method_rankings.keys())
        rankings = np.array([method_rankings[m] for m in methods])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(rankings)
        
        plt.figure(figsize=(8, 6))
        im = sns.heatmap(corr_matrix,
                        xticklabels=methods,
                        yticklabels=methods,
                        annot=True,
                        fmt='.2f',
                        cmap='RdYlBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        square=True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add colorbar label
        cbar = im.collections[0].colorbar
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return plt.gcf() 