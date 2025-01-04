from src.case_study import CaseStudy
from src.visualization import MCDMVisualizer
from src.statistics import MCDMStatistics
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
from src.mcdm_methods import MCDMethods
from src.method_sensitivity import MethodSensitivityAnalysis
import seaborn as sns
from data.case_data import CASE_DATA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'mcdm_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_case(case_id, visualizer):
    try:
        case = CaseStudy(case_id)
        mcdm = MCDMethods()
        sensitivity = MethodSensitivityAnalysis(mcdm)
        
        # Get base data
        decision_matrix = np.array(case.data['decision_matrix'])
        weights, cr = mcdm.ahp(case.data['comparison_matrix'])
        criteria_type = case.data['criteria_type']
        
        # Prepare case metadata
        case_data = {
            'sector': {
                'case1': 'Cement',
                'case2': 'Metal Casting',
                'case3': 'Industrial Park',
                'case4': 'District Heat',
                'case5': 'Multi-DH',
                'case6': 'Food Process'
            }[case_id],
            'alternatives': [f'A{i+1}' for i in range(len(decision_matrix))],
            'criteria': [f'C{i+1}' for i in range(len(weights))],
            'context': {
                'case1': 'Heat Recovery',
                'case2': 'Process Design',
                'case3': 'Integration',
                'case4': 'Network Design',
                'case5': 'Cross-sector',
                'case6': 'Energy Efficiency'
            }[case_id]
        }
        
        # Calculate rankings using all methods
        method_rankings = {}
        method_rankings['TOPSIS'] = mcdm.topsis(decision_matrix, weights, criteria_type)
        method_rankings['WSM'] = mcdm.wsm(decision_matrix, weights, criteria_type)
        method_rankings['WPM'] = mcdm.wpm(decision_matrix, weights, criteria_type)
        method_rankings['VIKOR'] = mcdm.vikor(decision_matrix, weights, criteria_type)
        method_rankings['PROMETHEE'] = mcdm.promethee(decision_matrix, weights, criteria_type)
        
        # Generate and save figures
        # Method comparison
        fig = visualizer.plot_method_comparison(method_rankings, case_id)
        visualizer.save_figure(fig, f'results/{case_id}_method_comparison')
        
        # Weights visualization
        fig = visualizer.plot_weights(case_data['criteria'], weights, case_id)
        visualizer.save_figure(fig, f'results/{case_id}_weights')
        
        # Perform cross-method analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rankings, correlation_matrix = sensitivity.cross_method_analysis(
                decision_matrix, weights, criteria_type
            )
            
        # Generate correlation plot
        fig = visualizer.plot_correlation_matrix(decision_matrix, case_data['criteria'], case_id)
        visualizer.save_figure(fig, f'results/{case_id}_correlation')
        
        # Perform sensitivity analysis
        sensitivity_data = {}
        variations = np.arange(-0.2, 0.21, 0.05)  # -20% to +20% in 5% steps
        for i, criterion in enumerate(case_data['criteria']):
            for variation in variations:
                modified_weights = weights.copy()
                modified_weights[i] *= (1 + variation)
                modified_weights /= modified_weights.sum()  # Renormalize
                rankings = mcdm.topsis(decision_matrix, modified_weights, criteria_type)
                sensitivity_data[(i, variation)] = rankings
                
        # Generate sensitivity plot
        fig = visualizer.plot_sensitivity(sensitivity_data, case_data['criteria'], case_id)
        visualizer.save_figure(fig, f'results/{case_id}_sensitivity')
        
        # Perform method-specific sensitivity analysis
        method_stability = {}
        for method_name in ['TOPSIS', 'WSM', 'WPM', 'VIKOR', 'PROMETHEE']:
            stability_results, base_ranking = sensitivity.analyze_method_stability(
                decision_matrix, weights, criteria_type, method_name
            )
            method_stability[method_name] = {
                'stability_results': stability_results,
                'base_ranking': base_ranking,
                'stability_score': np.mean([1 - np.std(list(stability_results.values()), axis=0)])
            }
        
        # Generate stability plot
        fig = visualizer.plot_stability_analysis(method_stability, case_id)
        visualizer.save_figure(fig, f'results/{case_id}_stability')
        
        # Statistical tests
        statistical_results = sensitivity.statistical_tests(method_rankings)
        
        # Generate statistical tests plot
        fig = visualizer.plot_statistical_tests(statistical_results, case_id)
        visualizer.save_figure(fig, f'results/{case_id}_statistical_tests')
        
        # Detailed rank reversal analysis
        reversal_results = sensitivity.detailed_rank_reversal(
            decision_matrix, weights, criteria_type
        )
        
        # Generate rank reversals plot
        fig = visualizer.plot_rank_reversals(reversal_results, case_id)
        visualizer.save_figure(fig, f'results/{case_id}_rank_reversals')
        
        # Return comprehensive results
        return {
            'case_data': case_data,
            'rankings': method_rankings,
            'correlation_matrix': correlation_matrix,
            'method_stability': method_stability,
            'statistical_tests': statistical_results,
            'detailed_reversals': reversal_results,
            'weights': weights,
            'consistency_ratio': cr,
            'sensitivity_data': sensitivity_data
        }
        
    except Exception as e:
        logger.error(f"Error analyzing case {case_id}: {str(e)}")
        raise

def format_results_for_publication(all_results, file=None):
    """Format all results into LaTeX tables
    Args:
        all_results: Dictionary containing all analysis results
        file: Optional file object to write results to. If None, prints to stdout
    """
    def write(text):
        if file:
            print(text, file=file)
        else:
            print(text)
            
    # Table 1: Overview of Case Studies
    write("\n%% Table 1: Overview of Case Studies and Consistency Analysis")
    write(r"\begin{table}[htbp]")
    write(r"\centering")
    write(r"\caption{Overview of Case Studies and Consistency Analysis}")
    write(r"\begin{tabular}{lccccl}")
    write(r"\hline")
    write(r"Case ID & Industry Sector & Alternatives & Criteria & CR & Decision Context \\")
    write(r"\hline")
    
    for case_id, results in all_results.items():
        case_data = results['case_data']
        write(f"{case_id} & {case_data['sector']} & {len(case_data['alternatives'])} & "
              f"{len(case_data['criteria'])} & {results['consistency_ratio']:.3f} & "
              f"{case_data['context']} \\\\")
    
    write(r"\hline")
    write(r"\end{tabular}")
    write(r"\label{tab:overview}")
    write(r"\end{table}")
    
    # Table 2: Method Comparison Results
    write("\n%% Table 2: Method Comparison Results (Rankings)")
    write(r"\begin{table}[htbp]")
    write(r"\centering")
    write(r"\caption{Method Comparison Results}")
    write(r"\begin{tabular}{llccccc}")
    write(r"\hline")
    write(r"Case & Alternative & TOPSIS & WSM & WPM & VIKOR & PROMETHEE \\")
    write(r"\hline")
    
    for case_id, results in all_results.items():
        rankings = results['rankings']
        alternatives = results['case_data']['alternatives']
        for i, alt in enumerate(alternatives):
            write(f"{case_id} & {alt} & {rankings['TOPSIS'][i]:.3f} & "
                  f"{rankings['WSM'][i]:.3f} & {rankings['WPM'][i]:.3f} & "
                  f"{rankings['VIKOR'][i]:.3f} & {rankings['PROMETHEE'][i]:.3f} \\\\")
    
    write(r"\hline")
    write(r"\end{tabular}")
    write(r"\label{tab:rankings}")
    write(r"\end{table}")
    
    # Table 3: Method Stability Analysis
    write("\n%% Table 3: Method Stability Analysis")
    write(r"\begin{table}[htbp]")
    write(r"\centering")
    write(r"\caption{Method Stability Analysis}")
    write(r"\begin{tabular}{lccccc}")
    write(r"\hline")
    write(r"Method & Rank Reversals & Avg Severity & Stability Score & Mean Correlation \\")
    write(r"\hline")
    
    methods = ['TOPSIS', 'WSM', 'WPM', 'VIKOR', 'PROMETHEE']
    for method in methods:
        reversals = sum(res['detailed_reversals']['frequency'][method.lower()] 
                       for res in all_results.values())
        severity = np.mean([res['detailed_reversals']['severity'][method.lower()] 
                          for res in all_results.values()])
        stability = np.mean([res['method_stability'][method]['stability_score'] 
                           for res in all_results.values()])
        correlation = np.mean([res['correlation_matrix'][methods.index(method)] 
                             for res in all_results.values()])
        
        write(f"{method} & {reversals} & {severity:.3f} & {stability:.3f} & {correlation:.3f} \\\\")
    
    write(r"\hline")
    write(r"\end{tabular}")
    write(r"\label{tab:stability}")
    write(r"\end{table}")
    
    # Table 4: Statistical Test Results
    write("\n%% Table 4: Statistical Test Results")
    write(r"\begin{table}[htbp]")
    write(r"\centering")
    write(r"\caption{Statistical Test Results}")
    write(r"\begin{tabular}{lcccccc}")
    write(r"\hline")
    write(r"Test & Case 1 & Case 2 & Case 3 & Case 4 & Case 5 & Case 6 \\")
    write(r"\hline")
    
    tests = ['friedman_test', 'kendall_w', 'kruskal_wallis']
    test_names = ['Friedman p-value', "Kendall's W", 'Kruskal-Wallis p']
    
    for test, name in zip(tests, test_names):
        values = []
        for case_id in sorted(all_results.keys()):
            if test == 'kendall_w':
                val = all_results[case_id]['statistical_tests'][test]['w']
            else:
                val = all_results[case_id]['statistical_tests'][test]['p_value']
            values.append(f"{val:.3f}")
        write(f"{name} & {' & '.join(values)} \\\\")
    
    write(r"\hline")
    write(r"\end{tabular}")
    write(r"\label{tab:statistics}")
    write(r"\end{table}")

def main():
    try:
        logger.info("Starting MCDM Analysis")
        logger.info("=" * 50)
        
        visualizer = MCDMVisualizer()
        all_results = {}
        
        # Analyze all cases
        for case_id in [f"case{i}" for i in range(1, 7)]:
            logger.info(f"\nAnalyzing {case_id}")
            logger.info("=" * 50)
            
            results = analyze_case(case_id, visualizer)
            all_results[case_id] = results
            
            # Log key results for verification
            logger.info(f"Case {case_id} analysis complete:")
            logger.info(f"- Consistency ratio: {results['consistency_ratio']:.3f}")
            logger.info(f"- Number of alternatives: {len(results['rankings']['TOPSIS'])}")
        
        # Save results to file only
        output_file = 'results/experimental_results.tex'
        logger.info(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            format_results_for_publication(all_results, file=f)
            
        logger.info(f"Analysis complete. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
