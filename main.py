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

# Add reference studies data
REFERENCE_STUDIES = {
    'Study1': {
        'name': 'Industrial Heat Recovery (2019)',
        'rankings': [0.82, 0.76, 0.65],
        'weights': [0.35, 0.25, 0.25, 0.15],
        'criteria': ['Economic', 'Technical', 'Environmental', 'Social']
    },
    'Study2': {
        'name': 'Waste Heat Integration (2020)',
        'rankings': [0.78, 0.72, 0.68],
        'weights': [0.30, 0.30, 0.25, 0.15],
        'criteria': ['Economic', 'Technical', 'Environmental', 'Social']
    }
    # Add more reference studies as needed
}

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
        
        # Perform cross-method analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rankings, correlation_matrix = sensitivity.cross_method_analysis(
                decision_matrix, weights, criteria_type
            )
        
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
        
        # Statistical tests
        statistical_results = sensitivity.statistical_tests(method_rankings)
        
        # Detailed rank reversal analysis
        reversal_results = sensitivity.detailed_rank_reversal(
            decision_matrix, weights, criteria_type
        )
        
        # Return comprehensive results
        return {
            'case_data': case_data,
            'rankings': method_rankings,
            'correlation_matrix': correlation_matrix,
            'method_stability': method_stability,
            'statistical_tests': statistical_results,
            'detailed_reversals': reversal_results,
            'weights': weights,
            'consistency_ratio': cr
        }
        
    except Exception as e:
        logger.error(f"Error analyzing case {case_id}: {str(e)}")
        raise

def format_results_for_publication(all_results):
    """Format all results into LaTeX tables"""
    # Table 1: Overview of Case Studies
    print("\n%% Table 1: Overview of Case Studies and Consistency Analysis")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Overview of Case Studies and Consistency Analysis}")
    print(r"\begin{tabular}{lccccl}")
    print(r"\hline")
    print(r"Case ID & Industry Sector & Alternatives & Criteria & CR & Decision Context \\")
    print(r"\hline")
    
    for case_id, results in all_results.items():
        case_data = results['case_data']
        print(f"{case_id} & {case_data['sector']} & {len(case_data['alternatives'])} & "
              f"{len(case_data['criteria'])} & {results['consistency_ratio']:.3f} & "
              f"{case_data['context']} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:overview}")
    print(r"\end{table}")
    
    # Table 2: Method Comparison Results
    print("\n%% Table 2: Method Comparison Results (Rankings)")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Method Comparison Results}")
    print(r"\begin{tabular}{llccccc}")
    print(r"\hline")
    print(r"Case & Alternative & TOPSIS & WSM & WPM & VIKOR & PROMETHEE \\")
    print(r"\hline")
    
    for case_id, results in all_results.items():
        rankings = results['rankings']
        alternatives = results['case_data']['alternatives']
        for i, alt in enumerate(alternatives):
            print(f"{case_id} & {alt} & {rankings['TOPSIS'][i]:.3f} & "
                  f"{rankings['WSM'][i]:.3f} & {rankings['WPM'][i]:.3f} & "
                  f"{rankings['VIKOR'][i]:.3f} & {rankings['PROMETHEE'][i]:.3f} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:rankings}")
    print(r"\end{table}")
    
    # Table 3: Method Stability Analysis
    print("\n%% Table 3: Method Stability Analysis")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Method Stability Analysis}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(r"Method & Rank Reversals & Avg Severity & Stability Score & Mean Correlation \\")
    print(r"\hline")
    
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
        
        print(f"{method} & {reversals} & {severity:.3f} & {stability:.3f} & {correlation:.3f} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:stability}")
    print(r"\end{table}")
    
    # Table 4: Statistical Test Results
    print("\n%% Table 4: Statistical Test Results")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Statistical Test Results}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(r"Test & Case 1 & Case 2 & Case 3 & Case 4 & Case 5 & Case 6 \\")
    print(r"\hline")
    
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
        print(f"{name} & {' & '.join(values)} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:statistics}")
    print(r"\end{table}")

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
        
        # Format and output results for publication
        format_results_for_publication(all_results)
        
        # Save all results to a file
        with open('results/experimental_results.tex', 'w') as f:
            format_results_for_publication(all_results)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
