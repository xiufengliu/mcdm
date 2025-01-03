import numpy as np
from .mcdm_analysis import MCDMAnalysis
from data.case_data import CASE_DATA

class CaseStudy:
    def __init__(self, case_id):
        self.case_id = case_id
        self.mcdm = MCDMAnalysis()
        self.data = CASE_DATA.get(case_id)
        
        if not self.data:
            raise ValueError(f"Case study {case_id} not found in data")
    
    def run_analysis(self):
        """Run the complete MCDM analysis for the case study"""
        print(f"\nCase {self.case_id}: {self.data['name']}")
        
        # Calculate weights using AHP
        weights, cr = self.mcdm.calculate_ahp_weights(self.data["comparison_matrix"])
        print(f"Consistency Ratio: {cr:.3f}")
        
        print("\nCriteria Weights:")
        for criterion, weight in zip(self.data["criteria"], weights):
            print(f"{criterion}: {weight:.3f}")
            
        # Run TOPSIS
        decision_matrix = np.array(self.data["decision_matrix"])
        rankings = self.mcdm.topsis(decision_matrix, weights, self.data["criteria_type"])
        
        print("\nTOPSIS Rankings:")
        ranked_options = sorted(zip(self.data["options"], rankings), 
                              key=lambda x: x[1], reverse=True)
        for option, score in ranked_options:
            print(f"{option}: {score:.3f}")
            
        # Sensitivity Analysis
        sensitivity = self.mcdm.sensitivity_analysis(
            decision_matrix, 
            weights,
            self.data["criteria_type"]
        )
        print("\nSensitivity Analysis:")
        for (criterion_idx, variation), new_ranking in sensitivity.items():
            print(f"Criterion {self.data['criteria'][criterion_idx]} ({variation*100:+}%): {new_ranking}")
        
        return {
            'weights': weights,
            'consistency_ratio': cr,
            'rankings': rankings,
            'sensitivity': sensitivity
        }