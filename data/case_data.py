CASE_DATA = {
    "case1": {
        "name": "Cement Production",
        "options": ["Internal Process Heat Recovery", "Alternative Fuel Drying", "District Heating"],
        "criteria": ["Investment Cost", "Operational Cost", "CO2 Reduction", "Social Acceptance"],
        "criteria_type": ["min", "min", "max", "max"],
        "comparison_matrix": [
            [1, 2, 1/7, 3],
            [1/2, 1, 1/9, 2],
            [7, 9, 1, 5],
            [1/3, 1/2, 1/5, 1]
        ],
        "decision_matrix": [
            [0.757, 0.930, 0.000, 1.000],
            [0.756, 0.874, 0.000, 0.616],
            [0.697, 0.744, 0.000, 0.000]
        ]
    },
    "case2": {
        "name": "Metal Casting",
        "options": ["Electricity Generation", "Heat-to-Fuel Conversion", "Process Integration"],
        "criteria": ["NPV", "Payback Period", "Environmental Impact", "Land Use Compatibility"],
        "criteria_type": ["max", "min", "max", "max"],
        "comparison_matrix": [
            [1, 5, 3, 3],
            [1/5, 1, 1/2, 1],
            [1/3, 2, 1, 2],
            [1/3, 1, 1/2, 1]
        ],
        "decision_matrix": [
            [0.787, 0.801, 0.764, 0.639],
            [0.754, 0.763, 0.711, 0.000],
            [0.793, 0.867, 0.805, 1.000]
        ]
    },
    "case3": {
        "name": "Industrial Park",
        "options": ["Inter-industry Heat Exchange", "Integration with DH Network", "Individual Solutions"],
        "criteria": ["Economic Synergy", "Energy Efficiency", "External Energy Dependence", "Ease of Implementation"],
        "criteria_type": ["max", "max", "max", "max"],
        "comparison_matrix": [
            [1, 3, 2, 1/2],
            [1/3, 1, 2, 1/3],
            [1/2, 1/2, 1, 1/5],
            [2, 3, 5, 1]
        ],
        "decision_matrix": [
            [0.90, 0.75, 0.80, 0.70],
            [0.60, 0.85, 0.90, 0.60],
            [0.40, 0.50, 0.60, 0.80]
        ]
    },
    "case4": {
        "name": "District Heating",
        "options": ["DH Network Integration", "Process Steam Supply"],
        "criteria": ["Investment Cost", "Operational Cost", "CO2 Reduction", "Public Acceptance", "Regulatory Compliance"],
        "criteria_type": ["min", "min", "max", "max", "max"],
        "comparison_matrix": [
            [1, 2, 1/3, 1, 1/4],
            [1/2, 1, 1/4, 1/2, 1/5],
            [3, 4, 1, 2, 1],
            [1, 2, 1/2, 1, 1/3],
            [4, 5, 1, 3, 1]
        ],
        "decision_matrix": [
            [0.70, 0.80, 0.90, 0.80, 1.00],
            [0.90, 0.90, 0.60, 0.70, 1.00]
        ]
    },
    "case5": {
        "name": "Multi-Sectoral District Heating",
        "options": ["DH Expansion with Industrial Heat", "Individual Industry Solutions"],
        "criteria": ["Cost-Effectiveness", "Contract Feasibility", "Scalability", "Stakeholder Alignment"],
        "criteria_type": ["max", "max", "max", "max"],
        "comparison_matrix": [
            [1, 2, 3, 2],
            [1/2, 1, 2, 1],
            [1/3, 1/2, 1, 1/2],
            [1/2, 1, 2, 1]
        ],
        "decision_matrix": [
            [0.80, 0.70, 0.90, 0.85],
            [0.95, 0.90, 0.60, 0.60]
        ]
    },
    "case6": {
        "name": "Residential DH with P2P",
        "options": ["P2P with CHP Integration", "DH without CHP"],
        "criteria": ["Consumer Benefits", "Grid Dependence", "Community Engagement", "Platform Adaptability"],
        "criteria_type": ["max", "min", "max", "max"],
        "comparison_matrix": [
            [1, 2, 3, 2],
            [1/2, 1, 2, 1],
            [1/3, 1/2, 1, 1/2],
            [1/2, 1, 2, 1]
        ],
        "decision_matrix": [
            [0.80, 0.75, 0.85, 0.90],
            [0.60, 0.60, 0.70, 0.75]
        ]
    }
}
