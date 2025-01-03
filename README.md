# Multi-Criteria Decision Making (MCDM) for Industrial Excess Heat Utilization

## Overview
This repository contains the implementation and analysis of multiple MCDM methods for evaluating industrial excess heat utilization options. The project analyzes six diverse industrial cases using five different MCDM methods (TOPSIS, WSM, WPM, VIKOR, and PROMETHEE) to provide robust decision support for industrial excess heat recovery projects.

## Key Features
- **Multiple MCDM Methods**: Implementation of five widely-used MCDM methods:
  - **TOPSIS**: Technique for Order Preference by Similarity to Ideal Solution
  - **WSM**: Weighted Sum Model
  - **WPM**: Weighted Product Model
  - **VIKOR**: VIseKriterijumska Optimizacija I Kompromisno Resenje
  - **PROMETHEE**: Preference Ranking Organization Method for Enrichment Evaluations

- **Comprehensive Case Studies**:
  1. Cement Production (Portugal)
  2. Metal Casting (UK)
  3. Industrial Park Integration (Greece)
  4. District Heating Network (Portugal)
  5. Multi-Sectoral District Heating (Sweden)
  6. Residential District Heating with P2P (Denmark)

- **Analysis Capabilities**:
  - Criteria weight determination using AHP
  - Sensitivity analysis
  - Rank reversal detection
  - Statistical validation
  - Performance visualization

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mcdm.git

# Navigate to the project directory
cd mcdm

# Install required packages
pip install -r requirements.txt
```

## Usage
### Basic Example
```python
from mcdm.methods import TOPSIS
from mcdm.data_processing import load_case_study

# Load case study data
case_data = load_case_study('case1')

# Initialize and run TOPSIS
topsis = TOPSIS(case_data)
rankings = topsis.calculate_rankings()
```

### Running Analysis
```python
from mcdm.analysis import MCDMAnalysis

# Initialize analysis
analysis = MCDMAnalysis(case_data)

# Run all methods
results = analysis.run_all_methods()

# Generate visualizations
analysis.plot_rankings()
analysis.plot_sensitivity()
```

## Project Structure
```plaintext
mcdm/
├── data/           # Case study data and results
├── src/            # Source code
│   ├── methods/    # MCDM method implementations
│   ├── analysis/   # Analysis tools
│   └── visualization/ # Visualization functions
├── tests/          # Unit tests
└── docs/           # Documentation
```

## Case Studies Overview
| Case Study | Industry Sector | Alternatives | Criteria | Key Features               |
|------------|------------------|--------------|----------|----------------------------|
| Case 1     | Cement           | 3            | 4        | High-temperature recovery  |
| Case 2     | Metal Casting    | 3            | 4        | Process integration focus  |
| Case 3     | Industrial Park  | 3            | 4        | Multi-stakeholder context  |
| Case 4     | District Heat    | 2            | 5        | Public infrastructure      |
| Case 5     | Multi-DH         | 2            | 4        | Cross-sector integration   |
| Case 6     | Food Process     | 3            | 4        | P2P energy trading         |

## Key Findings
- **TOPSIS** demonstrated highest stability (0.991) across all cases
- Strong correlation between **TOPSIS** and **WSM** (ρ = 0.92)
- **VIKOR** showed increased sensitivity to weight variations
- Statistical tests confirmed method consistency (Kendall's W = 1.000)

## Documentation
Detailed documentation is available in the `docs` directory:
- Method descriptions and mathematical formulations
- Case study details and data sources
- Analysis procedures and validation methods
- Visualization tools and examples

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Citation
Please cite:
```bibtex

```


## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## Contact
- **Author**: [Xiufeng Liu]  
- **Email**: [xiuli@dtu.dk]  
- **Project Link**: [https://github.com/xiufengliu/mcdm](https://github.com/xiufengliu/mcdm)

## Acknowledgments
This research was supported in part by the RE-INTEGRATE project (no. 101118217) funded by the European Union Horizon 2020 research and innovation programme and by the European Union's Horizon 2020 through the EU Framework Program for Research and Innovation, within the EMB3Rs project under agreement no. 847121.
```

This markdown file is clean and well-structured, ready to be used as a `README.md` for your project. Let me know if you need further adjustments!