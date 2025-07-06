# MCDM Quick Reference Guide

## Method Selection Guide

| Method | Best For | Advantages | Limitations |
|--------|----------|------------|-------------|
| **SAW** | Simple problems, quantitative criteria | Easy to understand, fast computation | Linear compensation, normalization sensitive |
| **WPM** | Mixed units, multiplicative relationships | Dimensionless, stable results | Complex calculations, requires positive values |
| **TOPSIS** | Multiple quantitative criteria | Considers ideal solutions, robust | Requires quantitative data, complex interpretation |
| **AHP** | Qualitative criteria, stakeholder input | Handles qualitative data, consistency checks | Time-intensive, complex for large problems |

## Key Formulas

### Simple Additive Weighting (SAW)
```
Score_i = Σ(w_j × r_ij)
```
- Normalize: r_ij = x_ij / max(x_ij) for benefits
- Normalize: r_ij = min(x_ij) / x_ij for costs

### Weighted Product Model (WPM)
```
P(A_k/A_l) = Π((a_kj/a_lj)^w_j)
```
- If P(A_k/A_l) ≥ 1, then A_k is preferred over A_l

### TOPSIS
```
1. Normalize: r_ij = x_ij / √(Σx_kj²)
2. Weight: t_ij = r_ij × w_j
3. Ideal: A_best, A_worst
4. Distance: d_ib, d_iw
5. Score: s_iw = d_iw/(d_iw + d_ib)
```

### AHP
```
1. Pairwise comparisons (1-9 scale)
2. Calculate weights from eigenvectors
3. Check consistency: CR < 0.1
4. Synthesize: Score_i = Σ(w_j × w_ij)
```

## Common Renewable Energy Criteria

### Economic Criteria
- **Capital Cost (CAPEX)** - Initial investment required
- **Operating Cost (OPEX)** - Annual operating and maintenance costs
- **Levelized Cost of Energy (LCOE)** - Cost per unit of energy over lifetime
- **Payback Period** - Time to recover initial investment
- **Net Present Value (NPV)** - Present value of future cash flows
- **Job Creation** - Employment opportunities generated

### Environmental Criteria
- **CO₂ Emissions** - Greenhouse gas emissions per unit energy
- **Land Use** - Area required for installation
- **Water Consumption** - Water needed for operation
- **Waste Generation** - Solid waste produced
- **Biodiversity Impact** - Effect on local ecosystems
- **Resource Depletion** - Use of finite resources

### Technical Criteria
- **Efficiency** - Energy conversion efficiency
- **Reliability** - System availability and uptime
- **Capacity Factor** - Actual vs. theoretical output
- **Scalability** - Ability to expand system
- **Technical Maturity** - Technology readiness level
- **Grid Integration** - Ease of connecting to existing grid

### Social Criteria
- **Public Acceptance** - Community support level
- **Energy Security** - Independence from imports
- **Energy Access** - Ability to serve remote areas
- **Health Impact** - Effects on public health
- **Aesthetic Impact** - Visual and noise effects
- **Cultural Impact** - Effects on local traditions

## Typical Weight Ranges

Based on renewable energy literature:

| Stakeholder Type | Economic | Environmental | Technical | Social |
|------------------|----------|---------------|-----------|--------|
| **Government** | 25-35% | 30-40% | 20-30% | 10-20% |
| **Utility** | 40-50% | 15-25% | 25-35% | 5-15% |
| **Community** | 20-30% | 25-35% | 15-25% | 20-30% |
| **Environmental NGO** | 10-20% | 50-60% | 15-25% | 15-25% |

## Platform Quick Start

### Accessing Your Problem
1. Go to platform URL
2. Select "Examples" tab
3. Choose your assigned problem
4. Click "Load Example"

### Setting Up Analysis
1. Review alternatives and criteria
2. Adjust weights in sidebar
3. Verify criterion types (benefit/cost)
4. Select MCDM method

### Running Analysis
1. Click "Calculate Results"
2. Review rankings in Results tab
3. Examine step-by-step calculations
4. Check visualizations

### Sensitivity Analysis
1. Go to Visualizations tab
2. Use weight sensitivity sliders
3. Compare method results
4. Download results for records

## Troubleshooting

### Common Issues
- **Weights don't sum to 1.0:** Adjust in sidebar until sum = 1.0
- **Unexpected rankings:** Check criterion types (benefit vs. cost)
- **Method differences:** Normal - compare and discuss reasons
- **Missing data:** Use expert estimates or literature values

### Platform Problems
- **Page won't load:** Refresh browser, check internet connection
- **Calculations stuck:** Verify all data entered correctly
- **Export not working:** Try different browser or download format

## Presentation Checklist

### Content
- [ ] Problem context clearly explained
- [ ] Method selection justified
- [ ] Key assumptions documented
- [ ] Results clearly presented
- [ ] Sensitivity analysis discussed
- [ ] Practical recommendations provided

### Delivery
- [ ] Within 5-minute time limit
- [ ] Visual aids prepared (screenshots)
- [ ] All group members participate
- [ ] Prepared for questions
- [ ] Clear conclusion stated

## Key Learning Points

### MCDM Process
1. **Problem Definition** - Most critical step
2. **Method Selection** - Match method to problem type
3. **Data Quality** - Garbage in, garbage out
4. **Stakeholder Input** - Essential for acceptance
5. **Sensitivity Analysis** - Test robustness
6. **Communication** - Results must be understandable

### Renewable Energy Insights
- **No perfect solution** - All alternatives have trade-offs
- **Context matters** - Local conditions affect rankings
- **Stakeholder perspectives** - Different groups have different priorities
- **Uncertainty** - Energy planning involves long-term projections
- **Integration** - Consider system-wide effects

### Practical Applications
- **Decision support** - MCDM informs, doesn't replace judgment
- **Transparency** - Process should be open and documented
- **Iteration** - Refine analysis based on feedback
- **Implementation** - Consider practical constraints
- **Monitoring** - Track actual performance vs. predictions

## Additional Resources

### Platform Resources
- Method guides in "Learn" section
- Example problems for practice
- Mathematical formulations
- Best practices documentation

### External Resources
- Academic papers on MCDM applications
- Renewable energy databases
- Government energy planning documents
- International energy agency reports

### Software Alternatives
- Excel templates for simple analyses
- R packages: MCDM, ahp
- Python libraries: PyMCDA, scikit-criteria
- Commercial tools: Expert Choice, 1000minds

Remember: MCDM is a tool to structure thinking and facilitate discussion. The process of applying MCDM is often as valuable as the final results!
