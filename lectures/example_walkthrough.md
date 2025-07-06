# MCDM Example Walkthrough: Campus Energy Planning

This detailed example shows how to apply MCDM methods step-by-step to a real renewable energy problem. Use this as a reference during your group work.

## Problem Description

**Context:** State University wants to reduce its energy costs and carbon footprint. The campus currently spends $2.5 million annually on electricity and has committed to carbon neutrality by 2030.

**Decision:** Choose the best energy solution for the campus's unique needs.

## Step 1: Define Alternatives

After initial screening, five alternatives remain:

1. **Rooftop Solar System** - Install solar panels on campus buildings
2. **Small Wind Turbines** - Install wind turbines on campus grounds  
3. **Geothermal Heat Pumps** - Use ground-source heat pumps for heating/cooling
4. **Combined Heat & Power** - Install natural gas CHP system
5. **Energy Efficiency Upgrades** - Comprehensive building efficiency improvements

## Step 2: Establish Criteria

Based on stakeholder input, five criteria are most important:

1. **Initial Cost** ($) - Capital investment required [COST - lower is better]
2. **Annual Savings** ($/year) - Operating cost reduction [BENEFIT - higher is better]
3. **Carbon Reduction** (tons CO₂/year) - Emissions reduction [BENEFIT - higher is better]
4. **Maintenance Requirements** (hours/year) - Staff time needed [COST - lower is better]
5. **Educational Value** (1-10 scale) - Learning opportunities for students [BENEFIT - higher is better]

## Step 3: Collect Data

| Alternative | Initial Cost | Annual Savings | Carbon Reduction | Maintenance | Educational Value |
|-------------|--------------|----------------|------------------|-------------|-------------------|
| Rooftop Solar | $800,000 | $120,000 | 450 | 40 | 9 |
| Wind Turbines | $600,000 | $80,000 | 300 | 60 | 8 |
| Geothermal | $1,200,000 | $150,000 | 500 | 20 | 6 |
| CHP System | $900,000 | $200,000 | 200 | 80 | 4 |
| Efficiency | $400,000 | $100,000 | 350 | 10 | 7 |

## Step 4: Determine Weights

**Stakeholder:** University administration prioritizing financial sustainability with environmental responsibility.

**Weights determined through group discussion:**
- Initial Cost: 0.25 (25%) - Important but not dominant
- Annual Savings: 0.30 (30%) - Most important for budget
- Carbon Reduction: 0.25 (25%) - Critical for sustainability goals
- Maintenance: 0.10 (10%) - Manageable with current staff
- Educational Value: 0.10 (10%) - Nice bonus but not primary goal

**Check:** 0.25 + 0.30 + 0.25 + 0.10 + 0.10 = 1.00 ✓

## Step 5: Apply SAW Method

### Step 5a: Normalize the Data

For **cost criteria** (Initial Cost, Maintenance): normalized = min(values) / actual_value
For **benefit criteria** (Annual Savings, Carbon Reduction, Educational Value): normalized = actual_value / max(values)

**Initial Cost normalization:** min = $400,000
- Rooftop Solar: 400,000 / 800,000 = 0.50
- Wind Turbines: 400,000 / 600,000 = 0.67
- Geothermal: 400,000 / 1,200,000 = 0.33
- CHP System: 400,000 / 900,000 = 0.44
- Efficiency: 400,000 / 400,000 = 1.00

**Annual Savings normalization:** max = $200,000
- Rooftop Solar: 120,000 / 200,000 = 0.60
- Wind Turbines: 80,000 / 200,000 = 0.40
- Geothermal: 150,000 / 200,000 = 0.75
- CHP System: 200,000 / 200,000 = 1.00
- Efficiency: 100,000 / 200,000 = 0.50

**Carbon Reduction normalization:** max = 500 tons
- Rooftop Solar: 450 / 500 = 0.90
- Wind Turbines: 300 / 500 = 0.60
- Geothermal: 500 / 500 = 1.00
- CHP System: 200 / 500 = 0.40
- Efficiency: 350 / 500 = 0.70

**Maintenance normalization:** min = 10 hours
- Rooftop Solar: 10 / 40 = 0.25
- Wind Turbines: 10 / 60 = 0.17
- Geothermal: 10 / 20 = 0.50
- CHP System: 10 / 80 = 0.13
- Efficiency: 10 / 10 = 1.00

**Educational Value normalization:** max = 9
- Rooftop Solar: 9 / 9 = 1.00
- Wind Turbines: 8 / 9 = 0.89
- Geothermal: 6 / 9 = 0.67
- CHP System: 4 / 9 = 0.44
- Efficiency: 7 / 9 = 0.78

### Step 5b: Apply Weights and Calculate Scores

**SAW Formula:** Score = Σ(weight × normalized_value)

**Rooftop Solar:**
Score = 0.25×0.50 + 0.30×0.60 + 0.25×0.90 + 0.10×0.25 + 0.10×1.00
Score = 0.125 + 0.180 + 0.225 + 0.025 + 0.100 = **0.655**

**Wind Turbines:**
Score = 0.25×0.67 + 0.30×0.40 + 0.25×0.60 + 0.10×0.17 + 0.10×0.89
Score = 0.168 + 0.120 + 0.150 + 0.017 + 0.089 = **0.544**

**Geothermal:**
Score = 0.25×0.33 + 0.30×0.75 + 0.25×1.00 + 0.10×0.50 + 0.10×0.67
Score = 0.083 + 0.225 + 0.250 + 0.050 + 0.067 = **0.675**

**CHP System:**
Score = 0.25×0.44 + 0.30×1.00 + 0.25×0.40 + 0.10×0.13 + 0.10×0.44
Score = 0.110 + 0.300 + 0.100 + 0.013 + 0.044 = **0.567**

**Efficiency:**
Score = 0.25×1.00 + 0.30×0.50 + 0.25×0.70 + 0.10×1.00 + 0.10×0.78
Score = 0.250 + 0.150 + 0.175 + 0.100 + 0.078 = **0.753**

### Step 5c: SAW Results

**Final SAW Ranking:**
1. **Energy Efficiency Upgrades** (0.753) - Winner!
2. **Geothermal Heat Pumps** (0.675)
3. **Rooftop Solar System** (0.655)
4. **CHP System** (0.567)
5. **Wind Turbines** (0.544)

## Step 6: Sensitivity Analysis

**What if environmental concerns were more important?**

Change weights to: Cost=0.15, Savings=0.20, Carbon=0.40, Maintenance=0.15, Education=0.10

**Recalculated scores:**
- Energy Efficiency: 0.698
- **Geothermal: 0.725** (now wins!)
- Rooftop Solar: 0.693
- CHP System: 0.487
- Wind Turbines: 0.527

**Insight:** Geothermal becomes the best choice when carbon reduction is prioritized more heavily.

## Step 7: Interpretation and Recommendations

### Key Findings:
1. **Energy Efficiency Upgrades** consistently rank highest due to excellent cost-effectiveness
2. **Geothermal** performs well when environmental impact is prioritized
3. **Wind Turbines** rank lowest due to moderate performance across all criteria
4. **CHP System** has excellent savings but poor environmental performance

### Practical Recommendations:
1. **Primary recommendation:** Implement Energy Efficiency Upgrades first
2. **Secondary phase:** Add Geothermal systems for high-impact buildings
3. **Consider:** Rooftop Solar for visible sustainability demonstration
4. **Avoid:** Wind turbines due to campus constraints and moderate performance

### Implementation Strategy:
- **Year 1:** Energy efficiency upgrades (quick wins, low cost)
- **Year 2-3:** Geothermal for major buildings (high impact)
- **Year 4-5:** Solar panels for educational and demonstration value

## Key Learning Points

1. **Method matters:** Different MCDM methods might give different rankings
2. **Weights are critical:** Small changes in weights can change the winner
3. **Data quality:** Accurate data is essential for meaningful results
4. **Stakeholder input:** Weights should reflect real stakeholder priorities
5. **Sensitivity analysis:** Always test how robust your conclusions are
6. **Implementation:** Consider practical constraints and phased approaches

This example demonstrates how MCDM provides a systematic, transparent way to make complex energy decisions while considering multiple objectives and stakeholder priorities.
