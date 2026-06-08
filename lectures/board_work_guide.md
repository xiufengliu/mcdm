# MCDM Course: Board Work Guide (现场板书指南)

## Purpose
This guide provides specific instructions for effective board work during the MCDM course, including what to write, when to write it, and how to organize the board space effectively.

## General Board Work Principles

### Board Organization
- **Left side:** Key concepts and definitions
- **Center:** Main calculations and examples  
- **Right side:** Summary points and next steps
- **Bottom:** Platform URL and important reminders

### Writing Guidelines
- **Use large, clear handwriting** - visible from back of room
- **Different colors for different concepts** (if available)
- **Leave space between sections** for clarity
- **Number steps clearly** (1, 2, 3...)
- **Box important formulas** and final answers

## Session 1: Board Work Plan

### Slide 3: MCDM Introduction
```
MCDM = Multiple Conflicting Goals

Car Example:
Low Cost ↔ High Safety
Fuel Efficiency ↔ Performance  
Reliability ↔ Style

Energy Example:
Cheap ↔ Clean
Reliable ↔ Renewable
Quick to build ↔ Long-term sustainable
```

### Slide 5: Austin Energy Example
```
Austin Energy Planning
• 500k people
• Need +320 MW by 2035
• Must reduce CO₂ by 50%

Options:
Solar:    $600M / 500MW
Wind:     $800M / 400MW  
Gas:      $400M / 320MW
Nuclear:  $8B / 1000MW

Poll Results:
Solar: [count] votes
Wind: [count] votes
Gas: [count] votes
Nuclear: [count] votes

→ Different priorities = Different choices
```

### Slide 10: Weighting Example
```
Stakeholder Weights:

Utility          Environmental    Community
Cost: 50%        Environment: 60%  Jobs: 40%
Reliability: 30% Cost: 20%        Cost: 30%
Environment: 20% Reliability: 20% Environment: 30%

Example Scores:
Cost: $1200/MW (actual data)
CO₂: 0.02 tons/MWh (measured)
Acceptance: 7/10 (survey)
Reliability: 35% (capacity factor)
```

## Session 2: Board Work Plan

### Slide 5: SAW Calculation Table
```
Step 1: Raw Data
Alternative | Cost ($/MW) | CO₂ (tons/MWh) | Reliability (%)
Solar       | 1200        | 0.02           | 25
Wind        | 1600        | 0.01           | 35  
Gas         | 800         | 0.35           | 85

Step 2: Normalized Values
Alternative | Cost        | CO₂            | Reliability
Solar       | 800/1200=0.67| 0.01/0.02=0.50| 25/85=0.29
Wind        | 800/1600=0.50| 0.01/0.01=1.00| 35/85=0.41
Gas         | 800/800=1.00 | 0.01/0.35=0.03| 85/85=1.00

Weights: Cost=0.3, CO₂=0.4, Reliability=0.3

Step 3: Final Scores
Solar: 0.67×0.3 + 0.50×0.4 + 0.29×0.3
     = 0.20 + 0.20 + 0.09 = 0.49

Wind:  0.50×0.3 + 1.00×0.4 + 0.41×0.3  
     = 0.15 + 0.40 + 0.12 = 0.67

Gas:   1.00×0.3 + 0.03×0.4 + 1.00×0.3
     = 0.30 + 0.01 + 0.30 = 0.61

Final Ranking:
1. Wind (0.67)
2. Gas (0.61)  
3. Solar (0.49)
```

### Slide 7: SAW Advantages/Limitations
```
SAW Method

Advantages:                 Limitations:
• Simple & intuitive       • Linear compensation
• Easy to explain          • Normalization sensitive  
• Computationally fast     • Assumes independence
• Widely accepted          • Rank reversal possible
• Good for screening       • May miss complex preferences

Best for: Simple problems, quantitative data, education
Avoid for: Interdependent criteria, qualitative factors
```

## Session 3: Board Work Plan

### Slide 7: Platform Information
```
MCDM Learning Platform
URL: [write clearly and large]

Features:
✓ All 4 methods (SAW, WPM, TOPSIS, AHP)
✓ Built-in examples
✓ Step-by-step explanations
✓ Sensitivity analysis
✓ Download results

Backup: Excel templates available
```

### Slide 10: Group Work Organization
```
Group Problems:
1. Regional Energy (City)
2. Campus Energy (University)
3. Rural Electrification  
4. Industrial Energy
5. Residential Systems
6. Community Projects

Interest Poll:
Problem 1: [count] people
Problem 2: [count] people
Problem 3: [count] people
Problem 4: [count] people
Problem 5: [count] people
Problem 6: [count] people

Schedule:
13:00-13:30  Group formation
13:30-15:00  MCDM analysis
15:00-15:15  Coffee break
15:15-16:30  Sensitivity analysis
16:30-17:00  Presentations
```

## Interactive Board Work Techniques

### Student Participation in Calculations
1. **Write the setup** - put the problem on board
2. **Ask for input** - "What's 800 divided by 1200?"
3. **Write student answer** - show their work
4. **Confirm or correct** - guide to right answer
5. **Move to next step** - keep momentum

### Building Tables Collaboratively
1. **Draw empty table** first
2. **Fill headers** with student input
3. **Add data row by row** with participation
4. **Calculate together** step by step
5. **Highlight final results** clearly

### Managing Board Space
- **Erase strategically** - keep key info visible
- **Use arrows** to show connections
- **Number sections** for easy reference
- **Save space** for student contributions
- **Take photos** of completed work if needed

## Specific Calculation Layouts

### SAW Normalization Formula
```
For BENEFIT criteria (higher = better):
normalized = actual value / maximum value

For COST criteria (lower = better):  
normalized = minimum value / actual value

Example:
Cost (COST): min=800, Solar=1200
→ normalized = 800/1200 = 0.67
```

### Weight Application
```
Final Score = Σ(weight × normalized value)

Solar Score:
= (Cost weight × Cost normalized) 
+ (CO₂ weight × CO₂ normalized)
+ (Reliability weight × Reliability normalized)
= (0.3 × 0.67) + (0.4 × 0.50) + (0.3 × 0.29)
= 0.20 + 0.20 + 0.09
= 0.49
```

## Board Work Timing

### Session 1 (60 minutes total)
- Slide 3: 2 minutes board work
- Slide 5: 3 minutes board work  
- Slide 10: 2 minutes board work
- **Total board time: 7 minutes**

### Session 2 (60 minutes total)
- Slide 5: 8 minutes board work (main calculation)
- Slide 6: 2 minutes board work (completion)
- Slide 7: 1 minute board work (summary)
- **Total board time: 11 minutes**

### Session 3 (60 minutes total)
- Slide 7: 1 minute board work (URL)
- Slide 10: 2 minutes board work (groups)
- **Total board time: 3 minutes**

## Emergency Board Work

### If Running Out of Space
1. **Erase oldest content** first
2. **Take photo** before erasing important work
3. **Use abbreviations** for repeated terms
4. **Write smaller** but still legible
5. **Use multiple boards** if available

### If Marker Runs Out
1. **Have backup markers** ready
2. **Ask students** if they have markers
3. **Use chalk** if whiteboard markers fail
4. **Continue verbally** while getting supplies
5. **Use projector** as backup

### If Students Can't See
1. **Write larger** immediately
2. **Move closer** to front
3. **Repeat verbally** what you write
4. **Ask students** to move forward
5. **Use projector** if available

## Board Work Best Practices

### Before Class
- [ ] Test all markers/chalk
- [ ] Clean board thoroughly
- [ ] Plan board layout
- [ ] Have backup supplies
- [ ] Check visibility from back row

### During Class
- [ ] Face students when talking
- [ ] Write clearly and large
- [ ] Use consistent organization
- [ ] Involve students in writing
- [ ] Take photos of key work

### After Each Section
- [ ] Highlight key results
- [ ] Check student understanding
- [ ] Erase non-essential content
- [ ] Prepare space for next section
- [ ] Save important formulas

This board work guide ensures that the visual component of the course reinforces the verbal instruction and provides students with clear, organized information they can reference throughout the session.
