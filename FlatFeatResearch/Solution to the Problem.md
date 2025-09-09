## The Big Picture

You want to:
1. **Check X-ray measurements** of feet and decide if someone has **radiographic AAFD** (the structural deformity).
2. **Check symptoms** (pain etc.) and decide if someone has **clinical AAFD** (symptomatic).
3. Find out which measurement(s) are **most trustworthy** (least variation between doctors) and **most useful** (best predictor of symptoms).

### Step by Step
#### 1. Clean and prepare the data
- Remove bad or missing values.
- Combine repeated measurements (take the **median** of 4 readings per variable → reduces errors). By taking the median, we cancel out the extreme and give a more stable, error-free number for analysis.
#### 2. Define **radiographic AAFD** (the “X-ray” diagnosis)
- Use rules like:
	- **Rule A:** If **even 1 angle is abnormal**, we call it AAFD. (More sensitive, catches more cases, but may over-diagnose).
	- **Rule B:** If **at least 2 angles are abnormal**, we call it AAFD. (Stricter, fewer false alarms, but may miss some cases).
- We will try both the definitions and see which one works better. 
#### 3. Define **clinical AAFD**
- Directly use the symptom column (`AS_0S_1`: 0 = no symptoms, 1 = symptoms).
#### 4. Check how reliable each variable is
- Compare the two doctors’ repeated measurements.
- We compare their readings:
	- **ICC (Intraclass Correlation Coefficient):**
		- Value between 0 and 1.
		- **1 = perfect agreement**, **0 = no agreement**.
- **Bland–Altman plot:**
	- A graph that shows the **difference between the two doctors’ measurements** against their average.
    - If differences are small and points are close to zero, it means good agreement.
#### 5. Normalize the values
- Put all variables on the same scale (z-scores) so they can be compared fairly.
#### 6. Explore the data
- Plot distributions(to see how each measurement is spread), correlations(Check if some measurements move together), scatterplots(Plot one variable against another → spot patterns or clusters.), etc.
- **Compare groups:** Separate patients into:
	- **X-ray AAFD (structural)** vs **No AAFD**
    - **Symptomatic (pain)** vs **No symptoms**  
    → See which measurements differ most between these groups.
#### 7. Test which variable links best with symptoms
- **Group comparison:**
    - Split patients into **symptomatic** vs **non-symptomatic**.
    - Compare each variable between the two groups.
    - Use:
        - **t-test** (if data is normal) - compares the means of the two groups
        - **Mann–Whitney test** (if not normal)  - compares the ranks of the two groups
            → This tells if the difference is **statistically significant**.
- **ROC curve (Receiver Operating Characteristic):**
    - For each variable, plot sensitivity vs 1-specificity. (true positives caught vs true negatives caught. lower the cutoff, sensitivity goes up, but specificity goes down). ROC curve show for all such cutoffs.
    - **AUC (Area Under Curve):**
        - **1.0 = perfect predictor**
        - **0.5 = useless (random guess)**  
            → Higher AUC = better at predicting symptoms.
#### 8. Build models
- Try **logistic regression** (simple yes/no predictor) with multiple variables.
- Report odds ratios and accuracy.
- Try Random Forest (optional) for variable importance. Helps to find out which measurements matter most for prediction. 
#### 9. Find the “dominant” variable
- Look at:
    1. Which variable is **most reliable** (highest ICC). ICC means how much two doctors agree on the measurement.
    2. Which variable is **most predictive** (highest AUC). Check **AUC** from ROC curves: how well the variable separates **symptomatic vs non-symptomatic**. High AUC = clinically useful.
- The best one (or two) is your **dominant parameter**. A variable with **high ICC + high AUC** is both **trustworthy and predictive**.
#### 10. Validation
##### Leave-One-Out Cross-Validation (LOOCV)
- Train the model on **28 feet**, test on the **1 left out**.
- Repeat this **29 times** (each foot gets tested once).
- Collect all predictions → compute performance.
##### What to Report
- **Accuracy** = overall % correct.
- **Sensitivity** = % of symptomatic correctly detected.
- **Specificity** = % of non-symptomatic correctly detected.
- **Confidence Intervals (CI)** = range showing uncertainty (important for small sample size).
#### 11. Handle imbalance
- A model can always guess the bigger group and still look "accurate". So, **accuracy hides poor performance on the smaller group**. That’s why we need sensitivity, specificity, AUC, or balanced accuracy.
#### 12. Acknowledge limitations
- Small sample, imbalanced groups.
- Need more patients for strong conclusions.
- This is a **pilot study**, basically a **small, preliminary study** done before a full-scale research project.