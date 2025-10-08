# Adult-Acquired-Flatfoot-Deformity

This project analyzes **Adult Acquired Flatfoot Deformity (AAFD)** using radiographic measurements and clinical symptoms. It identifies which X-ray angles are most **reliable** between observers and most **predictive** of symptoms.

## Features
- Computes **radiographic labels** based on angle thresholds (Rule A: sensitive, Rule B: stricter).  
- Calculates **deviation-from-normal** and **standardized Z-scores** for each measurement.  
- Evaluates **observer reliability** using ICC (Intraclass Correlation Coefficient).  
- Explores which radiographic parameters best predict **clinical AAFD**.  
- Provides a **simple decision rule** for quick clinical assessment.

## Technologies
- Python (pandas, numpy, scikit-learn, pingouin, matplotlib/seaborn)

## Usage
1. Load your dataset (`csv` or `excel`).  
2. Run the analysis scripts to compute labels, deviations, and ICCs.  
3. Visualize results and predictive performance.  

## Goal
To provide a **reproducible pipeline** for assessing AAFD from X-ray measurements and identifying the most important radiographic indicators.  
This project is intended for **researchers and clinicians** to better understand and predict foot deformities.

## Initial Analysis
The folder named initial contains the initial analysis performed during the early exploration phase of the project. It includes preliminary data checks, baseline summaries, and first-round insights before the main analysis.

## Second Analysis
Second analysis takes the idea of removing the outliers which were obtained in the Bland–Altman plot.

## Third Analysis
Third analysis takes the idea of working with the infra and intra measurements measured among the two professors at two different times. 