**Clinical background**
- Adult Acquired Flatfoot Deformity is a progressive condition where the medial longitudinal arch of the foot collapses, leading to deformity, pain and functional limitations.
- Diagnosing AAFD requires multiple radiographic and clinical measurements(calcaneal pitch, Meary’s angle, talocalcaneal angle, talar uncoverage angle, etc.).
- Currently, clinicians must manually measure all of these parameters, which is time-consuming and inefficient.
**The Gap / Problem**
- There is no single direct label in the dataset that says whether a patient has AAFD.
- Clinicians want a faster, more objective way to decide if a patient has flatfoot.
- Measuring _every_ parameter may not be necessary — maybe one or two dominant parameters can predict the diagnosis.
- Another challenge is **observer variation**:
	- **Inter-observer variation** → differences in measurements between different clinicians for the same patient.
	- **Intra-observer variation** → differences in measurements when the same clinician measures the same patient at different times.
    - This raises the question: which parameters are most reliable and consistent, regardless of who measures them or when?
**Research Objectives**
- We are trying to solve two key problems:
1. **Diagnosis Automation**
    - From a given patient’s radiographic measurements, determine whether the patient has AAFD (Yes/No).
    - This requires comparing measurements to established clinical thresholds and making a decision.
2. **Dominant Parameter Identification**
    - Among all the parameters, identify the single most decisive measurement** (or small subset) that can independently predict AAFD.
    - This helps clinicians focus on fewer measurements, saving time in practice.
**What we aim to solve**
- Provide a reliable, simplified diagnostic process for AAFD using radiographic measurements.
- Reduce the burden on clinicians by showing that one or two parameters (e.g., calcaneal pitch, Meary’s angle, or talar uncoverage angle) are sufficient in most cases.
- Address the issue of **observer variation** by identifying which parameters are least affected by intra- and inter-observer differences.
- Lay the groundwork for an automated decision support tool in clinics.