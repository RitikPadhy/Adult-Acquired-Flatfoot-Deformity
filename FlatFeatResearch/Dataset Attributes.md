Here the metadata and radiographic measurements of our dataset for AAFD:
## **A. Demographic / Metadata Variables**

1. **SN** → Serial number (just the row/patient index).
2. **R_1L_2** → Side of measurement: 1 = Right foot, 2 = Left foot.
3. **A** → Age of patient.
4. **HN** → Hospital number (unique patient identifier).
5. **AS_0S_1** → Symptom status: 0 = Asymptomatic, 1 = Symptomatic. This will help us tell if our prediction was right. But there might be scenarios where a patient might not experience pain, but is still experiencing AAFD. This is the difference between radiographic AAFD and Clinical AAFD. Therefore, we are going to predict both, one using angle thresholds and the other using this our label. 
6. **M_1 F_2** → Gender: 1 = Male, 2 = Female.
7. **CMP_1A_2** → Comorbidities: 1 = Present (diabetes, hypertension, high BMI), 2 = Absent.

---

## **B. Radiographic Variables**

### 1. **Calcaneal Pitch Angle (Cal PA)**

- **What it is:** Tells how high the heel bone (calcaneus) sits → the “height” of the foot arch. The lower the angle, the flatter the arch.
- **Normal range:** 20–40°
- **Abnormal:** <20° → flatfoot (low arch). 
- Columns:
    - **cal_PA_11** → 1st observer, 1st time.
    - **cal_PA_21** → 2nd observer, 1st time.
    - **cal_PA_12** → 1st observer, 2nd time.
    - **cal_PA_22** → 2nd observer, 2nd time.

---

### 2. **Meary’s Angle (MA) – Talo-First Metatarsal Angle**

- **What it is:** Checks if the talus (ankle bone) and 1st metatarsal (big toe bone) are in a straight line. In a normal arch, these two bones align. In flatfoot, they diverge.
- **Normal range:** 0–4°
- **Abnormal:** >4° → midfoot collapse.
- Columns:
    - **MA_11**, **MA_21**, **MA_12**, **MA_22** (same observer/time logic as above).

---

### 3. **Talocalcaneal Angle (TCA)**

- **What it is:** Shows the relationship between talus and calcaneus (hindfoot bones). Bigger angle = heel bone drifts outward (valgus), common in flatfoot.
- **Normal range:** 25–40°
- **Abnormal:** >40° (hindfoot valgus).
- Columns: **TCA_11, TCA_21, TCA_12, TCA_22**.

---

### 4. **Medial Column Length (ML)**

- Length of the inner “arch side” of the foot. - Shorter medial column → collapse of the inner arch.
- Columns: **ML_11, ML_21, ML_12, ML_22**.

---

### 5. **Lateral Column Length (LL)**

- Length of lateral column (outer side of foot). Used to compare with medial column to detect imbalance.
- Columns: **LL_11, LL_21, LL_12, LL_22**.

---

### 6. **Medial Cuneiform–5th Metatarsal Length (MCL)**

- Distance between medial cuneiform and 5th metatarsal (structural alignment). Measures structural length/alignment across the midfoot. Helps assess collapse/rotation in midfoot.
- Columns: **MCL_11, MCL_21, MCL_12, MCL_22**.

---

### 7. **5th Metatarsal–Calcaneal Angle (MT_calA)**

- Relationship of heel to 5th metatarsal bone. If angle is small, it suggests collapse of the arch.
- **Threshold:** ≤12° = abnormal.
- Columns: **MT_calA_11, MT_calA_21, MT_calA_12, MT_calA_22**.

---

### 8. **1st Metatarsal–Talar Angle (1MTA)**

- Checks alignment of talus and big toe bone (1st metatarsal). Large angle means talus is tilted down → flatfoot sign.
- **Normal:** 7° ± 5° (so ~2–12°). 
- **Abnormal:** >12°.
- Columns: **1MTA_11, 1MTA_21, 1MTA_12, 1MTA_22**.
    

---

### 9. **5th Metatarsal–Calcaneal Angle (5MTCA)**

- Similar to MT_calA but checks lateral (outer) side. Helps detect forefoot abduction and midfoot collapse.
- **Threshold:** ≤12° = abnormal.
- Columns: **5MTCA_11, 5MTCA_21, 5MTCA_12, 5MTCA_22**.

---

### 10. **Talar Uncoverage Angle (TUA)**

- Shows how much of the talar head is uncovered at the talonavicular joint. - More uncovering = forefoot drifts outward → flatfoot.
- **Normal:** <7°
- **Abnormal:** >7° (forefoot abduction, talonavicular uncoverage).
- Columns: **TUA_11, TUA_21, TUA_12, TUA_22**.