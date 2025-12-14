# Leakage-Free Adaptation Size Sweep Report

**Date:** 2025-12-14
**Goal:** Verify adaptation capability and optimal adaptation dataset size using the rigorous **Leakage-Free Protocol**.
**Run Duration:** ~9 minutes (with parallel execution optimization).

## 1. Executive Summary

This report documents the results of a parameter sweep over the Adaptation Dataset Size for the Markov Field Classifier. The sweep strictly adheres to the **Leakage-Free Protocol**:
1.  **Phase A (Baseline):** Evaluate accuracy on Test Set (10k) with feedback *disabled*.
2.  **Phase B (Adaptation):** Adapt model on a *strict subset* of Training Data (Size N) with feedback *enabled* and *learning enabled*.
3.  **Phase C (Frozen Eval):** Evaluate accuracy on Test Set (10k) with feedback *enabled* but *learning disabled*.
4.  **Reset:** Full MRF reconstruction and state reset between every single run (Seed/Size combination).

**Key Findings:**
*   **Optimum Found:** Adaptation size of **10,000** consistently yielded the highest accuracy gains across all feedback modes.
*   **Best Config:** `PATCH_ROW` mode with 10,000 samples achieved **70.04%** accuracy, a significant lift over the baseline **69.50%**.
*   **Performance:** All modes (Patch, Row, Column, Combined) showed positive delta, confirming the value of feedback adaptation.
*   **Diminishing Returns:** Increasing adaptation size to 20,000 often resulted in slightly lower or plateaued accuracy compared to 10,000, suggesting potential overfitting to the adaptation subset or saturation of the unigram updates.

## 2. Dataset Summary

*   **Training Set:** 60,000 images (MNIST Train)
    *   **Filtering:** **2 images** were removed from the training set because they were identified as duplicates of images in the test set.
    *   **Effective Train Size:** 59,998 images.
*   **Test Set:** 10,000 images (MNIST Test)
*   **Integrity Check:** **0 Overlaps** detected between Adaptation Subsets and Test Set (after filtering).

## 3. Results Table

**Configurations:**
*   **Adaptation Sizes:** 2000, 5000, 10000, 20000
*   **Seeds:** 5 seeds per configuration (12345, 22222, 33333, 44444, 55555)
*   **Feedback Modes:** PATCH, PATCH_ROW, PATCH_COL, PATCH_ROW_COL

### Baseline Accuracy
**Baseline Accuracy (Feedback Disabled):** `0.6950`

### Summary Table

| Mode | Adapt Size | Mean Frozen Acc | Mean Delta | Std Dev Delta | Min Delta | Max Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PATCH | 2000 | 0.6968 | +0.0018 | 0.0005 | +0.0010 | +0.0024 |
| PATCH | 5000 | 0.6987 | +0.0037 | 0.0017 | +0.0016 | +0.0050 |
| PATCH | 10000 | 0.6998 | +0.0048 | 0.0008 | +0.0039 | +0.0057 |
| PATCH | 20000 | 0.6998 | +0.0048 | 0.0008 | +0.0037 | +0.0060 |
| PATCH_ROW | 2000 | 0.6970 | +0.0020 | 0.0015 | -0.0003 | +0.0039 |
| PATCH_ROW | 5000 | 0.6994 | +0.0044 | 0.0014 | +0.0024 | +0.0056 |
| **PATCH_ROW** | **10000** | **0.7004** | **+0.0054** | **0.0011** | **+0.0038** | **+0.0064** |
| PATCH_ROW | 20000 | 0.6992 | +0.0042 | 0.0014 | +0.0021 | +0.0057 |
| PATCH_COL | 2000 | 0.6971 | +0.0021 | 0.0016 | -0.0003 | +0.0040 |
| PATCH_COL | 5000 | 0.6995 | +0.0045 | 0.0013 | +0.0027 | +0.0060 |
| PATCH_COL | 10000 | 0.7003 | +0.0053 | 0.0006 | +0.0045 | +0.0059 |
| PATCH_COL | 20000 | 0.6989 | +0.0039 | 0.0018 | +0.0011 | +0.0057 |
| PATCH_ROW_COL | 2000 | 0.6972 | +0.0022 | 0.0021 | -0.0003 | +0.0050 |
| PATCH_ROW_COL | 5000 | 0.6995 | +0.0045 | 0.0016 | +0.0024 | +0.0068 |
| PATCH_ROW_COL | 10000 | 0.6997 | +0.0047 | 0.0009 | +0.0035 | +0.0058 |
| PATCH_ROW_COL | 20000 | 0.6984 | +0.0034 | 0.0017 | +0.0008 | +0.0050 |

## 4. Verification Evidence

*   **Phase C Guardrail:** Logs confirmed `Updated Patch4x4 feedback config: enabled=true, learningEnabled=false` before final frozen evaluation.
*   **Adaptation Guardrail:** Learner updates were verified to be non-zero (2000 updates) during Phase B and zero during Phase C.
*   **Data Integrity:** The system automatically identified and **removed 2 overlapping image hashes** from the training set, ensuring strict separation from the test set.

## 5. Raw Output (CSV)

```csv
mode,adaptSize,meanFrozen,meanDelta,stdDelta,minDelta,maxDelta
PATCH,2000,0.6968,0.0018,0.0005,0.0010,0.0024
PATCH,5000,0.6987,0.0037,0.0017,0.0016,0.0050
PATCH,10000,0.6998,0.0048,0.0008,0.0039,0.0057
PATCH,20000,0.6998,0.0048,0.0008,0.0037,0.0060
PATCH_ROW,2000,0.6970,0.0020,0.0015,-0.0003,0.0039
PATCH_ROW,5000,0.6994,0.0044,0.0014,0.0024,0.0056
PATCH_ROW,10000,0.7004,0.0054,0.0011,0.0038,0.0064
PATCH_ROW,20000,0.6992,0.0042,0.0014,0.0021,0.0057
PATCH_COL,2000,0.6971,0.0021,0.0016,-0.0003,0.0040
PATCH_COL,5000,0.6995,0.0045,0.0013,0.0027,0.0060
PATCH_COL,10000,0.7003,0.0053,0.0006,0.0045,0.0059
PATCH_COL,20000,0.6989,0.0039,0.0018,0.0011,0.0057
PATCH_ROW_COL,2000,0.6972,0.0022,0.0021,-0.0003,0.0050
PATCH_ROW_COL,5000,0.6995,0.0045,0.0016,0.0024,0.0068
PATCH_ROW_COL,10000,0.6997,0.0047,0.0009,0.0035,0.0058
PATCH_ROW_COL,20000,0.6984,0.0034,0.0017,0.0008,0.0050
```
