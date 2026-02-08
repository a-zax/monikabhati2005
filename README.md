# 🧠 BrainDead 2K26: AI Innovation Suite

**Team:** monikabhati2005 | **Hackathon:** BrainDead 2026 | **Problem Statements:** PS_1 & PS_2

---

## 🚀 Final Submission Summary

### 📄 Core Deliverables
- **Final Report:** [BrainDead_Hackathon_Report.pdf](BrainDead_Hackathon_Report.pdf)
- **Submission Snapshot:** [submission.txt](submission.txt)

### 📂 Problem Statement 1: ReelSense
*Explainable Recommender System with Diversity Optimization*
- **Repository:** [View PS_1_ReelSense](PS_1_ReelSense/)
- **GitHub Link:** [https://github.com/a-zax/monikabhati2005/tree/main/PS_1_ReelSense](https://github.com/a-zax/monikabhati2005/tree/main/PS_1_ReelSense)
- **Key Metric:** **0.8266 Intra-List Diversity** (Exceeds benchmarks)

### 📂 Problem Statement 2: Cognitive Radiology Assistant
*Hierarchical Vision-Language Model for Chest X-Ray Reporting*
- **Repository:** [View PS_2_Cognitive_Radiology_Report](PS_2_Cognitive_Radiology_Report/)
- **GitHub Link:** [https://github.com/a-zax/monikabhati2005/tree/main/PS_2_Cognitive_Radiology_Report](https://github.com/a-zax/monikabhati2005/tree/main/PS_2_Cognitive_Radiology_Report)
- **Model Checkpoint:** [MEGA Link](https://mega.nz/file/azh1iAxK#9dlcjr4OUYtOixJXeKcaXTEy5hsvUw7CfVlXHrrYjjU)
- **Demo Video:** [MEGA Video Link](https://mega.nz/file/HjBGzAJJ#ZpAO8MrJ5ld1vkFGIQzosPT05kmsZF-TeT0uyRtWlZY)
- **Key Metric:** **0.6421 CheXpert F1** (Significantly above benchmark)

---

## 🏗️ Technical Architecture Details

### PS_1: ReelSense
- **Hybrid Ensemble:** Fusion of 5 algorithms (SVD, CF, Popularity, Content-based).
- **Diversity Core:** Maximal Marginal Relevance (MMR) optimization.
- **Explainability:** Template-based natural language justifications.

### PS_2: Cognitive Radiology Assistant
- **PRO-FA:** Hierarchical visual feature alignment.
- **MIX-MLP:** Multi-task disease classification head.
- **RCTA:** Recursive cognitive triangular attention mechanism.
- **Clinical Grounding:** Post-processing safety layer for medical accuracy.

---

## 🛠️ Combined Setup Guide

```bash
# Clone the repository
git clone https://github.com/a-zax/monikabhati2005.git

# Initialize Problem Statement 1 (ReelSense)
cd PS_1_ReelSense
pip install -r requirements.txt
python -m reelsense.main

# Initialize Problem Statement 2 (Radiology)
cd ../PS_2_Cognitive_Radiology_Report
pip install -r requirements.txt
python scripts/gui_app.py
```

---

**Done & Dusted by Team monikabhati2005 | 2026**
