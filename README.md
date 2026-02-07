# Cognitive Report Generator (Mock Implementation)

This project implements a deep learning model for generating radiology reports from chest X-rays.

## Running on Google Colab (Recommended)

Since training deep learning models requires significant GPU resources, we recommend running this project on Google Colab (Free Tier with T4 GPU).

### Prerequisites
1.  **Google Account** for Colab.
2.  **Kaggle Account** for dataset access.
3.  **`kaggle.json`** API token (Account -> Create New API Token).

### Steps
1.  **Download Codebase**: Use the `project_codebase.zip` provided.
2.  **Open Colab**: Go to [https://colab.research.google.com/](https://colab.research.google.com/) and standard "Upload Notebook".
3.  **Upload Notebook**: Upload `Run_on_Colab.ipynb` from the zip (or project root).
4.  **Upload Files**: In the Colab file explorer (left sidebar folder icon), upload:
    - `project_codebase.zip`
    - `kaggle.json`
5.  **Run All Cells**: The notebook will verify the environment, download data, and start training.

## Local Setup (Windows/Linux)

If you have a powerful GPU (8GB+ VRAM), you can run locally:

```bash
# 1. Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Data
# Ensure you have .kaggle/kaggle.json set up
python scripts/download_with_hub.py

# 4. Preprocess
python scripts/preprocess_iu_xray.py

# 5. Train
python training/train.py --batch_size 8 --epochs 15
```

## Troubleshooting
- **CUDA Errors**: Usually due to tokenizer mismatch. Ensure `distilbert-base-uncased` is used for encoder and `distilgpt2` for decoder.
- **Out of Memory**: Reduce batch size in `train.py` or Colab notebook.
- **Dataset Not Found**: Ensure `kaggle.json` is valid and uploaded.

