# 🎬 ReelSense: Explainable Movie Recommender System with Diversity Optimization

## 📋 Project Overview

ReelSense is a comprehensive movie recommendation system that goes beyond traditional rating prediction. It combines:

1. **Hybrid Recommendation Models** - Multiple approaches (Collaborative Filtering, Matrix Factorization, Content-Based)
2. **Diversity Optimization** - Ensures varied recommendations avoiding popularity bias
3. **Natural Language Explanations** - Transparent reasoning for each recommendation
4. **Comprehensive Evaluation** - Ranking, diversity, and novelty metrics

## 🎯 Features

- ✅ **Hybrid Recommendation**: Combines 5 different recommendation algorithms
- ✅ **Diversity Optimization**: MMR algorithm, genre balancing, popularity debiasing
- ✅ **Explainability**: Natural language explanations based on genres, tags, and collaborative signals
- ✅ **Comprehensive Metrics**: NDCG, MAP, Precision, Recall, Catalog Coverage, Gini Index, Novelty Score
- ✅ **Professional Visualizations**: High-quality plots for EDA and results
- ✅ **Modular Architecture**: Clean, reusable code structure

## 📁 Project Structure

```
reelsense/
├── src/                           # Source code
│   ├── reelsense_main.py          # Main execution script
│   ├── reelsense_part1_data.py    # Data loading, EDA, feature engineering
│   ├── reelsense_part2_models.py  # All recommendation models
│   ├── reelsense_part3_diversity.py # Diversity optimization & explainability
│   ├── reelsense_part4_evaluation.py # Evaluation metrics
├── data/                          # Dataset directory
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── outputs/                       # Generated outputs
│   ├── visualizations/
│   └── results/
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── ReelSense_Implementation_Plan.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Data items are already in `data/`. If setting up fresh, download the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/) and place csvs in `data/`.

### 3. Run the System

Navigate to the `src` directory and run:

```bash
cd src
python reelsense_main.py
```

## 📊 Expected Outputs

### Visualizations (in `visualizations/` folder)
- `rating_distribution.png` - Distribution of user ratings
- `user_activity.png` - User engagement patterns
- `genre_analysis.png` - Genre popularity and ratings
- `long_tail_analysis.png` - Movie popularity distribution
- `temporal_trends.png` - Rating trends over time
- `metrics_summary.png` - Model performance metrics

### Results (in `results/` folder)
- `evaluation_results.csv` - Detailed metrics per user
- `reelsense_report.txt` - Comprehensive final report

## 🔧 Configuration

Edit `Config` class in `reelsense_main.py` to tune parameters like `K` (number of recommendations), `LAMBDA_MMR` (diversity weight), and `HYBRID_WEIGHTS`.

## 📝 Citation

If using this code, please cite:

```
ReelSense: Explainable Movie Recommender System with Diversity Optimization
Hackathon Submission, 2026
Dataset: MovieLens Latest Small (GroupLens Research)
```
