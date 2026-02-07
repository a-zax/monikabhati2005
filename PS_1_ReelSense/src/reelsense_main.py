"""
ReelSense Main: Complete Execution Pipeline
Run this file to execute the full ReelSense system

Author: Hackathon Participant
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all components
from reelsense_part1_data import (
    DataLoader, EDAVisualizer, FeatureEngineer,
    create_user_item_matrix, get_movie_popularity
)
from reelsense_part2_models import (
    PopularityRecommender, UserBasedCF, ItemBasedCF,
    SimpleSVD, ContentBasedRecommender, HybridRecommender
)
from reelsense_part3_diversity import (
    DiversityOptimizer, ExplainabilityEngine, RecommendationPipeline
)
from reelsense_part4_evaluation import (
    RecommenderEvaluator, compare_models
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for ReelSense system"""
    
    # Data paths
    # Data paths
    DATA_PATH = '../data/'
    OUTPUT_DIR = '../outputs/results/'
    VIZ_DIR = '../outputs/visualizations/'
    
    # Model parameters
    K = 10  # Number of recommendations
    N_TEST = 3  # Number of test ratings per user
    
    # Diversity parameters
    LAMBDA_MMR = 0.5  # Balance between relevance and diversity
    
    # Hybrid model weights
    HYBRID_WEIGHTS = {
        'popularity': 0.1,
        'user_cf': 0.25,
        'item_cf': 0.25,
        'svd': 0.25,
        'content': 0.15
    }
    
    # Evaluation
    EVAL_SAMPLE_USERS = 50  # Number of users to evaluate (set to None for all)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print(" "*15 + "🎬 REELSENSE SYSTEM 🎬")
    print(" "*5 + "Explainable Movie Recommender with Diversity")
    print("="*70)
    
    # Create output directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.VIZ_DIR, exist_ok=True)
    
    # ========================================================================
    # STEP 1: DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    loader = DataLoader(data_path=Config.DATA_PATH)
    try:
        movies, ratings, tags, links = loader.load_data()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        return

    movies, ratings, tags = loader.clean_data()
    
    # Time-based split
    train_ratings, test_ratings = loader.time_based_split(n_test=Config.N_TEST)
    
    # ========================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    eda = EDAVisualizer(movies, ratings, tags, output_dir=Config.VIZ_DIR)
    eda.generate_all_visualizations()
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    engineer = FeatureEngineer(movies, tags)
    genre_features = engineer.create_genre_features()
    tag_features = engineer.create_tag_features()
    combined_features = engineer.combine_features()
    
    # Calculate movie popularity
    movie_popularity = get_movie_popularity(train_ratings)
    
    # ========================================================================
    # STEP 4: MODEL TRAINING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: MODEL TRAINING")
    print("="*70)
    
    # Initialize models
    print("\nInitializing models...")
    
    # Baseline
    pop_model = PopularityRecommender()
    
    # Collaborative Filtering
    user_cf = UserBasedCF(k_neighbors=20)
    item_cf = ItemBasedCF(k_neighbors=20)
    
    # Matrix Factorization
    svd_model = SimpleSVD(n_factors=50, n_iterations=20)
    
    # Content-Based
    content_model = ContentBasedRecommender()
    
    # Hybrid
    hybrid_model = HybridRecommender(weights=Config.HYBRID_WEIGHTS)
    
    # Train all models
    print("\nTraining all models...")
    pop_model.fit(train_ratings)
    user_cf.fit(train_ratings)
    item_cf.fit(train_ratings)
    svd_model.fit(train_ratings)
    content_model.fit(combined_features)
    hybrid_model.fit(train_ratings, combined_features)
    
    print("\n✓ All models trained successfully!")
    
    # ========================================================================
    # STEP 5: DIVERSITY OPTIMIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: DIVERSITY OPTIMIZATION")
    print("="*70)
    
    diversity_optimizer = DiversityOptimizer(lambda_param=Config.LAMBDA_MMR)
    
    # ========================================================================
    # STEP 6: EXPLAINABILITY SETUP
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: EXPLAINABILITY SETUP")
    print("="*70)
    
    explainer = ExplainabilityEngine(movies, tags, train_ratings)
    
    # ========================================================================
    # STEP 7: GENERATE RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 7: GENERATING RECOMMENDATIONS")
    print("="*70)
    
    # Create recommendation pipeline
    pipeline = RecommendationPipeline(
        recommender=hybrid_model,
        diversity_optimizer=diversity_optimizer,
        explainer=explainer
    )
    
    # Get test users
    test_users = test_ratings['userId'].unique()
    if Config.EVAL_SAMPLE_USERS:
        # Ensure we don't sample more than available
        n_sample = min(Config.EVAL_SAMPLE_USERS, len(test_users))
        test_users = np.random.choice(
            test_users, 
            size=n_sample,
            replace=False
        )
    
    print(f"\nGenerating recommendations for {len(test_users)} users...")
    
    all_movies = movies['movieId'].unique()
    
    # Store recommendations
    user_recommendations = {}
    user_relevance = {}
    
    for idx, user_id in enumerate(test_users):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(test_users)} users")
        
        # Generate recommendations with diversity and explanations
        recs = pipeline.recommend_with_diversity_and_explanations(
            user_id=user_id,
            ratings_df=train_ratings,
            all_movies=all_movies,
            movie_features=combined_features,
            movies_df=movies,
            k=Config.K,
            diversity_method='combined'
        )
        
        # Extract movie IDs
        rec_movie_ids = [movie_id for movie_id, _, _ in recs]
        user_recommendations[user_id] = rec_movie_ids
        
        # Get relevant items (test set items rated >= 4.0)
        user_test = test_ratings[
            (test_ratings['userId'] == user_id) &
            (test_ratings['rating'] >= 4.0)
        ]
        user_relevance[user_id] = user_test['movieId'].tolist()
    
    print(f"\n✓ Generated recommendations for {len(test_users)} users")
    
    # ========================================================================
    # STEP 8: EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 8: EVALUATION")
    print("="*70)
    
    evaluator = RecommenderEvaluator()
    
    # Evaluate hybrid model with diversity
    results_df = evaluator.evaluate_multiple_users(
        user_recommendations=user_recommendations,
        user_relevance=user_relevance,
        movie_features=combined_features,
        movies_df=movies,
        item_popularity=movie_popularity,
        total_items=len(movies),
        k=Config.K
    )
    
    # Print summary
    evaluator.print_evaluation_summary(results_df)
    
    # Save results
    results_df.to_csv(
        f'{Config.OUTPUT_DIR}evaluation_results.csv',
        index=False
    )
    print(f"\n✓ Results saved to {Config.OUTPUT_DIR}evaluation_results.csv")
    
    # ========================================================================
    # STEP 9: GENERATE SAMPLE EXPLANATIONS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 9: SAMPLE RECOMMENDATIONS WITH EXPLANATIONS")
    print("="*70)
    
    # Show sample recommendations for first 3 users
    sample_users = list(test_users)[:3]
    
    for user_id in sample_users:
        print(f"\n{'='*70}")
        print(f"USER {user_id} - Top {Config.K} Recommendations:")
        print(f"{'='*70}")
        
        # Get recommendations with explanations
        recs = pipeline.recommend_with_diversity_and_explanations(
            user_id=user_id,
            ratings_df=train_ratings,
            all_movies=all_movies,
            movie_features=combined_features,
            movies_df=movies,
            k=Config.K,
            diversity_method='combined'
        )
        
        for rank, (movie_id, title, explanation) in enumerate(recs, 1):
            print(f"\n{rank}. {title}")
            print(f"   💡 {explanation}")
    
    # ========================================================================
    # STEP 10: VISUALIZE RESULTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 10: VISUALIZING RESULTS")
    print("="*70)
    
    # Plot metrics comparison
    avg_results = results_df[results_df['user_id'] == 'AVERAGE'].iloc[0]
    
    # Ranking metrics
    ranking_metrics = ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K']
    ranking_values = [avg_results.get(m, 0) for m in ranking_metrics]
    
    # Diversity metrics
    diversity_metrics = ['Catalog_Coverage', 'Intra_List_Diversity', 
                        'Genre_Diversity', 'Long_Tail_%']
    diversity_values = [avg_results.get(m, 0) for m in diversity_metrics]

    if all(v == 0 for v in ranking_values) and all(v == 0 for v in diversity_values):
        print("Skipping metrics plot as all values are zero (likely no relevant items found for sample users).")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(ranking_metrics, ranking_values, color='steelblue', edgecolor='black')
        axes[0].set_title('Ranking Metrics', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(diversity_metrics, diversity_values, color='coral', edgecolor='black')
        axes[1].set_title('Diversity Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{Config.VIZ_DIR}metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Metrics visualization saved to {Config.VIZ_DIR}metrics_summary.png")
    
    # ========================================================================
    # STEP 11: GENERATE REPORT
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 11: GENERATING FINAL REPORT")
    print("="*70)
    
    report_path = f'{Config.OUTPUT_DIR}reelsense_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" "*15 + "REELSENSE FINAL REPORT\n")
        f.write(" "*5 + "Explainable Movie Recommender with Diversity\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*70 + "\n")
        f.write("DATASET STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Total Movies:     {len(movies)}\n")
        f.write(f"Total Ratings:    {len(ratings)}\n")
        f.write(f"Total Users:      {ratings['userId'].nunique()}\n")
        f.write(f"Total Tags:       {len(tags)}\n")
        f.write(f"Train Ratings:    {len(train_ratings)}\n")
        f.write(f"Test Ratings:     {len(test_ratings)}\n")
        
        sparsity = 1.0
        n_users = ratings['userId'].nunique()
        n_movies = len(movies)
        if n_users > 0 and n_movies > 0:
            sparsity = 1 - (len(ratings) / (n_users * n_movies))
        
        f.write(f"Sparsity:         {sparsity:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*70 + "\n")
        f.write(f"Hybrid Weights:\n")
        for model, weight in Config.HYBRID_WEIGHTS.items():
            f.write(f"  - {model}: {weight}\n")
        f.write(f"\nDiversity Lambda: {Config.LAMBDA_MMR}\n")
        f.write(f"Recommendation Count (K): {Config.K}\n\n")
        
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("RANKING METRICS:\n")
        f.write(f"  Precision@{Config.K}:  {avg_results.get('Precision@K', 0):.4f}\n")
        f.write(f"  Recall@{Config.K}:     {avg_results.get('Recall@K', 0):.4f}\n")
        f.write(f"  NDCG@{Config.K}:       {avg_results.get('NDCG@K', 0):.4f}\n")
        f.write(f"  MAP@{Config.K}:        {avg_results.get('MAP@K', 0):.4f}\n\n")
        
        f.write("DIVERSITY METRICS:\n")
        f.write(f"  Catalog Coverage:      {avg_results.get('Catalog_Coverage', 0):.4f}\n")
        f.write(f"  Intra-List Diversity:  {avg_results.get('Intra_List_Diversity', 0):.4f}\n")
        f.write(f"  Genre Diversity:       {avg_results.get('Genre_Diversity', 0):.4f}\n")
        f.write(f"  Gini Index:            {avg_results.get('Gini_Index', 0):.4f}\n\n")
        
        f.write("NOVELTY METRICS:\n")
        f.write(f"  Novelty Score:    {avg_results.get('Novelty_Score', 0):.4f}\n")
        f.write(f"  Long-tail %:      {avg_results.get('Long_Tail_%', 0):.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*70 + "\n")
        f.write("ReelSense successfully combines multiple recommendation approaches\n")
        f.write("with diversity optimization and natural language explanations.\n")
        f.write(f"\nThe system achieves an NDCG@{Config.K} of {avg_results.get('NDCG@K', 0):.4f}, ")
        f.write(f"while maintaining\n")
        f.write(f"high diversity (Intra-List: {avg_results.get('Intra_List_Diversity', 0):.4f}) and ")
        f.write(f"novelty (Long-tail: {avg_results.get('Long_Tail_%', 0):.2%}).\n\n")
    
    print(f"✓ Report saved to {report_path}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print("\n" + "="*70)
    print("✅ REELSENSE EXECUTION COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Visualizations: {Config.VIZ_DIR}")
    print(f"  - Results: {Config.OUTPUT_DIR}")
    print(f"\nKey Files:")
    print(f"  - Evaluation Results: {Config.OUTPUT_DIR}evaluation_results.csv")
    print(f"  - Final Report: {Config.OUTPUT_DIR}reelsense_report.txt")
    print(f"  - Metrics Plot: {Config.VIZ_DIR}metrics_summary.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
