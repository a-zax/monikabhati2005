"""
ReelSense Part 4: Evaluation Metrics
Comprehensive evaluation for ranking, diversity, and novelty
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    def __init__(self):
        self.metrics = {}
    
    # ========================================================================
    # RANKING METRICS
    # ========================================================================
    
    def precision_at_k(self, recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        if len(recommended_k) == 0:
            return 0.0
        return len(set(recommended_k).intersection(relevant_set)) / len(recommended_k)
    
    def recall_at_k(self, recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        if len(relevant_set) == 0:
            return 0.0
        return len(set(recommended_k).intersection(relevant_set)) / len(relevant_set)
    
    def ndcg_at_k(self, recommended, relevant, k=10):
        """Normalized Discounted Cumulative Gain"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0
        
        # Ideal DCG
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k))])
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def map_at_k(self, recommended, relevant, k=10):
        """Mean Average Precision"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        if len(relevant_set) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(relevant_set), k)
    
    # ========================================================================
    # DIVERSITY METRICS
    # ========================================================================
    
    def catalog_coverage(self, all_recommendations, total_items):
        """Percentage of catalog covered by recommendations"""
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        if total_items == 0: return 0.0
        return len(unique_items) / total_items
    
    def intra_list_diversity(self, recommendations, movie_features):
        """Average pairwise distance within recommendation list"""
        if len(recommendations) < 2:
            return 0.0
        
        # Filter items present in features
        valid_items = [item for item in recommendations if item in movie_features.index]
        if len(valid_items) < 2:
            return 0.0
            
        # Get features for valid items
        features = movie_features.loc[valid_items].values
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(features)
        
        # Sum upper triangle (excluding diagonal) to get pairwise similarities
        n = len(valid_items)
        sum_sim = (np.sum(sim_matrix) - n) / 2
        num_pairs = n * (n - 1) / 2
        
        avg_sim = sum_sim / num_pairs if num_pairs > 0 else 0
        return 1.0 - avg_sim
    
    def gini_index(self, all_recs_flat):
        """Gini coefficient to measure inequality in item recommendation distribution"""
        if not all_recs_flat:
            return 0.0
            
        item_counts = Counter(all_recs_flat)
        counts = sorted(item_counts.values())
        n = len(counts)
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(counts)
        return (2 * np.sum((np.arange(1, n + 1)) * counts) / (n * cumsum[-1])) - (n + 1) / n
    
    def genre_diversity(self, recommendations, movies_df, k=10):
        """Measure diversity of genres in recommendations"""
        rec_k = recommendations[:k]
        genres_set = set()
        total_genres = 0
        
        for mid in rec_k:
            rows = movies_df[movies_df['movieId'] == mid]
            if len(rows) > 0:
                gs = str(rows.iloc[0]['genres']).split('|')
                genres_set.update(gs)
                total_genres += len(gs)
        
        # Simple metric: number of unique genres / total number of genre slots (k * avg_genres/movie?)
        # Or just number of unique genres.
        # Let's normalize by total number of possible genres found in these movies? 
        # A simple interpretable metric is difficult without a baseline.
        # Let's return just count of unique genres for now, normalized by k maybe? 
        # Actually in main result reporting we often just want a score [0,1].
        # Let's use simple unique_genres / total_unique_genres_in_catalog (approximation)
        # OR: unique_genres / k (if max 1 genre per movie). 
        # Let's just return normalized count: unique genres / total genre occurrences
        if total_genres == 0:
            return 0.0
        return len(genres_set) / total_genres

    # ========================================================================
    # NOVELTY METRICS
    # ========================================================================
    
    def novelty_score(self, recommendations, item_popularity):
        """Average novelty (inverse popularity) of recommendations"""
        if len(recommendations) == 0:
            return 0.0
        
        # Avoid log(0)
        pop_values = list(item_popularity.values())
        if not pop_values: return 0.0
        
        max_pop = max(pop_values)
        novelty_scores = []
        
        for item in recommendations:
            popularity = item_popularity.get(item, 0)
            # Self-information: -log2(p(i)) where p(i) = pop/total_interactions?
            # Or standard definition: -log2(pop/max_pop)
            val = (popularity + 1) / (max_pop + 1) # Smoothed
            novelty = -np.log2(val)
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def long_tail_percentage(self, recommendations, item_popularity, percentile=80):
        """Percentage of recommendations from long-tail (items with <= percentile popularity)"""
        if not item_popularity: return 0.0
        
        threshold = np.percentile(list(item_popularity.values()), 100 - percentile) # Bottom 80% means popularity < 20th percentile of top? 
        # Usually long tail = items that are NOT in the top head.
        # If we sort by popularity descending. Head = top X%. Tail = rest.
        # Let's say long tail is bottom 80% of ITEMS (not ratings).
        
        # Simple approach: items with fewer ratings than the 80th percentile item?
        # No, usually "long tail" means items that are NOT popular.
        # Let's define it as items in the bottom 80% of popularity distribution.
        
        pop_values = list(item_popularity.values())
        threshold_val = np.percentile(pop_values, percentile) # This gives score below which X% of data falls.
        # if percentile=80, it gives the value below which 80% of items fall.
        
        long_tail_count = sum(1 for item in recommendations if item_popularity.get(item, 0) <= threshold_val)
        
        return long_tail_count / len(recommendations) if len(recommendations) > 0 else 0.0
    
    # ========================================================================
    # EVALUATION PIPELINE
    # ========================================================================
    
    def evaluate_multiple_users(self, user_recommendations, user_relevance, 
                               movie_features, movies_df, item_popularity, total_items, k=10):
        """
        Evaluate for multiple users and aggregate results
        """
        dataset_results = []
        
        all_recs_flat = []
        for user_id, recs in user_recommendations.items():
            all_recs_flat.extend(recs)
            
            relevant = user_relevance.get(user_id, [])
            
            # Per-user metrics
            metrics = {
                'user_id': user_id,
                'Precision@K': self.precision_at_k(recs, relevant, k),
                'Recall@K': self.recall_at_k(recs, relevant, k),
                'NDCG@K': self.ndcg_at_k(recs, relevant, k),
                'MAP@K': self.map_at_k(recs, relevant, k),
                'Intra_List_Diversity': self.intra_list_diversity(recs, movie_features),
                'Genre_Diversity': self.genre_diversity(recs, movies_df, k),
                'Novelty_Score': self.novelty_score(recs, item_popularity),
                'Long_Tail_%': self.long_tail_percentage(recs, item_popularity)
            }
            dataset_results.append(metrics)
            
        results_df = pd.DataFrame(dataset_results)
        
        # System-wide metrics
        coverage = self.catalog_coverage(user_recommendations.values(), total_items)
        gini = self.gini_index(all_recs_flat)
        
        # Add average row
        avg_metrics = results_df.drop('user_id', axis=1).mean().to_dict()
        avg_metrics['user_id'] = 'AVERAGE'
        
        # Add system wide metrics to average row (for reporting convenience)
        avg_metrics['Catalog_Coverage'] = coverage
        avg_metrics['Gini_Index'] = gini
        
        # Append average row
        results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)
        
        return results_df
        
    def print_evaluation_summary(self, results_df):
        """Print formatted summary"""
        avg_row = results_df[results_df['user_id'] == 'AVERAGE'].iloc[0]
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        cols = ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K', 
                'Intra_List_Diversity', 'Catalog_Coverage', 'Gini_Index', 
                'Novelty_Score', 'Long_Tail_%']
        
        for col in cols:
            if col in avg_row:
                print(f"{col:<25}: {avg_row[col]:.4f}")
        print("="*50 + "\n")


def compare_models(evaluator, models_results):
    """
    Compare multiple models
    models_results: dict {model_name: results_df}
    """
    print("\nMODEL COMPARISON:")
    print(f"{'Metric':<25} | " + " | ".join([f"{name:<15}" for name in models_results.keys()]))
    print("-" * (25 + 18 * len(models_results)))
    
    metrics = ['NDCG@K', 'Precision@K', 'Intra_List_Diversity', 'Catalog_Coverage']
    
    for metric in metrics:
        values = []
        for res_df in models_results.values():
            val = res_df[res_df['user_id'] == 'AVERAGE'].iloc[0].get(metric, 0)
            values.append(f"{val:.4f}")
        
        print(f"{metric:<25} | " + " | ".join([f"{v:<15}" for v in values]))

if __name__ == "__main__":
    print("ReelSense Part 4: Evaluation Module")
