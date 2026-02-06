"""
ReelSense Part 3: Diversity Optimization & Explainability
Implements diversity re-ranking and explanation generation
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DIVERSITY OPTIMIZATION
# ============================================================================

class DiversityOptimizer:
    """
    Implements multiple diversity optimization strategies
    """
    
    def __init__(self, lambda_param=0.5):
        """
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param
    
    def mmr_rerank(self, recommendations, movie_features, k=10):
        """
        Maximal Marginal Relevance (MMR) algorithm
        Balances relevance and diversity
        
        Args:
            recommendations: List of movie IDs in order of relevance
            movie_features: DataFrame of movie feature vectors
            k: Number of items to return
        """
        if len(recommendations) == 0:
            return []
        
        selected = []
        candidates = recommendations.copy()
        
        # Select first item (highest relevance)
        # Check if first item is in features, if not skip it or take it blindly? 
        # Better to check.
        found_first = False
        for i, candid in enumerate(candidates):
            if candid in movie_features.index:
                selected.append(candidates.pop(i))
                found_first = True
                break
        
        if not found_first and candidates:
             # Just take the first one even if no features
             selected.append(candidates.pop(0))

        if len(selected) == 0:
            return recommendations[:k]
        
        # Pre-process features for fast access (DataFrame.loc is slow in loops)
        # Convert to numpy matrix and create index map
        valid_candidates = [c for c in candidates if c in movie_features.index]
        if not valid_candidates:
             return recommendations[:k]

        # Use efficient numpy operations
        feature_matrix = movie_features.values
        # Create map from movie_id to row index
        id_to_idx = {mid: i for i, mid in enumerate(movie_features.index)}
        
        # Helper to get vector
        def get_vector(mid):
             return feature_matrix[id_to_idx[mid]]

        while len(selected) < k and len(candidates) > 0:
            mmr_scores = []
            
            for candidate in candidates:
                if candidate not in id_to_idx:
                     # Fallback if not in features
                    position = recommendations.index(candidate)
                    relevance = 1.0 / (position + 1)
                    mmr = self.lambda_param * relevance + (1 - self.lambda_param) * 0.5
                    mmr_scores.append((candidate, mmr))
                    continue
                
                try:
                    position = recommendations.index(candidate)
                except ValueError:
                    position = len(recommendations)
                
                relevance = 1.0 / (position + 1)
                
                # Diversity score
                cand_vec = get_vector(candidate)
                
                max_sim = 0
                for selected_item in selected:
                    if selected_item in id_to_idx:
                        sel_vec = get_vector(selected_item)
                        # numpy cosine similarity: dot product (vectors are normalized?)
                        # Tfidf and dummies usually are not unit vectors by default?
                        # Cosine sim = dot(u, v) / (norm(u)*norm(v))
                        # For speed, let's assume doing dot product is enough if we normalized?
                        # Or just use sklearn cosine_similarity but on single vectors it's slow.
                        # Manual calculation is faster here.
                        dot = np.dot(cand_vec, sel_vec)
                        norm_a = np.linalg.norm(cand_vec)
                        norm_b = np.linalg.norm(sel_vec)
                        if norm_a > 0 and norm_b > 0:
                            sim = dot / (norm_a * norm_b)
                        else:
                            sim = 0
                        
                        if sim > max_sim:
                            max_sim = sim
                
                diversity = 1 - max_sim
                mmr = self.lambda_param * relevance + (1 - self.lambda_param) * diversity
                mmr_scores.append((candidate, mmr))
            
            if not mmr_scores:
                break
            
            # Select best
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate = mmr_scores[0][0]
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        
        # Fill rest if needed
        while len(selected) < k and candidates:
             selected.append(candidates.pop(0))

        return selected[:k]
    
    def genre_diversity_rerank(self, recommendations, movies_df, k=10, max_per_genre=3):
        """
        Ensure genre diversity in recommendations
        Limits number of movies per genre
        
        Args:
            recommendations: List of movie IDs
            movies_df: DataFrame with movie metadata
            k: Number of items to return
            max_per_genre: Maximum movies per genre
        """
        selected = []
        genre_count = defaultdict(int)
        
        # Pre-fetch genres to avoid repeated lookups
        movie_genres_map = {}
        for mid in recommendations:
             rows = movies_df[movies_df['movieId'] == mid]
             if len(rows) > 0:
                 movie_genres_map[mid] = str(rows.iloc[0]['genres']).split('|')
             else:
                 movie_genres_map[mid] = []

        for movie_id in recommendations:
            if len(selected) >= k:
                break
            
            genres = movie_genres_map.get(movie_id, [])
            
            # Check if adding this movie violates genre limits
            # We are lenient: if ANY genre is over limit, we might skip, 
            # BUT if the movie introduces a NEW genre, we might prioritize it.
            # Simple logic: skip if ALL its genres are already 'full' or if MAJORITY are full?
            # Strict logic: skip if ANY genre is full.
            
            can_add = True
            for genre in genres:
                if genre_count[genre] >= max_per_genre:
                    can_add = False
                    break
            
            # Soften the constraint if we are running out of items
            if can_add or len(selected) < k // 2:
                selected.append(movie_id)
                for genre in genres:
                    genre_count[genre] += 1
        
        # If we don't have enough, add remaining
        if len(selected) < k:
            for movie_id in recommendations:
                if movie_id not in selected and len(selected) < k:
                    selected.append(movie_id)
        
        return selected[:k]
    
    def popularity_debiasing(self, recommendations, ratings_df, k=10, 
                           boost_factor=0.3):
        """
        Boost less popular (long-tail) movies
        
        Args:
            recommendations: List of movie IDs
            ratings_df: DataFrame with ratings
            k: Number of items to return
            boost_factor: How much to boost unpopular items
        """
        # Calculate popularity
        movie_popularity = ratings_df.groupby('movieId').size()
        max_popularity = movie_popularity.max()
        
        # Rerank with popularity penalty/boost
        reranked = []
        for idx, movie_id in enumerate(recommendations):
            popularity = movie_popularity.get(movie_id, 1)
            
            # Position-based relevance
            relevance = 1.0 / (idx + 1)
            
            # Popularity penalty (high popularity = penalty)
            popularity_norm = popularity / max_popularity if max_popularity > 0 else 0
            
            # Adjusted score: Higher score means BETTER rank
            # We want to boost items where popularity is LOW.
            # penalty = popularity_norm * boost_factor
            # This logic: 1 - penalty. If pop=1, penalty=boost_factor. Score drops.
            # If pop=0, penalty=0. Score stays.
            
            penalty = popularity_norm * boost_factor
            adjusted_score = relevance * (1 + (1 - popularity_norm) * boost_factor)
            
            reranked.append((movie_id, adjusted_score))
        
        # Sort by adjusted score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in reranked[:k]]
    
    def combined_diversity(self, recommendations, movie_features, movies_df, 
                          ratings_df, k=10):
        """
        Apply multiple diversity strategies in sequence
        
        1. MMR for feature-based diversity
        2. Genre balancing
        3. Popularity debiasing
        """
        # Step 1: MMR
        diverse_recs = self.mmr_rerank(recommendations, movie_features, k=k*2)
        
        # Step 2: Genre diversity
        diverse_recs = self.genre_diversity_rerank(diverse_recs, movies_df, k=k*2)
        
        # Step 3: Popularity debiasing
        diverse_recs = self.popularity_debiasing(diverse_recs, ratings_df, k=k)
        
        return diverse_recs


# ============================================================================
# EXPLAINABILITY ENGINE
# ============================================================================

class ExplainabilityEngine:
    """
    Generates natural language explanations for recommendations
    """
    
    def __init__(self, movies_df, tags_df, ratings_df):
        self.movies_df = movies_df
        self.tags_df = tags_df
        self.ratings_df = ratings_df
        
        # Prepare data structures
        self.movie_genres = self._prepare_genres()
        self.movie_tags = self._prepare_tags()
        self.movie_titles = self._prepare_titles()
    
    def _prepare_genres(self):
        """Create genre dictionary"""
        genre_dict = {}
        for _, row in self.movies_df.iterrows():
            genre_dict[row['movieId']] = set(str(row['genres']).split('|'))
        return genre_dict
    
    def _prepare_tags(self):
        """Create tag dictionary"""
        tag_dict = defaultdict(set)
        for _, row in self.tags_df.iterrows():
            tag_dict[row['movieId']].add(str(row['tag']).lower())
        return tag_dict
    
    def _prepare_titles(self):
        """Create title dictionary"""
        return dict(zip(self.movies_df['movieId'], self.movies_df['title']))
    
    def generate_explanation(self, user_id, recommended_movie_id, max_examples=2):
        """
        Generate comprehensive explanation for a recommendation
        
        Returns:
            String explanation combining multiple reasons
        """
        explanations = []
        
        # 1. Genre-based explanation
        genre_exp = self._explain_by_genre(user_id, recommended_movie_id, max_examples)
        if genre_exp:
            explanations.append(genre_exp)
        
        # 2. Tag-based explanation
        tag_exp = self._explain_by_tags(user_id, recommended_movie_id, max_examples)
        if tag_exp:
            explanations.append(tag_exp)
        
        # 3. Collaborative explanation
        collab_exp = self._explain_by_collaboration(user_id, recommended_movie_id)
        if collab_exp:
            explanations.append(collab_exp)
        
        # Combine explanations
        if not explanations:
            return "This movie matches your viewing preferences."
        
        if len(explanations) == 1:
            return explanations[0] + "."
        
        # Join multiple explanations
        main_explanation = explanations[0]
        additional = " Additionally, " + " and ".join(explanations[1:])
        
        return main_explanation + "." + additional + "."
    
    def _explain_by_genre(self, user_id, recommended_movie_id, max_examples=2):
        """Explain based on genre overlap with user's favorites"""
        # Get user's highly rated movies
        user_ratings = self.ratings_df[
            (self.ratings_df['userId'] == user_id) &
            (self.ratings_df['rating'] >= 4.0)
        ].sort_values('rating', ascending=False)
        
        if len(user_ratings) == 0:
            return None
        
        # Get genres of recommended movie
        rec_genres = self.movie_genres.get(recommended_movie_id, set())
        if not rec_genres or rec_genres == {'(no genres listed)'}:
            return None
        
        # Find movies with genre overlap
        similar_movies = []
        for _, row in user_ratings.head(10).iterrows():
            movie_id = row['movieId']
            movie_genres = self.movie_genres.get(movie_id, set())
            overlap = rec_genres.intersection(movie_genres)
            
            if overlap and overlap != {'(no genres listed)'}:
                title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
                similar_movies.append((title, overlap))
        
        if not similar_movies:
            return None
        
        # Create explanation
        if len(similar_movies) == 1:
            movie_name = similar_movies[0][0]
            genres = list(similar_movies[0][1])[:2]
            genre_str = " and ".join(genres)
            return f"Because you enjoyed '{movie_name}', which shares the {genre_str} genre(s)"
        else:
            # Multiple similar movies
            movie_names = [m[0] for m in similar_movies[:max_examples]]
            all_genres = set()
            for _, genres in similar_movies[:max_examples]:
                all_genres.update(genres)
            
            genre_list = list(all_genres)[:3]
            
            if len(movie_names) == 2:
                movies_str = f"'{movie_names[0]}' and '{movie_names[1]}'"
            else:
                movies_str = ", ".join([f"'{m}'" for m in movie_names])
            
            genre_str = ", ".join(genre_list)
            return f"Because you enjoyed {movies_str}, which share genres like {genre_str}"
    
    def _explain_by_tags(self, user_id, recommended_movie_id, max_examples=2):
        """Explain based on tag similarity"""
        # Get user's top-rated movies
        user_ratings = self.ratings_df[
            (self.ratings_df['userId'] == user_id) &
            (self.ratings_df['rating'] >= 4.0)
        ].sort_values('rating', ascending=False)
        
        if len(user_ratings) == 0:
            return None
        
        # Get tags of recommended movie
        rec_tags = self.movie_tags.get(recommended_movie_id, set())
        if not rec_tags:
            return None
        
        # Find movies with tag overlap
        similar_tag_movies = []
        for _, row in user_ratings.head(10).iterrows():
            movie_id = row['movieId']
            movie_tags = self.movie_tags.get(movie_id, set())
            overlap = rec_tags.intersection(movie_tags)
            
            if overlap:
                title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
                similar_tag_movies.append((title, overlap))
        
        if not similar_tag_movies:
            return None
        
        # Create explanation
        movie_name = similar_tag_movies[0][0]
        tags = list(similar_tag_movies[0][1])[:2]
        tag_str = "' and '".join(tags)
        
        return f"it has similar themes to '{movie_name}', including tags like '{tag_str}'"
    
    def _explain_by_collaboration(self, user_id, recommended_movie_id):
        """Explain based on similar users"""
        # Find users who rated this movie highly
        movie_ratings = self.ratings_df[
            (self.ratings_df['movieId'] == recommended_movie_id) &
            (self.ratings_df['rating'] >= 4.0)
        ]
        
        if len(movie_ratings) == 0:
            return None
        
        # Get user's rated movies
        user_rated = set(self.ratings_df[
            self.ratings_df['userId'] == user_id
        ]['movieId'])
        
        if len(user_rated) < 5:
            return None
        
        # Find similar users (who rated similar movies)
        similar_user_count = 0
        for other_user in movie_ratings['userId'].unique()[:20]:
            if other_user == user_id:
                continue
            
            other_rated = set(self.ratings_df[
                self.ratings_df['userId'] == other_user
            ]['movieId'])
            
            # Jaccard similarity
            overlap = len(user_rated.intersection(other_rated))
            
            if overlap >= 5:  # At least 5 common movies
                similar_user_count += 1
        
        if similar_user_count >= 3:
            return "users with similar taste to yours rated this movie highly"
        
        return None
    
    def get_movie_title(self, movie_id):
        """Get movie title by ID"""
        return self.movie_titles.get(movie_id, f"Movie {movie_id}")
    
    def bulk_explain(self, user_id, recommended_movies):
        """
        Generate explanations for multiple recommendations
        
        Returns:
            Dictionary mapping movie_id to explanation
        """
        explanations = {}
        for movie_id in recommended_movies:
            explanation = self.generate_explanation(user_id, movie_id)
            title = self.get_movie_title(movie_id)
            explanations[movie_id] = {
                'title': title,
                'explanation': explanation
            }
        
        return explanations


# ============================================================================
# COMPLETE RECOMMENDATION PIPELINE
# ============================================================================

class RecommendationPipeline:
    """
    End-to-end pipeline combining:
    1. Recommendation generation
    2. Diversity optimization
    3. Explanation generation
    """
    
    def __init__(self, recommender, diversity_optimizer, explainer):
        self.recommender = recommender
        self.diversity_optimizer = diversity_optimizer
        self.explainer = explainer
    
    def recommend_with_diversity_and_explanations(
        self,
        user_id,
        ratings_df,
        all_movies,
        movie_features,
        movies_df,
        k=10,
        diversity_method='combined'
    ):
        """
        Generate diverse recommendations with explanations
        
        Args:
            user_id: User ID to generate recommendations for
            ratings_df: Ratings dataframe
            all_movies: List of all movie IDs
            movie_features: Movie feature matrix
            movies_df: Movie metadata
            k: Number of recommendations
            diversity_method: 'mmr', 'genre', 'popularity', or 'combined'
        
        Returns:
            List of tuples: (movie_id, title, explanation)
        """
        # Get user's rated movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        rated_movies = set(user_ratings['movieId'])
        
        # Step 1: Generate initial recommendations (get more than k)
        initial_recs = self.recommender.recommend(
            user_id,
            ratings_df,
            all_movies,
            rated_movies,
            k=k*3  # Get 3x to allow for diversity optimization
        )
        
        if not initial_recs:
            return []
        
        # Step 2: Apply diversity optimization
        if diversity_method == 'mmr':
            diverse_recs = self.diversity_optimizer.mmr_rerank(
                initial_recs, movie_features, k=k
            )
        elif diversity_method == 'genre':
            diverse_recs = self.diversity_optimizer.genre_diversity_rerank(
                initial_recs, movies_df, k=k
            )
        elif diversity_method == 'popularity':
            diverse_recs = self.diversity_optimizer.popularity_debiasing(
                initial_recs, ratings_df, k=k
            )
        else:  # combined
            diverse_recs = self.diversity_optimizer.combined_diversity(
                initial_recs, movie_features, movies_df, ratings_df, k=k
            )
        
        # Step 3: Generate explanations
        results = []
        for movie_id in diverse_recs:
            title = self.explainer.get_movie_title(movie_id)
            explanation = self.explainer.generate_explanation(user_id, movie_id)
            results.append((movie_id, title, explanation))
        
        return results


if __name__ == "__main__":
    print("ReelSense Part 3: Diversity & Explainability")
    print("This file implements diversity optimization and explanation generation")
