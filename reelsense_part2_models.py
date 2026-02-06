"""
ReelSense Part 2: Recommendation Models
Implements all recommendation algorithms: Popularity, CF, SVD, Content-Based, and Hybrid
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# POPULARITY-BASED RECOMMENDER (BASELINE)
# ============================================================================

class PopularityRecommender:
    """
    Baseline recommender using weighted rating formula
    Similar to IMDB's Top 250 formula
    """
    
    def __init__(self):
        self.popular_movies = None
        self.movie_stats = None
    
    def fit(self, ratings_df):
        """Train the model"""
        print("\n[Popularity] Training...")
        
        # Calculate statistics per movie
        self.movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        self.movie_stats.columns = ['avg_rating', 'rating_count']
        
        # Weighted rating (Bayesian average)
        # Avoid division by zero if empty
        if len(self.movie_stats) == 0:
            print("[Popularity] No ratings found.")
            return

        C = self.movie_stats['avg_rating'].mean()  # Mean rating across all movies
        m = self.movie_stats['rating_count'].quantile(0.60)  # Minimum votes required
        
        def weighted_rating(row):
            v = row['rating_count']
            R = row['avg_rating']
            if (v + m) == 0:
                return 0
            return (v/(v+m) * R) + (m/(v+m) * C)
        
        self.movie_stats['weighted_score'] = self.movie_stats.apply(weighted_rating, axis=1)
        self.popular_movies = self.movie_stats.sort_values('weighted_score', ascending=False)
        
        print(f"[Popularity] Trained on {len(self.movie_stats)} movies")
    
    def recommend(self, user_id, rated_movies=None, k=10):
        """Generate recommendations"""
        if self.popular_movies is None:
            return []
            
        if rated_movies is None:
            rated_movies = set()
        
        # Filter out already rated movies
        candidates = self.popular_movies[~self.popular_movies.index.isin(rated_movies)]
        
        return candidates.head(k).index.tolist()


# ============================================================================
# USER-BASED COLLABORATIVE FILTERING
# ============================================================================

class UserBasedCF:
    """User-based collaborative filtering using cosine similarity"""
    
    def __init__(self, k_neighbors=20, min_support=3):
        self.k_neighbors = k_neighbors
        self.min_support = min_support
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_mean = None
    
    def fit(self, ratings_df):
        """Train the model"""
        print("\n[User-Based CF] Training...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Calculate user mean ratings (for centering)
        self.user_mean = self.user_item_matrix.replace(0, np.nan).mean(axis=1)
        
        # Calculate user similarity
        print("[User-Based CF] Computing user similarities...")
        # Check if matrix is too large for memory, though for MovieLens Small it's fine
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print(f"[User-Based CF] Trained on {len(self.user_item_matrix)} users")
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_similarity.index:
            return 3.0  # Return average rating for cold users
        
        if movie_id not in self.user_item_matrix.columns:
            return 3.0  # Return average rating for cold items
        
        # Find similar users who rated this movie
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:]
        
        numerator = 0
        denominator = 0
        count = 0
        
        for sim_user, similarity in similar_users.items():
            if self.user_item_matrix.loc[sim_user, movie_id] > 0:
                numerator += similarity * self.user_item_matrix.loc[sim_user, movie_id]
                denominator += abs(similarity)
                count += 1
                
                if count >= self.k_neighbors:
                    break
        
        if denominator == 0 or count < self.min_support:
            return self.user_mean.get(user_id, 3.0)
        
        predicted = numerator / denominator
        return np.clip(predicted, 0.5, 5.0)
    
    def recommend(self, user_id, rated_movies=None, k=10):
        """Generate recommendations"""
        if self.user_similarity is None or user_id not in self.user_similarity.index:
            return []
        
        if rated_movies is None:
            # We assume user_item_matrix has all history
            rated_movies = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
        
        # Optimization: Only consider movies rated by top-k similar users
        # This drastically reduces the search space compared to all_movies
        
        # Get top k neighbors
        if user_id in self.user_similarity.index:
            sim_users = self.user_similarity[user_id].sort_values(ascending=False)[1:self.k_neighbors*2].index
        else:
            sim_users = []
            
        # Collect candidate movies from neighbors
        candidates = set()
        for u in sim_users:
            # Get movies rated by neighbor
            u_movies = self.user_item_matrix.loc[u]
            u_rated = u_movies[u_movies > 0].index
            candidates.update(u_rated)
        
        # Remove already rated
        candidates = candidates - rated_movies
        
        predictions = []
        for movie_id in candidates:
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        # Sort and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in predictions[:k]]


# ============================================================================
# ITEM-BASED COLLABORATIVE FILTERING
# ============================================================================

class ItemBasedCF:
    """Item-based collaborative filtering"""
    
    def __init__(self, k_neighbors=20):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.user_item_matrix = None
    
    def fit(self, ratings_df):
        """Train the model"""
        print("\n[Item-Based CF] Training...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Calculate item similarity (transpose matrix)
        print("[Item-Based CF] Computing item similarities...")
        # Calculate item similarity (transpose matrix)
        print("[Item-Based CF] Computing item similarities...")
        similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Set diagonal to 0
        np.fill_diagonal(similarity_matrix, 0)
        
        self.item_similarity = pd.DataFrame(
            similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print(f"[Item-Based CF] Pre-computing neighbors for {len(self.user_item_matrix.columns)} items...")
        # Pre-compute neighbors to speed up recommendation
        # Store as dictionary: movie_id -> list of (sim_movie_id, score)
        self.neighbors = {}
        for item in self.item_similarity.columns:
            # Get top K similar items (excluding self which is 0)
            # nlargest is faster than sort_values
            top = self.item_similarity[item].nlargest(self.k_neighbors)
            self.neighbors[item] = list(zip(top.index, top.values))
            
        print(f"[Item-Based CF] Training complete")
    
    def recommend(self, user_id, rated_movies=None, k=10):
        """Generate recommendations"""
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's rated movies and ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        user_ratings = user_ratings[user_ratings > 0]
        
        if len(user_ratings) == 0:
            return []
        
        if rated_movies is None:
            rated_movies = set(user_ratings.index)
        
        # Calculate scores for all unrated movies
        scores = defaultdict(float)
        weights = defaultdict(float)
        
        for rated_movie, rating in user_ratings.items():
            if rated_movie not in self.neighbors:
                continue
            
            # Get pre-computed similar items
            similar_items = self.neighbors[rated_movie]
            
            for movie_id, similarity in similar_items:
                if movie_id not in rated_movies:
                    scores[movie_id] += similarity * rating
                    weights[movie_id] += abs(similarity)
        
        # Normalize scores
        recommendations = []
        for movie_id, score in scores.items():
            if weights[movie_id] > 0:
                normalized_score = score / weights[movie_id]
                recommendations.append((movie_id, normalized_score))
        
        # Sort and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in recommendations[:k]]


# ============================================================================
# MATRIX FACTORIZATION (SVD)
# ============================================================================

class SimpleSVD:
    """
    Simplified SVD implementation using numpy.
    For production, consider using surprise library's SVD for better performance.
    """
    
    def __init__(self, n_factors=20, n_iterations=20, learning_rate=0.01, regularization=0.02):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lr = learning_rate
        self.reg = regularization
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_to_idx = {}
        self.item_to_idx = {}
    
    def fit(self, ratings_df):
        """Train SVD model using SGD"""
        print(f"\n[SVD] Training with {self.n_factors} factors...")
        
        # Initialize
        users = ratings_df['userId'].unique()
        items = ratings_df['movieId'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        n_users = len(users)
        n_items = len(items)
        
        # Random initialization
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()
        
        # Training
        for iteration in range(self.n_iterations):
            # Shuffle data
            shuffled = ratings_df.sample(frac=1)
            
            for _, row in shuffled.iterrows():
                user = row['userId']
                item = row['movieId']
                rating = row['rating']
                
                u_idx = self.user_to_idx[user]
                i_idx = self.item_to_idx[item]
                
                # Predict
                pred = (self.global_mean + 
                       self.user_bias[u_idx] + 
                       self.item_bias[i_idx] + 
                       np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
                
                # Error
                error = rating - pred
                
                # Update biases
                self.user_bias[u_idx] += self.lr * (error - self.reg * self.user_bias[u_idx])
                self.item_bias[i_idx] += self.lr * (error - self.reg * self.item_bias[i_idx])
                
                # Update factors
                user_f = self.user_factors[u_idx].copy()
                self.user_factors[u_idx] += self.lr * (error * self.item_factors[i_idx] - 
                                                       self.reg * self.user_factors[u_idx])
                self.item_factors[i_idx] += self.lr * (error * user_f - 
                                                       self.reg * self.item_factors[i_idx])
            
            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iterations} complete")
        
        print(f"[SVD] Training complete!")
    
    def predict(self, user_id, movie_id):
        """Predict rating"""
        if user_id not in self.user_to_idx or movie_id not in self.item_to_idx:
            return self.global_mean
        
        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[movie_id]
        
        pred = (self.global_mean + 
               self.user_bias[u_idx] + 
               self.item_bias[i_idx] + 
               np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        
        return np.clip(pred, 0.5, 5.0)
    
    def recommend(self, user_id, all_movies, rated_movies=None, k=10):
        """Generate recommendations"""
        if user_id not in self.user_to_idx:
            return []
        
        if rated_movies is None:
            rated_movies = set()
        
        # Vectorized prediction for speed
        if user_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[user_id]
        
        # Filter movies that exist in our model
        valid_movies = [m for m in all_movies if m in self.item_to_idx]
        valid_indices = [self.item_to_idx[m] for m in valid_movies]
        
        if not valid_indices:
            return []
            
        # Calculate all scores at once
        # Score = mean + user_bias + item_bias + user_factors . item_factors
        
        user_f = self.user_factors[u_idx] # (F,)
        item_fs = self.item_factors[valid_indices] # (N, F)
        item_bs = self.item_bias[valid_indices] # (N,)
        
        # Broadcasting adds scalar + (N,) + scalar + (N,)
        scores = self.global_mean + self.user_bias[u_idx] + item_bs + np.dot(item_fs, user_f)
        
        # Create pairs and sort
        predictions = []
        for i, movie_id in enumerate(valid_movies):
            if movie_id not in rated_movies:
                predictions.append((movie_id, scores[i]))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in predictions[:k]]


# ============================================================================
# CONTENT-BASED RECOMMENDER
# ============================================================================

class ContentBasedRecommender:
    """Content-based filtering using movie features"""
    
    def __init__(self):
        self.movie_features = None
        self.movie_similarity = None
    
    def fit(self, movie_features):
        """Train model with pre-computed features"""
        print("\n[Content-Based] Training...")
        
        self.movie_features = movie_features
        
        # Calculate movie similarity
        print("[Content-Based] Computing similarity matrix...")
        # Calculate movie similarity
        print("[Content-Based] Computing similarity matrix...")
        similarity_matrix = cosine_similarity(movie_features)
        
        # Set diagonal to 0
        np.fill_diagonal(similarity_matrix, 0)
        
        self.movie_similarity = pd.DataFrame(
            similarity_matrix,
            index=movie_features.index,
            columns=movie_features.index
        )
        
        print(f"[Content-Based] Trained on {len(movie_features)} movies")
    
    def recommend(self, user_id, ratings_df, rated_movies=None, k=10):
        """Generate recommendations based on user's top-rated movies"""
        # Get user's highly rated movies (>= 4.0)
        user_ratings = ratings_df[
            (ratings_df['userId'] == user_id) & 
            (ratings_df['rating'] >= 4.0)
        ].sort_values('rating', ascending=False)
        
        if len(user_ratings) == 0:
            # Fallback to all rated movies
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return []

        if rated_movies is None:
             rated_movies = set()

        top_movies = user_ratings['movieId'].head(5).tolist()
        
        # Find similar movies
        scores = defaultdict(float)
        
        for movie_id in top_movies:
            if movie_id not in self.movie_similarity.index:
                continue
            
            # Use top 50 similar movies to speed up
            similar_movies = self.movie_similarity[movie_id].sort_values(ascending=False).head(50)
            
            for sim_movie, similarity in similar_movies.items():
                if sim_movie not in rated_movies:
                    scores[sim_movie] += similarity
        
        # Sort and return top-k
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in recommendations[:k]]


# ============================================================================
# HYBRID RECOMMENDER
# ============================================================================

class HybridRecommender:
    """
    Hybrid model combining multiple approaches
    Uses weighted linear combination
    """
    
    def __init__(self, weights=None):
        if weights is None:
            weights = {
                'popularity': 0.1,
                'user_cf': 0.25,
                'item_cf': 0.25,
                'svd': 0.25,
                'content': 0.15
            }
        self.weights = weights
        
        self.pop_model = PopularityRecommender()
        self.user_cf = UserBasedCF()
        self.item_cf = ItemBasedCF()
        self.svd = SimpleSVD()
        self.content = ContentBasedRecommender()
    
    def fit(self, ratings_df, movie_features):
        """Train all sub-models"""
        print("\n" + "="*60)
        print("TRAINING HYBRID MODEL")
        print("="*60)
        
        self.pop_model.fit(ratings_df)
        self.user_cf.fit(ratings_df)
        self.item_cf.fit(ratings_df)
        self.svd.fit(ratings_df)
        self.content.fit(movie_features)
        
        print("\n✓ All models trained successfully!")
    
    def recommend(self, user_id, ratings_df, all_movies, rated_movies=None, k=10):
        """Generate hybrid recommendations"""
        if rated_movies is None:
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            rated_movies = set(user_ratings['movieId'])
        
        # Get recommendations from each model (fetch more candidates to blend)
        # Using k=50 for candidates
        pop_recs = self.pop_model.recommend(user_id, rated_movies, k=50)
        user_cf_recs = self.user_cf.recommend(user_id, rated_movies, k=50)
        item_cf_recs = self.item_cf.recommend(user_id, rated_movies, k=50)
        svd_recs = self.svd.recommend(user_id, all_movies, rated_movies, k=50)
        content_recs = self.content.recommend(user_id, ratings_df, rated_movies, k=50)
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        # Helper to add scores
        def add_scores(recs, weight):
            for idx, movie_id in enumerate(recs):
                # Simple Rank-based scoring: 1 / (rank + 1)
                score = 1.0 / (idx + 1)
                combined_scores[movie_id] += weight * score

        add_scores(pop_recs, self.weights['popularity'])
        add_scores(user_cf_recs, self.weights['user_cf'])
        add_scores(item_cf_recs, self.weights['item_cf'])
        add_scores(svd_recs, self.weights['svd'])
        add_scores(content_recs, self.weights['content'])
        
        # Sort and return top-k
        final_recommendations = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [movie_id for movie_id, _ in final_recommendations[:k]]


if __name__ == "__main__":
    print("ReelSense Part 2: Recommendation Models")
    print("This file contains all recommendation algorithms")
