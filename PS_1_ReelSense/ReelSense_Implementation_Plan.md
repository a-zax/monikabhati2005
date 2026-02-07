# 🎬 ReelSense: Complete Implementation Plan & Architecture

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Implementation Timeline](#implementation-timeline)
4. [Detailed Implementation Guide](#detailed-implementation-guide)
5. [Validation Strategy](#validation-strategy)
6. [Tips for Best Outcome](#tips-for-best-outcome)

---

## 🎯 Project Overview

**Goal**: Build an explainable, diverse movie recommender system that:
- Provides personalized Top-K recommendations
- Avoids popularity bias through diversity optimization
- Generates natural language explanations
- Achieves high scores on ranking, diversity, and novelty metrics

**Dataset**: MovieLens Latest Small
- 100,836 ratings from 610 users on 9,742 movies
- Files: ratings.csv, movies.csv, tags.csv, links.csv

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING LAYER                  │
│  • Load & Clean Data                                         │
│  • Time-based Train-Test Split (Leave-Last-N)               │
│  • Feature Engineering (Genre vectors, Tag embeddings)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   RECOMMENDATION MODELS LAYER                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Popularity   │  │ Collaborative│  │   Matrix     │     │
│  │   Based      │  │  Filtering   │  │Factorization │     │
│  │  (Baseline)  │  │  (User/Item) │  │    (SVD)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │         HYBRID MODEL (Weighted Ensemble)         │       │
│  │  • Combines CF + Content-Based + MF              │       │
│  │  • Genre + Tag features for content filtering    │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               DIVERSITY OPTIMIZATION LAYER                   │
│  • Re-rank results to maximize diversity                    │
│  • MMR (Maximal Marginal Relevance) algorithm               │
│  • Genre diversity enforcement                              │
│  • Popularity-based penalization                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY LAYER                        │
│  • Tag similarity analysis                                   │
│  • Genre overlap detection                                   │
│  • Similar user neighborhood explanation                     │
│  • Natural language generation                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                          │
│  Ranking: NDCG@K, MAP@K, Precision@K, Recall@K             │
│  Diversity: Coverage, Intra-List Diversity, Gini Index      │
│  Novelty: Long-tail %, Serendipity Score                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📅 Implementation Timeline

### **Phase 1: Setup & EDA (Day 1)**
- [ ] Environment setup and library installation
- [ ] Data loading and initial exploration
- [ ] Comprehensive EDA with visualizations
- [ ] Data quality checks

### **Phase 2: Data Preprocessing (Day 1-2)**
- [ ] Clean and standardize data
- [ ] Time-based train-test split
- [ ] Feature engineering (genre vectors, tag processing)
- [ ] User-item interaction matrix construction

### **Phase 3: Baseline Models (Day 2)**
- [ ] Popularity-based recommender
- [ ] Simple collaborative filtering
- [ ] Evaluate baselines

### **Phase 4: Advanced Models (Day 2-3)**
- [ ] Matrix Factorization (SVD)
- [ ] User-User and Item-Item CF
- [ ] Content-based filtering (genre + tags)
- [ ] Hybrid model integration

### **Phase 5: Diversity Optimization (Day 3)**
- [ ] Implement MMR algorithm
- [ ] Genre diversity enforcement
- [ ] Popularity debiasing techniques
- [ ] Fine-tune diversity parameters

### **Phase 6: Explainability (Day 3-4)**
- [ ] Build explanation generation system
- [ ] Tag similarity explanations
- [ ] Genre overlap explanations
- [ ] User neighborhood explanations
- [ ] Natural language templates

### **Phase 7: Evaluation & Tuning (Day 4)**
- [ ] Comprehensive metric evaluation
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Model comparison

### **Phase 8: Documentation & Delivery (Day 4-5)**
- [ ] Clean and document code
- [ ] Create visualizations
- [ ] Write comprehensive report
- [ ] Prepare final submission

---

## 🔧 Detailed Implementation Guide

### **1. Environment Setup**

```python
# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split
import warnings
warnings.filterwarnings('ignore')

# For diversity metrics
from collections import Counter
import random
```

---

### **2. Data Loading & Preprocessing**

**Key Steps:**

1. **Load Data**
```python
# Load all datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
links = pd.read_csv('links.csv')
```

2. **Data Cleaning**
- Handle missing values
- Standardize genres (split by '|')
- Clean and lowercase tags
- Parse timestamps to datetime

3. **Time-based Split**
```python
# Leave-last-N approach (N=3 for test)
def leave_last_n_split(ratings_df, n=3):
    train_data = []
    test_data = []
    
    for user_id in ratings_df['userId'].unique():
        user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp')
        
        if len(user_ratings) > n:
            train_data.append(user_ratings.iloc[:-n])
            test_data.append(user_ratings.iloc[-n:])
        else:
            train_data.append(user_ratings)
    
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    return train_df, test_df
```

4. **Feature Engineering**
- Create genre binary vectors
- TF-IDF for tags
- User activity features
- Movie popularity features

---

### **3. Exploratory Data Analysis**

**Must-Have Visualizations:**

1. **Rating Distribution**
```python
plt.figure(figsize=(10, 6))
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
```

2. **Genre Popularity vs Average Rating**
```python
# Genre analysis with ratings
genre_stats = analyze_genre_performance(movies, ratings)
plt.figure(figsize=(14, 8))
sns.scatterplot(data=genre_stats, x='count', y='avg_rating', size='count')
plt.title('Genre Popularity vs Average Rating')
```

3. **User Activity Distribution**
```python
user_activity = ratings.groupby('userId').size()
plt.figure(figsize=(10, 6))
plt.hist(user_activity, bins=50, edgecolor='black')
plt.title('User Activity Distribution')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Number of Users')
```

4. **Long-tail Analysis**
```python
movie_popularity = ratings.groupby('movieId').size().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
plt.plot(range(len(movie_popularity)), movie_popularity.values)
plt.title('Long-tail Distribution of Movie Popularity')
plt.xlabel('Movie Rank')
plt.ylabel('Number of Ratings')
plt.yscale('log')
```

5. **Rating Trends Over Time**
```python
ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
rating_trends = ratings.groupby(ratings['date'].dt.to_period('M'))['rating'].mean()
rating_trends.plot(figsize=(12, 6))
plt.title('Average Rating Trends Over Time')
```

---

### **4. Recommendation Models Implementation**

#### **A. Popularity-Based (Baseline)**

```python
class PopularityRecommender:
    def __init__(self):
        self.popular_movies = None
    
    def fit(self, ratings_df):
        # Calculate popularity score (weighted rating)
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        movie_stats.columns = ['avg_rating', 'rating_count']
        
        # Weighted rating formula (IMDB-style)
        C = movie_stats['avg_rating'].mean()
        m = movie_stats['rating_count'].quantile(0.60)
        
        def weighted_rating(x):
            v = x['rating_count']
            R = x['avg_rating']
            return (v/(v+m) * R) + (m/(v+m) * C)
        
        movie_stats['score'] = movie_stats.apply(weighted_rating, axis=1)
        self.popular_movies = movie_stats.sort_values('score', ascending=False)
    
    def recommend(self, user_id, k=10):
        return self.popular_movies.head(k).index.tolist()
```

#### **B. Collaborative Filtering**

```python
class UserBasedCF:
    def __init__(self, k_neighbors=20):
        self.k_neighbors = k_neighbors
        self.user_similarity = None
        self.user_item_matrix = None
    
    def fit(self, ratings_df):
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Calculate user similarity using cosine similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
    
    def recommend(self, user_id, k=10):
        if user_id not in self.user_similarity.index:
            return []
        
        # Find similar users
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:self.k_neighbors+1]
        
        # Get movies rated by similar users but not by target user
        user_movies = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        recommendations = {}
        for sim_user, similarity in similar_users.items():
            sim_user_movies = self.user_item_matrix.loc[sim_user]
            sim_user_movies = sim_user_movies[sim_user_movies > 0]
            
            for movie_id, rating in sim_user_movies.items():
                if movie_id not in user_movies:
                    if movie_id not in recommendations:
                        recommendations[movie_id] = 0
                    recommendations[movie_id] += similarity * rating
        
        # Sort and return top-k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recs[:k]]
```

#### **C. Matrix Factorization (SVD)**

```python
def train_svd_model(train_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    
    # Train SVD
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    
    return svd

def get_svd_recommendations(svd, user_id, movies_df, ratings_df, k=10):
    # Get all movies
    all_movies = movies_df['movieId'].unique()
    
    # Get movies user has already rated
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in all_movies:
        if movie_id not in rated_movies:
            pred = svd.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
    
    # Sort and return top-k
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in predictions[:k]]
```

#### **D. Content-Based Filtering**

```python
class ContentBasedRecommender:
    def __init__(self):
        self.movie_features = None
        self.movie_similarity = None
    
    def fit(self, movies_df, tags_df):
        # Create genre features
        genres = movies_df['genres'].str.get_dummies('|')
        
        # Create tag features (TF-IDF)
        movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tag_features = tfidf.fit_transform(movie_tags['tag'].fillna(''))
        tag_df = pd.DataFrame(tag_features.toarray(), index=movie_tags['movieId'])
        
        # Combine features
        self.movie_features = pd.concat([genres, tag_df], axis=1).fillna(0)
        
        # Calculate movie similarity
        self.movie_similarity = cosine_similarity(self.movie_features)
        self.movie_similarity = pd.DataFrame(
            self.movie_similarity,
            index=self.movie_features.index,
            columns=self.movie_features.index
        )
    
    def recommend(self, user_id, ratings_df, k=10):
        # Get user's top-rated movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('rating', ascending=False)
        top_movies = user_ratings.head(5)['movieId'].tolist()
        
        # Find similar movies
        recommendations = {}
        for movie_id in top_movies:
            if movie_id in self.movie_similarity.index:
                similar_movies = self.movie_similarity[movie_id].sort_values(ascending=False)[1:21]
                for sim_movie, similarity in similar_movies.items():
                    if sim_movie not in user_ratings['movieId'].values:
                        if sim_movie not in recommendations:
                            recommendations[sim_movie] = 0
                        recommendations[sim_movie] += similarity
        
        # Sort and return top-k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recs[:k]]
```

#### **E. Hybrid Model**

```python
class HybridRecommender:
    def __init__(self, weights={'cf': 0.4, 'svd': 0.4, 'content': 0.2}):
        self.weights = weights
        self.cf_model = None
        self.svd_model = None
        self.content_model = None
    
    def fit(self, ratings_df, movies_df, tags_df):
        print("Training Collaborative Filtering...")
        self.cf_model = UserBasedCF()
        self.cf_model.fit(ratings_df)
        
        print("Training SVD...")
        self.svd_model = train_svd_model(ratings_df)
        
        print("Training Content-Based...")
        self.content_model = ContentBasedRecommender()
        self.content_model.fit(movies_df, tags_df)
        
        print("Hybrid model training complete!")
    
    def recommend(self, user_id, ratings_df, k=10):
        # Get recommendations from each model
        cf_recs = self.cf_model.recommend(user_id, k=50)
        svd_recs = get_svd_recommendations(self.svd_model, user_id, movies_df, ratings_df, k=50)
        content_recs = self.content_model.recommend(user_id, ratings_df, k=50)
        
        # Combine with weighted scoring
        combined_scores = {}
        
        for movie_id in cf_recs:
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + self.weights['cf']
        
        for movie_id in svd_recs:
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + self.weights['svd']
        
        for movie_id in content_recs:
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + self.weights['content']
        
        # Sort and return top-k
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recs[:k]]
```

---

### **5. Diversity Optimization**

```python
class DiversityOptimizer:
    def __init__(self, lambda_param=0.5):
        self.lambda_param = lambda_param  # Balance between relevance and diversity
    
    def mmr_rerank(self, recommendations, movie_features, k=10):
        """
        Maximal Marginal Relevance (MMR) algorithm
        """
        selected = []
        candidates = recommendations.copy()
        
        # Select first item (highest score)
        selected.append(candidates.pop(0))
        
        while len(selected) < k and candidates:
            mmr_scores = []
            
            for candidate in candidates:
                # Relevance score (position-based)
                relevance = 1.0 / (recommendations.index(candidate) + 1)
                
                # Diversity score (minimum similarity to selected items)
                if candidate in movie_features.index:
                    similarities = []
                    for selected_item in selected:
                        if selected_item in movie_features.index:
                            sim = cosine_similarity(
                                movie_features.loc[candidate].values.reshape(1, -1),
                                movie_features.loc[selected_item].values.reshape(1, -1)
                            )[0][0]
                            similarities.append(sim)
                    
                    diversity = 1 - (max(similarities) if similarities else 0)
                else:
                    diversity = 0.5
                
                # MMR score
                mmr = self.lambda_param * relevance + (1 - self.lambda_param) * diversity
                mmr_scores.append((candidate, mmr))
            
            # Select item with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate = mmr_scores[0][0]
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        
        return selected
    
    def genre_diversity_rerank(self, recommendations, movies_df, k=10):
        """
        Ensure genre diversity in recommendations
        """
        selected = []
        genre_count = {}
        
        for movie_id in recommendations:
            if len(selected) >= k:
                break
            
            # Get movie genres
            movie_genres = movies_df[movies_df['movieId'] == movie_id]['genres'].values
            if len(movie_genres) == 0:
                continue
            
            genres = movie_genres[0].split('|')
            
            # Check if adding this movie increases diversity
            diversity_score = 0
            for genre in genres:
                if genre not in genre_count:
                    diversity_score += 1
            
            # Prefer movies that add new genres
            if diversity_score > 0 or len(selected) < 3:
                selected.append(movie_id)
                for genre in genres:
                    genre_count[genre] = genre_count.get(genre, 0) + 1
        
        return selected
    
    def popularity_debiasing(self, recommendations, ratings_df, k=10, bias_factor=0.3):
        """
        Penalize overly popular movies to promote long-tail items
        """
        movie_popularity = ratings_df.groupby('movieId').size()
        max_popularity = movie_popularity.max()
        
        reranked = []
        for movie_id in recommendations:
            popularity = movie_popularity.get(movie_id, 0)
            # Penalize popular items
            penalty = (popularity / max_popularity) * bias_factor
            adjusted_score = 1.0 - penalty
            reranked.append((movie_id, adjusted_score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in reranked[:k]]
```

---

### **6. Explainability System**

```python
class ExplainabilityEngine:
    def __init__(self, movies_df, tags_df, ratings_df):
        self.movies_df = movies_df
        self.tags_df = tags_df
        self.ratings_df = ratings_df
        self.user_item_matrix = None
        self.movie_genres = self._prepare_genres()
        self.movie_tags = self._prepare_tags()
    
    def _prepare_genres(self):
        genre_dict = {}
        for _, row in self.movies_df.iterrows():
            genre_dict[row['movieId']] = set(row['genres'].split('|'))
        return genre_dict
    
    def _prepare_tags(self):
        tag_dict = {}
        for movie_id in self.movies_df['movieId'].unique():
            movie_tags = self.tags_df[self.tags_df['movieId'] == movie_id]['tag'].tolist()
            tag_dict[movie_id] = set([tag.lower() for tag in movie_tags])
        return tag_dict
    
    def generate_explanation(self, user_id, recommended_movie_id):
        """
        Generate natural language explanation for a recommendation
        """
        explanations = []
        
        # 1. Genre-based explanation
        genre_explanation = self._explain_by_genre(user_id, recommended_movie_id)
        if genre_explanation:
            explanations.append(genre_explanation)
        
        # 2. Tag-based explanation
        tag_explanation = self._explain_by_tags(user_id, recommended_movie_id)
        if tag_explanation:
            explanations.append(tag_explanation)
        
        # 3. Collaborative explanation
        collab_explanation = self._explain_by_collaboration(user_id, recommended_movie_id)
        if collab_explanation:
            explanations.append(collab_explanation)
        
        # Combine explanations
        if not explanations:
            return "This movie matches your general preferences."
        
        return " Also, ".join(explanations)
    
    def _explain_by_genre(self, user_id, recommended_movie_id):
        # Get user's top-rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        user_ratings = user_ratings.sort_values('rating', ascending=False)
        
        top_rated = user_ratings.head(5)
        
        # Find genre overlap
        rec_genres = self.movie_genres.get(recommended_movie_id, set())
        
        similar_movies = []
        for _, row in top_rated.iterrows():
            movie_id = row['movieId']
            movie_genres = self.movie_genres.get(movie_id, set())
            overlap = rec_genres.intersection(movie_genres)
            
            if overlap:
                movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
                similar_movies.append((movie_title, overlap))
        
        if similar_movies:
            movie_names = [m[0] for m in similar_movies[:2]]
            genres = list(similar_movies[0][1])[:2]
            
            return f"Because you enjoyed {' and '.join(movie_names)}, which share genres like {', '.join(genres)}"
        
        return None
    
    def _explain_by_tags(self, user_id, recommended_movie_id):
        # Get user's top-rated movies with tags
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        user_ratings = user_ratings.sort_values('rating', ascending=False).head(10)
        
        rec_tags = self.movie_tags.get(recommended_movie_id, set())
        
        if not rec_tags:
            return None
        
        similar_tags_movies = []
        for _, row in user_ratings.iterrows():
            movie_id = row['movieId']
            movie_tags = self.movie_tags.get(movie_id, set())
            overlap = rec_tags.intersection(movie_tags)
            
            if overlap:
                movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
                similar_tags_movies.append((movie_title, overlap))
        
        if similar_tags_movies:
            movie_name = similar_tags_movies[0][0]
            tags = list(similar_tags_movies[0][1])[:2]
            
            return f"it has similar themes to {movie_name}, including tags like '{', '.join(tags)}'"
        
        return None
    
    def _explain_by_collaboration(self, user_id, recommended_movie_id):
        # Find users who rated this movie highly
        movie_ratings = self.ratings_df[
            (self.ratings_df['movieId'] == recommended_movie_id) & 
            (self.ratings_df['rating'] >= 4.0)
        ]
        
        if len(movie_ratings) == 0:
            return None
        
        # Find if any of these users rated similar movies as current user
        user_rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        
        similar_users = 0
        for other_user in movie_ratings['userId'].unique():
            other_user_movies = set(self.ratings_df[self.ratings_df['userId'] == other_user]['movieId'])
            overlap = len(user_rated_movies.intersection(other_user_movies))
            
            if overlap >= 5:
                similar_users += 1
        
        if similar_users > 0:
            return f"users with similar taste to yours rated this movie highly"
        
        return None
    
    def get_movie_title(self, movie_id):
        title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values
        return title[0] if len(title) > 0 else f"Movie {movie_id}"
```

---

### **7. Evaluation Metrics**

```python
class RecommenderEvaluator:
    def __init__(self):
        self.metrics = {}
    
    # A. Ranking Metrics
    def precision_at_k(self, recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        if len(recommended_k) == 0:
            return 0.0
        
        return len(recommended_set.intersection(relevant_set)) / len(recommended_k)
    
    def recall_at_k(self, recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        if len(relevant_set) == 0:
            return 0.0
        
        return len(recommended_set.intersection(relevant_set)) / len(relevant_set)
    
    def ndcg_at_k(self, recommended, relevant, k=10):
        """
        Normalized Discounted Cumulative Gain
        """
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
        """
        Mean Average Precision
        """
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
    
    # B. Diversity Metrics
    def catalog_coverage(self, all_recommendations, total_items):
        """
        Percentage of catalog covered by recommendations
        """
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        
        return len(unique_items) / total_items
    
    def intra_list_diversity(self, recommendations, movie_features):
        """
        Average pairwise distance within recommendation list
        """
        if len(recommendations) < 2:
            return 0.0
        
        distances = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                movie_i = recommendations[i]
                movie_j = recommendations[j]
                
                if movie_i in movie_features.index and movie_j in movie_features.index:
                    sim = cosine_similarity(
                        movie_features.loc[movie_i].values.reshape(1, -1),
                        movie_features.loc[movie_j].values.reshape(1, -1)
                    )[0][0]
                    distances.append(1 - sim)  # Distance = 1 - similarity
        
        return np.mean(distances) if distances else 0.0
    
    def gini_index(self, recommendations):
        """
        Gini coefficient to measure popularity bias
        Lower is better (more equal distribution)
        """
        item_counts = Counter(recommendations)
        counts = sorted(item_counts.values())
        n = len(counts)
        
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(counts)
        return (2 * np.sum((np.arange(1, n + 1)) * counts) / (n * cumsum[-1])) - (n + 1) / n
    
    # C. Novelty Metrics
    def novelty_score(self, recommendations, item_popularity):
        """
        Average novelty (inverse popularity) of recommendations
        """
        if len(recommendations) == 0:
            return 0.0
        
        max_pop = max(item_popularity.values())
        novelty_scores = []
        
        for item in recommendations:
            popularity = item_popularity.get(item, 1)
            novelty = -np.log2(popularity / max_pop) if popularity > 0 else 0
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def long_tail_percentage(self, recommendations, item_popularity, percentile=80):
        """
        Percentage of recommendations from long-tail (bottom percentile)
        """
        threshold = np.percentile(list(item_popularity.values()), percentile)
        long_tail_count = sum(1 for item in recommendations if item_popularity.get(item, 0) <= threshold)
        
        return long_tail_count / len(recommendations) if len(recommendations) > 0 else 0.0
    
    def evaluate_all(self, recommended, relevant, all_recommendations, movie_features, 
                     item_popularity, total_items, k=10):
        """
        Comprehensive evaluation
        """
        results = {
            'Precision@K': self.precision_at_k(recommended, relevant, k),
            'Recall@K': self.recall_at_k(recommended, relevant, k),
            'NDCG@K': self.ndcg_at_k(recommended, relevant, k),
            'MAP@K': self.map_at_k(recommended, relevant, k),
            'Catalog_Coverage': self.catalog_coverage(all_recommendations, total_items),
            'Intra_List_Diversity': self.intra_list_diversity(recommended, movie_features),
            'Gini_Index': self.gini_index([item for recs in all_recommendations for item in recs]),
            'Novelty_Score': self.novelty_score(recommended, item_popularity),
            'Long_Tail_%': self.long_tail_percentage(recommended, item_popularity)
        }
        
        return results
```

---

## ✅ Validation Strategy

### **1. Cross-Validation**
- Use time-based k-fold validation (not random)
- Ensure temporal consistency (train on past, test on future)
- Validate on multiple user segments (active users, cold users, etc.)

### **2. A/B Testing Simulation**
- Compare hybrid model vs individual models
- Test different diversity parameters
- Measure user satisfaction proxies

### **3. Statistical Significance**
- Use paired t-tests to compare models
- Calculate confidence intervals for metrics
- Ensure results are reproducible

### **4. Qualitative Validation**
- Manually review sample recommendations
- Check explanation quality
- Verify diversity in practice

---

## 🏆 Tips for Best Outcome

### **1. Code Quality**
- ✅ Clean, well-commented code
- ✅ Modular design (separate classes/functions)
- ✅ Use configuration files for hyperparameters
- ✅ Include requirements.txt

### **2. Visualizations**
- ✅ High-quality plots (300 DPI)
- ✅ Clear labels and titles
- ✅ Color-blind friendly palettes
- ✅ Consistent styling

### **3. Report Writing**
- ✅ Executive summary (1 page)
- ✅ Clear problem statement
- ✅ Methodology with justifications
- ✅ Results with statistical analysis
- ✅ Limitations and future work
- ✅ Proper citations (APA/IEEE format)

### **4. Differentiation Factors**
- ✅ Novel diversity optimization approach
- ✅ High-quality natural language explanations
- ✅ Comprehensive evaluation across all metrics
- ✅ User-centric design considerations
- ✅ Scalability discussion

### **5. Common Pitfalls to Avoid**
- ❌ Data leakage in train-test split
- ❌ Ignoring cold-start problem
- ❌ Over-optimizing for one metric
- ❌ Poor code documentation
- ❌ Lack of error handling
- ❌ Missing baseline comparisons

---

## 📊 Expected Results Benchmark

**Good Performance:**
- NDCG@10: > 0.25
- Precision@10: > 0.15
- Catalog Coverage: > 30%
- Intra-List Diversity: > 0.5
- Long-tail %: > 20%

**Excellent Performance:**
- NDCG@10: > 0.35
- Precision@10: > 0.25
- Catalog Coverage: > 50%
- Intra-List Diversity: > 0.7
- Long-tail %: > 35%

---

## 🎯 Final Checklist

**Before Submission:**
- [ ] All code runs without errors
- [ ] All visualizations saved in high resolution
- [ ] Report is proofread and formatted
- [ ] References are properly cited
- [ ] README.md explains how to run the code
- [ ] Results are reproducible
- [ ] All deliverables are included
- [ ] File naming is consistent
- [ ] Code is pushed to repository (if required)
- [ ] Presentation ready (if required)

---

**Good luck with your hackathon! This implementation plan should guide you to create a comprehensive, high-quality ReelSense system. Focus on code quality, thorough evaluation, and clear explanations to stand out.**
