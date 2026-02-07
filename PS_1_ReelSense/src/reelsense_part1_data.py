"""
ReelSense: Explainable Movie Recommender System with Diversity Optimization
Complete Implementation

Author: Hackathon Participant
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import warnings
warnings.filterwarnings('ignore')

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """Load and preprocess MovieLens dataset"""
    
    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.movies = None
        self.ratings = None
        self.tags = None
        self.links = None
    
    def load_data(self):
        """Load all CSV files"""
        print("Loading datasets...")
        try:
            self.movies = pd.read_csv(os.path.join(self.data_path, 'movies.csv'))
            self.ratings = pd.read_csv(os.path.join(self.data_path, 'ratings.csv'))
            self.tags = pd.read_csv(os.path.join(self.data_path, 'tags.csv'))
            self.links = pd.read_csv(os.path.join(self.data_path, 'links.csv'))
            
            print(f"Movies: {len(self.movies)}")
            print(f"Ratings: {len(self.ratings)}")
            print(f"Tags: {len(self.tags)}")
            print(f"Users: {self.ratings['userId'].nunique()}")
            
            return self.movies, self.ratings, self.tags, self.links
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure movies.csv, ratings.csv, tags.csv, and links.csv are in the data path.")
            raise

    def clean_data(self):
        """Clean and standardize data"""
        print("\nCleaning data...")
        
        # Convert timestamp to datetime
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        if 'timestamp' in self.tags.columns:
            self.tags['datetime'] = pd.to_datetime(self.tags['timestamp'], unit='s')
        
        # Clean tags
        if len(self.tags) > 0:
            self.tags['tag'] = self.tags['tag'].astype(str).str.lower().str.strip()
            # Remove very rare tags (appear only once)
            tag_counts = self.tags['tag'].value_counts()
            valid_tags = tag_counts[tag_counts > 1].index
            self.tags = self.tags[self.tags['tag'].isin(valid_tags)]
        
        # Handle missing genres
        self.movies['genres'] = self.movies['genres'].fillna('(no genres listed)')
        
        print("Data cleaning complete!")
        
        return self.movies, self.ratings, self.tags
    
    def time_based_split(self, n_test=3):
        """
        Leave-last-N split: For each user, last N ratings go to test set
        This simulates real-world scenario where we predict future preferences
        """
        print(f"\nPerforming time-based train-test split (Leave-Last-{n_test})...")
        
        train_data = []
        test_data = []
        
        for user_id in self.ratings['userId'].unique():
            user_ratings = self.ratings[self.ratings['userId'] == user_id].sort_values('timestamp')
            
            if len(user_ratings) > n_test:
                train_data.append(user_ratings.iloc[:-n_test])
                test_data.append(user_ratings.iloc[-n_test:])
            else:
                # If user has fewer ratings than n_test, put all in training
                train_data.append(user_ratings)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"Train set: {len(train_df)} ratings")
        print(f"Test set: {len(test_df)} ratings")
        print(f"Train users: {train_df['userId'].nunique()}")
        print(f"Test users: {test_df['userId'].nunique()}")
        
        return train_df, test_df


# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

class EDAVisualizer:
    """Generate comprehensive EDA visualizations"""
    
    def __init__(self, movies, ratings, tags, output_dir='./visualizations/'):
        self.movies = movies
        self.ratings = ratings
        self.tags = tags
        self.output_dir = output_dir
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_rating_distribution(self):
        """Plot distribution of ratings"""
        plt.figure(figsize=(10, 6))
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        
        bars = plt.bar(rating_counts.index, rating_counts.values, 
                       color='steelblue', edgecolor='black', alpha=0.7, width=0.4)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Rating distribution saved")
    
    def plot_user_activity(self):
        """Plot user activity distribution"""
        user_activity = self.ratings.groupby('userId').size()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        axes[0].hist(user_activity, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_title('User Activity Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Ratings per User', fontsize=12)
        axes[0].set_ylabel('Number of Users', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        axes[1].boxplot(user_activity, vert=True)
        axes[1].set_title('User Activity Box Plot', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Ratings per User', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}user_activity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ User activity saved")
    
    def plot_genre_analysis(self):
        """Analyze genre popularity and ratings"""
        # Explode genres
        genre_data = []
        for _, row in self.movies.iterrows():
            genres = str(row['genres']).split('|')
            for genre in genres:
                genre_data.append({'movieId': row['movieId'], 'genre': genre})
        
        genre_df = pd.DataFrame(genre_data)
        
        # Merge with ratings
        genre_ratings = genre_df.merge(self.ratings, on='movieId')
        
        # Calculate statistics
        genre_stats = genre_ratings.groupby('genre').agg({
            'rating': ['mean', 'count'],
            'movieId': 'nunique'
        }).reset_index()
        genre_stats.columns = ['genre', 'avg_rating', 'rating_count', 'movie_count']
        genre_stats = genre_stats[genre_stats['genre'] != '(no genres listed)']
        genre_stats = genre_stats.sort_values('rating_count', ascending=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Genre popularity
        axes[0].barh(genre_stats['genre'], genre_stats['rating_count'], 
                     color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[0].set_title('Genre Popularity (Total Ratings)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Ratings', fontsize=12)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Genre ratings
        genre_stats_sorted = genre_stats.sort_values('avg_rating', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(genre_stats_sorted)))
        axes[1].barh(genre_stats_sorted['genre'], genre_stats_sorted['avg_rating'], 
                     color=colors, edgecolor='black', alpha=0.7)
        axes[1].set_title('Average Rating by Genre', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Average Rating', fontsize=12)
        axes[1].axvline(x=self.ratings['rating'].mean(), color='red', 
                       linestyle='--', label='Overall Average')
        axes[1].legend()
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Genre analysis saved")
    
    def plot_long_tail(self):
        """Visualize long-tail distribution"""
        movie_popularity = self.ratings.groupby('movieId').size().sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Long-tail curve
        axes[0].plot(range(len(movie_popularity)), movie_popularity.values, 
                    color='darkblue', linewidth=2)
        axes[0].set_title('Long-tail Distribution of Movie Popularity', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Movie Rank', fontsize=12)
        axes[0].set_ylabel('Number of Ratings', fontsize=12)
        axes[0].set_yscale('log')
        axes[0].grid(alpha=0.3)
        
        # Cumulative percentage
        cumulative_pct = np.cumsum(movie_popularity.values) / movie_popularity.sum() * 100
        axes[1].plot(range(len(cumulative_pct)), cumulative_pct, 
                    color='darkred', linewidth=2)
        axes[1].axhline(y=80, color='green', linestyle='--', 
                       label='80% of ratings')
        axes[1].set_title('Cumulative Percentage of Ratings', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Movie Rank', fontsize=12)
        axes[1].set_ylabel('Cumulative % of Ratings', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}long_tail_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Long-tail analysis saved")
    
    def plot_temporal_trends(self):
        """Plot rating trends over time"""
        self.ratings['year_month'] = self.ratings['datetime'].dt.to_period('M')
        
        temporal_stats = self.ratings.groupby('year_month').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        temporal_stats.columns = ['year_month', 'avg_rating', 'rating_count']
        temporal_stats['year_month'] = temporal_stats['year_month'].astype(str)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Average rating over time
        axes[0].plot(range(len(temporal_stats)), temporal_stats['avg_rating'], 
                    color='purple', linewidth=2, marker='o', markersize=3)
        axes[0].set_title('Average Rating Trends Over Time', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Average Rating', fontsize=12)
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks([])
        
        # Rating count over time
        axes[1].bar(range(len(temporal_stats)), temporal_stats['rating_count'], 
                   color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_title('Rating Activity Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time Period', fontsize=12)
        axes[1].set_ylabel('Number of Ratings', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticks([])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Temporal trends saved")
    
    def generate_all_visualizations(self):
        """Generate all EDA visualizations"""
        print("\n" + "="*60)
        print("GENERATING EDA VISUALIZATIONS")
        print("="*60)
        
        self.plot_rating_distribution()
        self.plot_user_activity()
        self.plot_genre_analysis()
        self.plot_long_tail()
        self.plot_temporal_trends()
        
        print("\n✓ All visualizations generated successfully!")
        print(f"  Saved to: {self.output_dir}")


# ============================================================================
# PART 3: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Engineer features for content-based recommendations"""
    
    def __init__(self, movies, tags):
        self.movies = movies
        self.tags = tags
        self.genre_features = None
        self.tag_features = None
        self.combined_features = None
    
    def create_genre_features(self):
        """Create binary genre vectors"""
        print("\nCreating genre features...")
        
        # One-hot encode genres
        self.genre_features = self.movies['genres'].str.get_dummies('|')
        self.genre_features.index = self.movies['movieId']
        
        print(f"✓ Genre features: {self.genre_features.shape}")
        return self.genre_features
    
    def create_tag_features(self, max_features=100):
        """Create TF-IDF tag features"""
        print("\nCreating tag features...")
        
        # Aggregate tags per movie
        movie_tags = self.tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(str(val) for val in x)
        ).reset_index()
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        if len(movie_tags) > 0:
            tag_matrix = tfidf.fit_transform(movie_tags['tag'])
            
            self.tag_features = pd.DataFrame(
                tag_matrix.toarray(),
                index=movie_tags['movieId'],
                columns=[f'tag_{i}' for i in range(tag_matrix.shape[1])]
            )
        else:
            print("Warning: No tags available for feature engineering")
            self.tag_features = pd.DataFrame()
        
        print(f"✓ Tag features: {self.tag_features.shape}")
        return self.tag_features
    
    def combine_features(self):
        """Combine genre and tag features"""
        print("\nCombining features...")
        
        if self.tag_features is None or self.tag_features.empty:
            self.combined_features = self.genre_features
            return self.combined_features

        # Align indices
        common_movies = self.genre_features.index.intersection(self.tag_features.index)
        
        genre_aligned = self.genre_features.loc[common_movies]
        tag_aligned = self.tag_features.loc[common_movies]
        
        # Combine
        self.combined_features = pd.concat([genre_aligned, tag_aligned], axis=1)
        
        # Add movies that only have genre features
        genre_only = self.genre_features.index.difference(common_movies)
        if len(genre_only) > 0:
            genre_only_features = self.genre_features.loc[genre_only]
            # Pad with zeros for tag features
            tag_cols = [col for col in self.combined_features.columns if col.startswith('tag_')]
            for col in tag_cols:
                genre_only_features[col] = 0
            
            self.combined_features = pd.concat([self.combined_features, genre_only_features])
        
        print(f"✓ Combined features: {self.combined_features.shape}")
        return self.combined_features


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_user_item_matrix(ratings_df):
    """Create user-item interaction matrix"""
    user_item = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    return user_item


def get_movie_popularity(ratings_df):
    """Calculate movie popularity"""
    popularity = ratings_df.groupby('movieId').size().to_dict()
    return popularity


if __name__ == "__main__":
    print("ReelSense Part 1: Data Processing Module")
    # Quick test if run directly
    try:
        if os.path.exists('movies.csv'):
            loader = DataLoader()
            loader.load_data()
            print("Test successful")
    except Exception as e:
        print(f"Test skipped or failed: {e}")
