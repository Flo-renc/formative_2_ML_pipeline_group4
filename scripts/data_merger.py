import pandas as pd
import numpy as np
from datetime import datetime

class DataMerger:
    """
    Class to handle data merging and feature engineering
    
    This class merges customer social media profiles with transaction history
    and creates engineered features for machine learning models.
    """
    
    def __init__(self, social_file, transaction_file):
        """
        Initialize DataMerger
        
        Parameters:
        -----------
        social_file : str
            Path to social profiles CSV file
        transaction_file : str
            Path to transactions CSV file
        """
        self.social_file = social_file
        self.transaction_file = transaction_file
        self.merged_data = None
        self.social_data = None
        self.transaction_data = None
    
    def load_data(self):
        """Load CSV files and clean column names"""
        print("   Loading social profiles...")
        self.social_data = pd.read_csv(self.social_file)
        self.social_data.columns = self.social_data.columns.str.strip()
        
        print("   Loading transactions...")
        self.transaction_data = pd.read_csv(self.transaction_file)
        self.transaction_data.columns = self.transaction_data.columns.str.strip()
        
        print(f"   ✓ Social profiles: {self.social_data.shape}")
        print(f"   ✓ Transactions: {self.transaction_data.shape}")
    
    def create_customer_mapping(self):
        """Create customer ID mapping from legacy to new format"""
        print("\n   Creating customer ID mapping...")
        # Map customer_id_legacy (100-199) to customer_id_new (A100-A199)
        self.transaction_data['customer_id_new'] = 'A' + self.transaction_data['customer_id_legacy'].astype(str)
        print(f"   Mapped {len(self.transaction_data['customer_id_new'].unique())} unique customers")
    
    def aggregate_social_features(self):
        """Aggregate social media features by customer"""
        print("\n   Aggregating social media features...")
        
        # Aggregate by customer
        social_agg = self.social_data.groupby('customer_id_new').agg({
            'engagement_score': ['mean', 'max', 'count'],
            'purchase_interest_score': 'mean',
            'social_media_platform': lambda x: x.tolist(),
            'review_sentiment': lambda x: x.tolist()
        }).reset_index()
        
        # Flatten column names
        social_agg.columns = ['customer_id', 'avg_engagement_score', 'max_engagement_score', 
                              'total_social_platforms', 'avg_purchase_interest', 
                              'platforms', 'sentiments']
        
        # Create sentiment counts
        social_agg['num_positive_sentiment'] = social_agg['sentiments'].apply(
            lambda x: sum(1 for s in x if s == 'Positive')
        )
        social_agg['num_neutral_sentiment'] = social_agg['sentiments'].apply(
            lambda x: sum(1 for s in x if s == 'Neutral')
        )
        social_agg['num_negative_sentiment'] = social_agg['sentiments'].apply(
            lambda x: sum(1 for s in x if s == 'Negative')
        )
        
        # Dominant sentiment
        def get_dominant_sentiment(sentiments):
            if len(sentiments) == 0:
                return 'Unknown'
            sentiment_counts = pd.Series(sentiments).value_counts()
            return sentiment_counts.index[0]
        
        social_agg['dominant_sentiment'] = social_agg['sentiments'].apply(get_dominant_sentiment)
        
        # Calculate sentiment score (Positive=1, Neutral=0, Negative=-1)
        def calculate_sentiment_score(sentiments):
            if len(sentiments) == 0:
                return 0
            score = sum(1 if s == 'Positive' else -1 if s == 'Negative' else 0 for s in sentiments)
            return score / len(sentiments)
        
        social_agg['sentiment_score'] = social_agg['sentiments'].apply(calculate_sentiment_score)
        
        # Platform presence indicators (one-hot encoding)
        social_agg['on_facebook'] = social_agg['platforms'].apply(lambda x: 1 if 'Facebook' in x else 0)
        social_agg['on_instagram'] = social_agg['platforms'].apply(lambda x: 1 if 'Instagram' in x else 0)
        social_agg['on_twitter'] = social_agg['platforms'].apply(lambda x: 1 if 'Twitter' in x else 0)
        social_agg['on_linkedin'] = social_agg['platforms'].apply(lambda x: 1 if 'LinkedIn' in x else 0)
        social_agg['on_tiktok'] = social_agg['platforms'].apply(lambda x: 1 if 'TikTok' in x else 0)
        
        # Create flags for high engagement and purchase intent
        social_agg['high_engagement_flag'] = (social_agg['avg_engagement_score'] >= 80).astype(int)
        social_agg['high_purchase_intent_flag'] = (social_agg['avg_purchase_interest'] >= 4.0).astype(int)
        
        # Drop intermediate columns
        social_agg = social_agg.drop(['platforms', 'sentiments'], axis=1)
        
        print(f"    Aggregated {len(social_agg)} customer social profiles")
        return social_agg
    
    def aggregate_transaction_features(self):
        """Aggregate transaction features by customer"""
        print("\n   Aggregating transaction features...")
        
        # Convert customer_rating to numeric, handle missing values
        self.transaction_data['customer_rating'] = pd.to_numeric(
            self.transaction_data['customer_rating'], errors='coerce'
        )
        
        # Aggregate transactions by customer
        transaction_agg = self.transaction_data.groupby('customer_id_new').agg({
            'transaction_id': 'count',
            'purchase_amount': ['sum', 'mean'],
            'customer_rating': 'mean',
            'product_category': lambda x: x.tolist(),
            'purchase_date': 'max'
        }).reset_index()
        
        # Flatten column names
        transaction_agg.columns = ['customer_id', 'total_transactions', 'total_spend', 
                                   'avg_transaction_value', 'avg_customer_rating', 
                                   'categories', 'last_purchase_date']
        
        # Create category-specific purchase counts
        transaction_agg['num_electronics'] = transaction_agg['categories'].apply(
            lambda x: sum(1 for c in x if c == 'Electronics')
        )
        transaction_agg['num_clothing'] = transaction_agg['categories'].apply(
            lambda x: sum(1 for c in x if c == 'Clothing')
        )
        transaction_agg['num_sports'] = transaction_agg['categories'].apply(
            lambda x: sum(1 for c in x if c == 'Sports')
        )
        transaction_agg['num_groceries'] = transaction_agg['categories'].apply(
            lambda x: sum(1 for c in x if c == 'Groceries')
        )
        transaction_agg['num_books'] = transaction_agg['categories'].apply(
            lambda x: sum(1 for c in x if c == 'Books')
        )
        
        # Most frequent category (target variable)
        def get_most_frequent_category(categories):
            if len(categories) == 0:
                return 'Unknown'
            category_counts = pd.Series(categories).value_counts()
            return category_counts.index[0]
        
        transaction_agg['most_frequent_category'] = transaction_agg['categories'].apply(
            get_most_frequent_category
        )
        
        # Get last purchase category
        last_purchases = self.transaction_data.sort_values('purchase_date').groupby('customer_id_new').last()
        transaction_agg = transaction_agg.merge(
            last_purchases[['product_category']].rename(columns={'product_category': 'last_purchase_category'}),
            left_on='customer_id',
            right_index=True,
            how='left'
        )
        
        # Calculate days since last purchase
        reference_date = pd.to_datetime('2024-05-30')
        transaction_agg['last_purchase_date'] = pd.to_datetime(transaction_agg['last_purchase_date'])
        transaction_agg['days_since_last_purchase'] = (
            reference_date - transaction_agg['last_purchase_date']
        ).dt.days
        
        # Drop intermediate columns
        transaction_agg = transaction_agg.drop(['categories', 'last_purchase_date'], axis=1)
        
        print(f"    Aggregated {len(transaction_agg)} customer transactions")
        return transaction_agg
    
    def merge_datasets(self, social_agg, transaction_agg):
        """Merge social and transaction datasets"""
        print("\n   Merging datasets...")
        
        # Outer join to keep all customers
        merged = social_agg.merge(transaction_agg, on='customer_id', how='outer')
        
        # Fill missing numeric values with 0
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0)
        
        # Fill missing categorical values
        merged['most_frequent_category'] = merged['most_frequent_category'].fillna('Unknown')
        merged['last_purchase_category'] = merged['last_purchase_category'].fillna('Unknown')
        merged['dominant_sentiment'] = merged['dominant_sentiment'].fillna('Unknown')
        
        print(f"    Merged dataset: {merged.shape}")
        return merged
    
    def engineer_features(self, merged):
        """Engineer additional features"""
        print("\n   Engineering features...")
        
        # Engagement to purchase ratio
        # Shows relationship between social engagement and spending
        merged['engagement_purchase_ratio'] = np.where(
            merged['total_spend'] > 0,
            merged['avg_engagement_score'] / (merged['total_spend'] / 100),
            0
        )
        
        # Round numeric columns for cleaner output
        numeric_columns_to_round = [
            'avg_engagement_score', 'avg_purchase_interest', 
            'avg_transaction_value', 'avg_customer_rating',
            'sentiment_score', 'engagement_purchase_ratio'
        ]
        
        for col in numeric_columns_to_round:
            if col in merged.columns:
                merged[col] = merged[col].round(2)
        
        merged['total_spend'] = merged['total_spend'].round(2)
        
        print(f"    Feature engineering complete")
        print(f"    Total features created: {len(merged.columns)}")
        return merged
    
    def merge_and_engineer(self):
        """
        Execute full merge and feature engineering pipeline
        
        Returns:
        --------
        DataFrame : Merged dataset with engineered features
        """
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create customer mapping
        self.create_customer_mapping()
        
        # Step 3: Aggregate social features
        social_agg = self.aggregate_social_features()
        
        # Step 4: Aggregate transaction features
        transaction_agg = self.aggregate_transaction_features()
        
        # Step 5: Merge datasets
        merged = self.merge_datasets(social_agg, transaction_agg)
        
        # Step 6: Engineer features
        self.merged_data = self.engineer_features(merged)
        
        return self.merged_data
    
    def save(self, output_file):
        """
        Save merged dataset to CSV
        
        Parameters:
        -----------
        output_file : str
            Path to output CSV file
        """
        if self.merged_data is not None:
            self.merged_data.to_csv(output_file, index=False)
            print(f"\n    Saved to: {output_file}")
        else:
            print("    No merged data to save. Run merge_and_engineer() first.")
    
    def display_summary(self):
        """Display summary statistics of merged dataset"""
        if self.merged_data is None:
            print("    No data to summarize")
            return
        
        print("\n" + "="*70)
        print("MERGED DATASET SUMMARY")
        print("="*70)
        
        print(f"\nShape: {self.merged_data.shape}")
        print(f"Total Customers: {len(self.merged_data)}")
        print(f"Total Features: {len(self.merged_data.columns)}")
        
        print(f"\nFeature Categories:")
        print(f"  • Social Media Features: 15")
        print(f"  • Transaction Features: 5")
        print(f"  • Category History: 5")
        print(f"  • Engineered Features: 7")
        
        print(f"\nNumeric Features Summary:")
        numeric_summary = self.merged_data.describe()
        print(numeric_summary[['avg_engagement_score', 'avg_purchase_interest', 
                               'total_spend', 'total_transactions']].round(2))
        
        print(f"\nTarget Distribution (Most Frequent Category):")
        target_dist = self.merged_data['most_frequent_category'].value_counts()
        for category, count in target_dist.items():
            percentage = (count / len(self.merged_data)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print(f"\nPlatform Distribution:")
        platform_cols = ['on_facebook', 'on_instagram', 'on_twitter', 'on_linkedin', 'on_tiktok']
        for col in platform_cols:
            platform_name = col.replace('on_', '').title()
            count = self.merged_data[col].sum()
            print(f"  {platform_name}: {int(count)} customers")
        
        print(f"\nSentiment Distribution:")
        sentiment_dist = self.merged_data['dominant_sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count}")
        
        print(f"\nMissing Values:")
        missing = self.merged_data.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values ")
        else:
            print(missing[missing > 0])
        
        print("\n" + "="*70)