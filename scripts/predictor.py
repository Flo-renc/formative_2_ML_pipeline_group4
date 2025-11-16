import pandas as pd
import numpy as np
import pickle
import os

class Predictor:
    """
    Class to handle predictions on new data
    
    Loads trained models and makes predictions with confidence scores
    for customer product category preferences.
    """
    
    def __init__(self, model_file, scaler_file=None):
        """
        Initialize Predictor
        
        Parameters:
        -----------
        model_file : str
            Path to trained model pickle file
        scaler_file : str, optional
            Path to feature scaler pickle file
        """
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.model = None
        self.scaler = None
        self.model_type = None
        
        # Feature columns (must match training)
        self.feature_columns = [
            'avg_engagement_score', 'max_engagement_score', 'avg_purchase_interest',
            'total_social_platforms', 'num_positive_sentiment', 'num_neutral_sentiment',
            'num_negative_sentiment', 'sentiment_score', 'on_facebook', 'on_instagram',
            'on_twitter', 'on_linkedin', 'on_tiktok', 'high_engagement_flag',
            'high_purchase_intent_flag', 'total_transactions', 'total_spend',
            'avg_transaction_value', 'avg_customer_rating', 'days_since_last_purchase',
            'num_electronics', 'num_clothing', 'num_sports', 'num_groceries',
            'num_books', 'engagement_purchase_ratio'
        ]
    
    def load_model(self):
        """Load trained model and scaler from disk"""
        # Load model
        try:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = type(self.model).__name__
            print(f"   ✓ Model loaded: {self.model_type}")
        except FileNotFoundError:
            print(f"   ✗ Error: Model file not found: {self.model_file}")
            raise
        except Exception as e:
            print(f"   ✗ Error loading model: {str(e)}")
            raise
        
        # Load scaler if provided
        if self.scaler_file and os.path.exists(self.scaler_file):
            try:
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"   ✓ Scaler loaded")
            except Exception as e:
                print(f"   ⚠ Warning: Could not load scaler: {str(e)}")
        else:
            # Try to find scaler in same directory as model
            model_dir = os.path.dirname(self.model_file)
            scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
            
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"   ✓ Scaler loaded from: {scaler_path}")
                except Exception as e:
                    print(f"   ⚠ Warning: Could not load scaler: {str(e)}")
            else:
                print(f"   ℹ No scaler found (not required for tree-based models)")
    
    def predict_single(self, customer_data):
        """
        Predict product category for a single customer
        
        Parameters:
        -----------
        customer_data : dict
            Dictionary containing customer features
        
        Returns:
        --------
        dict : Prediction results with category and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Check for missing features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            # Fill missing features with zeros (default values)
            for feature in missing_features:
                df[feature] = 0
        
        # Extract features in correct order
        X = df[self.feature_columns]
        
        # Make prediction
        if self.model_type == 'LogisticRegression' and self.scaler:
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
        else:
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
        
        # Format results
        classes = self.model.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        
        return {
            'predicted_category': str(prediction),
            'confidence': float(max(probabilities)),
            'all_probabilities': prob_dict
        }
    
    def predict_from_file(self, input_file):
        """
        Make predictions on data from CSV file
        
        Parameters:
        -----------
        input_file : str
            Path to CSV file containing customer data
        
        Returns:
        --------
        DataFrame : Original data with predictions added
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load data
        try:
            data = pd.read_csv(input_file)
            print(f"   Loaded {len(data)} customers from {input_file}")
        except FileNotFoundError:
            print(f"   ✗ Error: Input file not found: {input_file}")
            raise
        except Exception as e:
            print(f"   ✗ Error loading file: {str(e)}")
            raise
        
        # Check for required features
        missing_features = set(self.feature_columns) - set(data.columns)
        if missing_features:
            print(f"   ⚠ Warning: Missing features: {missing_features}")
            print(f"   Filling missing features with zeros...")
            for feature in missing_features:
                data[feature] = 0
        
        # Extract features in correct order
        X = data[self.feature_columns]
        
        # Make predictions
        try:
            if self.model_type == 'LogisticRegression' and self.scaler:
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled)
                probabilities = self.model.predict_proba(X_scaled)
            else:
                predictions = self.model.predict(X)
                probabilities = self.model.predict_proba(X)
        except Exception as e:
            print(f"   ✗ Error making predictions: {str(e)}")
            raise
        
        # Add predictions to dataframe
        data['predicted_category'] = predictions
        data['prediction_confidence'] = [max(probs) for probs in probabilities]
        
        # Add individual class probabilities
        for i, cls in enumerate(self.model.classes_):
            data[f'prob_{cls}'] = probabilities[:, i]
        
        print(f"   ✓ Predictions complete for {len(data)} customers")
        
        return data
    
    def save_predictions(self, predictions_df, output_file):
        """
        Save predictions to CSV file
        
        Parameters:
        -----------
        predictions_df : DataFrame
            DataFrame containing predictions
        output_file : str
            Path to output CSV file
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            predictions_df.to_csv(output_file, index=False)
            print(f"   ✓ Predictions saved to: {output_file}")
        except Exception as e:
            print(f"   ✗ Error saving predictions: {str(e)}")
            raise
    
    def display_sample_predictions(self, predictions_df, n=10):
        """
        Display sample predictions
        
        Parameters:
        -----------
        predictions_df : DataFrame
            DataFrame containing predictions
        n : int
            Number of samples to display (default: 10)
        """
        print("\n" + "="*70)
        print(f"SAMPLE PREDICTIONS (First {n})")
        print("="*70)
        
        # Select columns to display
        display_cols = ['customer_id', 'predicted_category', 'prediction_confidence']
        
        # Add some feature columns if available
        feature_samples = ['avg_engagement_score', 'avg_purchase_interest', 'total_spend']
        for col in feature_samples:
            if col in predictions_df.columns:
                display_cols.append(col)
        
        # Get sample
        sample = predictions_df[display_cols].head(n).copy()
        
        # Format confidence as percentage
        sample['confidence_%'] = (sample['prediction_confidence'] * 100).round(2)
        sample = sample.drop('prediction_confidence', axis=1)
        
        # Round numeric columns
        numeric_cols = sample.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'confidence_%':
                sample[col] = sample[col].round(2)
        
        print(sample.to_string(index=False))
    
    def display_statistics(self, predictions_df):
        """
        Display prediction statistics
        
        Parameters:
        -----------
        predictions_df : DataFrame
            DataFrame containing predictions
        """
        print("\n" + "="*70)
        print("PREDICTION STATISTICS")
        print("="*70)
        
        print(f"\nTotal Predictions: {len(predictions_df)}")
        
        # Category distribution
        print(f"\n Predicted Category Distribution:")
        category_dist = predictions_df['predicted_category'].value_counts()
        for category, count in category_dist.items():
            percentage = (count / len(predictions_df)) * 100
            bar = '█' * int(percentage / 2)
            print(f"   {category:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Confidence statistics
        print(f"\n Confidence Statistics:")
        confidence = predictions_df['prediction_confidence']
        print(f"   Mean:   {confidence.mean():.2%}")
        print(f"   Median: {confidence.median():.2%}")
        print(f"   Std:    {confidence.std():.2%}")
        print(f"   Min:    {confidence.min():.2%}")
        print(f"   Max:    {confidence.max():.2%}")
        
        # Confidence distribution
        print(f"\n Confidence Distribution:")
        confidence_bins = [0, 0.5, 0.7, 0.85, 1.0]
        confidence_labels = ['Low (0-50%)', 'Medium (50-70%)', 'High (70-85%)', 'Very High (85-100%)']
        
        predictions_df['confidence_bin'] = pd.cut(
            predictions_df['prediction_confidence'],
            bins=confidence_bins,
            labels=confidence_labels,
            include_lowest=True
        )
        
        conf_dist = predictions_df['confidence_bin'].value_counts().sort_index()
        for label, count in conf_dist.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"   {label:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # High confidence predictions
        high_conf = predictions_df[predictions_df['prediction_confidence'] >= 0.7]
        print(f"\n High Confidence Predictions (≥70%): {len(high_conf)} ({len(high_conf)/len(predictions_df)*100:.1f}%)")
        
        # Low confidence predictions (may need review)
        low_conf = predictions_df[predictions_df['prediction_confidence'] < 0.5]
        if len(low_conf) > 0:
            print(f" Low Confidence Predictions (<50%): {len(low_conf)} ({len(low_conf)/len(predictions_df)*100:.1f}%)")
    
    def predict_batch_with_details(self, customers_list):
        """
        Predict for multiple customers with detailed output
        
        Parameters:
        -----------
        customers_list : list of dict
            List of customer feature dictionaries
        
        Returns:
        --------
        list : List of prediction results for each customer
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for i, customer in enumerate(customers_list):
            customer_id = customer.get('customer_id', f'customer_{i}')
            
            try:
                prediction = self.predict_single(customer)
                prediction['customer_id'] = customer_id
                prediction['status'] = 'success'
                results.append(prediction)
            except Exception as e:
                results.append({
                    'customer_id': customer_id,
                    'status': 'error',
                    'error_message': str(e)
                })
        
        return results