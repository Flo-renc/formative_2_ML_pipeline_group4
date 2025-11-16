

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Class to handle model training and evaluation
    
    Trains multiple classification models, evaluates performance,
    and selects the best model for deployment.
    """
    
    def __init__(self, data_file, target_column='most_frequent_category'):
        """
        Initialize ModelTrainer
        
        Parameters:
        -----------
        data_file : str
            Path to merged dataset CSV
        target_column : str
            Name of target column (default: 'most_frequent_category')
        """
        self.data_file = data_file
        self.target_column = target_column
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
        # Define feature columns for model training
        self.feature_columns = [
            # Social media features
            'avg_engagement_score', 'max_engagement_score', 'avg_purchase_interest',
            'total_social_platforms', 'num_positive_sentiment', 'num_neutral_sentiment',
            'num_negative_sentiment', 'sentiment_score', 'on_facebook', 'on_instagram',
            'on_twitter', 'on_linkedin', 'on_tiktok', 'high_engagement_flag',
            'high_purchase_intent_flag',
            # Transaction features
            'total_transactions', 'total_spend',
            'avg_transaction_value', 'avg_customer_rating', 'days_since_last_purchase',
            # Category history
            'num_electronics', 'num_clothing', 'num_sports', 'num_groceries',
            'num_books',
            # Engineered features
            'engagement_purchase_ratio'
        ]
    
    def load_data(self):
        """Load the merged dataset"""
        self.df = pd.read_csv(self.data_file)
        
        # Remove Unknown categories (insufficient data)
        initial_count = len(self.df)
        self.df = self.df[self.df[self.target_column] != 'Unknown']
        removed = initial_count - len(self.df)
        
        print(f"    Loaded {len(self.df)} customers")
        if removed > 0:
            print(f"    Removed {removed} customers with 'Unknown' category")
        print(f"    Features: {len(self.df.columns)}")
        print(f"    Target classes: {self.df[self.target_column].nunique()}")
    
    def prepare_features(self):
        """Prepare features for training"""
        # Extract features and target
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        # Train-test split with stratification to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Feature scaling (important for Logistic Regression)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   ✓ Train set: {len(self.X_train)} samples (75%)")
        print(f"   ✓ Test set: {len(self.X_test)} samples (25%)")
        print(f"   ✓ Features: {len(self.feature_columns)}")
        print(f"   ✓ Feature scaling applied")
    
    def train_models(self, model_list):
        """
        Train specified classification models
        
        Parameters:
        -----------
        model_list : list
            List of model names to train
            Options: 'random_forest', 'gradient_boosting', 
                     'logistic_regression', 'decision_tree'
        
        Returns:
        --------
        dict : Dictionary containing results for each model
        """
        # Define available models
        available_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                multi_class='multinomial'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=10
            )
        }
        
        # Train each model
        for model_name in model_list:
            if model_name not in available_models:
                print(f"   ⚠ Warning: Unknown model '{model_name}', skipping...")
                continue
            
            print(f"\n   Training {model_name.replace('_', ' ').title()}...")
            model = available_models[model_name]
            
            # Train model (use scaled data for Logistic Regression)
            if model_name == 'logistic_regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_train_pred = model.predict(self.X_train_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_train_pred = model.predict(self.X_train)
            
            # Calculate performance metrics
            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store model and results
            self.models[model_name] = model
            self.results[model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            print(f"      Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")
        
        # Select best model based on F1-score
        if self.results:
            self.best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
            self.best_model = self.models[self.best_model_name]
            print(f"\n   🏆 Best model: {self.best_model_name.replace('_', ' ').title()}")
            print(f"      F1-Score: {self.results[self.best_model_name]['f1']:.4f}")
        
        return self.results
    
    def display_results(self):
        """Display training results and comparison"""
        if not self.results:
            print("   No results to display")
            return
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': [name.replace('_', ' ').title() for name in self.results.keys()],
            'Train_Acc': [self.results[m]['train_accuracy'] for m in self.results],
            'Test_Acc': [self.results[m]['test_accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1_Score': [self.results[m]['f1'] for m in self.results]
        }).sort_values('F1_Score', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison.to_string(index=False))
        
        # Detailed classification report for best model
        print("\n" + "="*70)
        print(f"DETAILED REPORT - {self.best_model_name.replace('_', ' ').upper()}")
        print("="*70)
        print(classification_report(
            self.y_test, 
            self.results[self.best_model_name]['predictions']
        ))
    
    def save_models(self, output_dir='models'):
        """
        Save trained models to disk
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models (default: 'models')
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save all trained models
        for name, model in self.models.items():
            model_file = os.path.join(output_dir, f'{name}_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"    Saved: {model_file}")
        
        # Save best model separately for easy access
        if self.best_model:
            best_file = os.path.join(output_dir, 'best_product_classifier.pkl')
            with open(best_file, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"    Saved best model: {best_file}")
        
        # Save feature scaler
        if self.scaler:
            scaler_file = os.path.join(output_dir, 'feature_scaler.pkl')
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"    Saved scaler: {scaler_file}")
        
        # Save feature column names
        feature_file = os.path.join(output_dir, 'feature_columns.pkl')
        with open(feature_file, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"    Saved feature columns: {feature_file}")
        
        # Save comparison results as CSV
        if self.results:
            comparison = pd.DataFrame({
                'Model': list(self.results.keys()),
                'Train_Accuracy': [self.results[m]['train_accuracy'] for m in self.results],
                'Test_Accuracy': [self.results[m]['test_accuracy'] for m in self.results],
                'Precision': [self.results[m]['precision'] for m in self.results],
                'Recall': [self.results[m]['recall'] for m in self.results],
                'F1_Score': [self.results[m]['f1'] for m in self.results]
            })
            comparison_file = os.path.join(output_dir, 'model_comparison.csv')
            comparison.to_csv(comparison_file, index=False)
            print(f"    Saved comparison: {comparison_file}")
    
    def generate_visualizations(self, output_dir='models'):
        """
        Generate visualization plots
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        if not self.best_model or not self.results:
            print("   No model to visualize")
            return
        
        # Create results directory if it doesn't exist
        results_dir = output_dir.replace('models', 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.results[self.best_model_name]['predictions'])
        categories = sorted(self.y_test.unique())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pd.DataFrame(cm, index=categories, columns=categories),
            annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {self.best_model_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Actual Category', fontsize=12)
        plt.xlabel('Predicted Category', fontsize=12)
        plt.tight_layout()
        cm_file = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {cm_file}")
        
        # 2. Feature Importance (if applicable)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
            bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top 15 Feature Importance - {self.best_model_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            fi_file = os.path.join(results_dir, 'feature_importance.png')
            plt.savefig(fi_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {fi_file}")
            
            # Save feature importance as CSV
            fi_csv = os.path.join(results_dir, 'feature_importance.csv')
            feature_importance.to_csv(fi_csv, index=False)
            print(f"    Saved: {fi_csv}")
        
        # 3. Model Comparison Chart
        comparison = pd.DataFrame({
            'Model': [name.replace('_', ' ').title() for name in self.results.keys()],
            'Accuracy': [self.results[m]['test_accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1'] for m in self.results]
        })
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(comparison))
        width = 0.2
        
        plt.bar(x - 1.5*width, comparison['Accuracy'], width, label='Accuracy', alpha=0.8)
        plt.bar(x - 0.5*width, comparison['Precision'], width, label='Precision', alpha=0.8)
        plt.bar(x + 0.5*width, comparison['Recall'], width, label='Recall', alpha=0.8)
        plt.bar(x + 1.5*width, comparison['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, comparison['Model'], rotation=15, ha='right')
        plt.legend()
        plt.ylim([0, 1.05])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        comp_file = os.path.join(results_dir, 'model_comparison.png')
        plt.savefig(comp_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {comp_file}")
    
    def generate_report(self, output_file):
        """
        Generate text report of training results
        
        Parameters:
        -----------
        output_file : str
            Path to output report file
        """
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PRODUCT CATEGORY PREDICTION - TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {self.data_file}\n")
            f.write(f"Target: {self.target_column}\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Training Samples: {len(self.X_train)}\n")
            f.write(f"Test Samples: {len(self.X_test)}\n")
            f.write(f"Features: {len(self.feature_columns)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("MODEL RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for name in self.results:
                f.write(f"{name.replace('_', ' ').upper()}\n")
                f.write(f"  Train Accuracy: {self.results[name]['train_accuracy']:.4f}\n")
                f.write(f"  Test Accuracy: {self.results[name]['test_accuracy']:.4f}\n")
                f.write(f"  Precision: {self.results[name]['precision']:.4f}\n")
                f.write(f"  Recall: {self.results[name]['recall']:.4f}\n")
                f.write(f"  F1-Score: {self.results[name]['f1']:.4f}\n\n")
            
            f.write("="*70 + "\n")
            f.write(f"BEST MODEL: {self.best_model_name.replace('_', ' ').upper()}\n")
            f.write("="*70 + "\n\n")
            
            f.write(classification_report(
                self.y_test, 
                self.results[self.best_model_name]['predictions']
            ))
        
        print(f"   ✓ Report saved: {output_file}")