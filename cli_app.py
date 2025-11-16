#!/usr/bin/env python3
"""
Product Category Prediction - Command Line Application
Main Pipeline Script

This script provides a command-line interface for:
1. Data merging and feature engineering
2. Model training and evaluation
3. Making predictions on new data

Usage:
    python cli_app.py --help
    python cli_app.py merge --social social_profiles.csv --transactions transactions.csv
    python cli_app.py train --data merged_customer_features.csv
    python cli_app.py predict --model best_model.pkl --input merged_customer_features.csv
"""

import argparse
import sys
import os
from datetime import datetime

# Import pipeline modules
from scripts.data_merger import DataMerger
from scripts.model_trainer import ModelTrainer
from scripts.predictor import Predictor

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║     PRODUCT CATEGORY PREDICTION - COMMAND LINE APP          ║
    ║                                                              ║
    ║     Predict customer product preferences from social        ║
    ║     media engagement and transaction history                ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def merge_command(args):
    """Execute data merging and feature engineering"""
    print("\n" + "="*70)
    print("STEP 1: DATA MERGING & FEATURE ENGINEERING")
    print("="*70)
    
    # Validate input files
    if not os.path.exists(args.social):
        print(f" Error: Social profiles file not found: {args.social}")
        sys.exit(1)
    
    if not os.path.exists(args.transactions):
        print(f" Error: Transactions file not found: {args.transactions}")
        sys.exit(1)
    
    # Initialize merger
    merger = DataMerger(
        social_file=args.social,
        transaction_file=args.transactions
    )
    
    # Load and merge data
    print(f"\n Loading data...")
    print(f"   Social profiles: {args.social}")
    print(f"   Transactions: {args.transactions}")
    
    merged_df = merger.merge_and_engineer()
    
    # Save merged dataset
    output_file = args.output or 'merged_customer_features.csv'
    merger.save(output_file)
    
    print(f"\n Merge complete!")
    print(f"   Output: {output_file}")
    print(f"   Total customers: {len(merged_df)}")
    print(f"   Total features: {len(merged_df.columns)}")
    
    # Display summary
    if args.summary:
        merger.display_summary()

def train_command(args):
    """Execute model training"""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    # Validate input file
    if not os.path.exists(args.data):
        print(f" Error: Merged dataset not found: {args.data}")
        sys.exit(1)
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_file=args.data,
        target_column=args.target
    )
    
    # Load data
    print(f"\n Loading merged dataset: {args.data}")
    trainer.load_data()
    
    # Prepare features
    print(f"\n Preparing features...")
    trainer.prepare_features()
    
    # Train models
    print(f"\n Training models...")
    models_to_train = args.models if args.models else ['all']
    
    if 'all' in models_to_train:
        models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression', 'decision_tree']
    
    results = trainer.train_models(models_to_train)
    
    # Display results
    print(f"\n Training Results:")
    trainer.display_results()
    
    # Save best model
    output_dir = args.output or 'models'
    trainer.save_models(output_dir)
    
    print(f"\n Training complete!")
    print(f"   Models saved to: {output_dir}/")
    print(f"   Best model: {trainer.best_model_name}")
    
    # Generate report
    if args.report:
        report_file = os.path.join(output_dir, 'training_report.txt')
        trainer.generate_report(report_file)
        print(f"   Report: {report_file}")
    
    # Generate visualizations
    if args.visualize:
        trainer.generate_visualizations(output_dir)
        print(f"   Visualizations saved to: {output_dir}/")

def predict_command(args):
    """Execute predictions on new data"""
    print("\n" + "="*70)
    print("STEP 3: MAKING PREDICTIONS")
    print("="*70)
    
    # Validate input files
    if not os.path.exists(args.model):
        print(f" Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f" Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = Predictor(
        model_file=args.model,
        scaler_file=args.scaler
    )
    
    # Load model
    print(f"\n Loading model: {args.model}")
    predictor.load_model()
    
    # Make predictions
    print(f"\n Making predictions on: {args.input}")
    predictions_df = predictor.predict_from_file(args.input)
    
    # Save predictions
    output_file = args.output or 'predictions.csv'
    predictor.save_predictions(predictions_df, output_file)
    
    print(f"\n Predictions complete!")
    print(f"   Output: {output_file}")
    print(f"   Total predictions: {len(predictions_df)}")
    
    # Display sample predictions
    if args.sample:
        print(f"\n Sample predictions:")
        predictor.display_sample_predictions(predictions_df, n=args.sample)
    
    # Display confidence distribution
    if args.stats:
        predictor.display_statistics(predictions_df)

def info_command(args):
    """Display system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    print(f"\n Python Version: {sys.version}")
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    print(f"\n Package Versions:")
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"    {package}: {version}")
        except ImportError:
            print(f"    {package}: Not installed")
    
    # Check for data files
    print(f"\n Data Files:")
    data_files = [
        'merged_customer_features.csv',
        'customer_social_profiles.csv',
        'customer_transactions.csv'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"   ✓ {file} ({size:.2f} KB)")
        else:
            print(f"   ✗ {file} (Not found)")
    
    # Check for model files
    print(f"\n Model Files:")
    model_files = [
        'best_product_classifier.pkl',
        'feature_scaler.pkl',
        'models/random_forest_model.pkl',
        'models/gradient_boosting_model.pkl'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"   ✓ {file} ({size:.2f} KB)")
        else:
            print(f"   ✗ {file} (Not found)")

def main():
    """Main entry point"""
    print_banner()
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Product Category Prediction - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge datasets
  python cli_app.py merge --social social_profiles.csv --transactions transactions.csv
  
  # Train models
  python cli_app.py train --data merged_customer_features.csv --visualize
  
  # Make predictions
  python cli_app.py predict --model models/best_product_classifier.pkl --input new_customers.csv
  
  # Full pipeline
  python cli_app.py merge --social social_profiles.csv --transactions transactions.csv
  python cli_app.py train --data merged_customer_features.csv --visualize
  python cli_app.py predict --model models/best_product_classifier.pkl --input test_customers.csv
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge datasets and engineer features')
    merge_parser.add_argument('--social', required=True, help='Social profiles CSV file')
    merge_parser.add_argument('--transactions', required=True, help='Transactions CSV file')
    merge_parser.add_argument('--output', help='Output file (default: merged_customer_features.csv)')
    merge_parser.add_argument('--summary', action='store_true', help='Display summary statistics')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train prediction models')
    train_parser.add_argument('--data', required=True, help='Merged dataset CSV file')
    train_parser.add_argument('--target', default='most_frequent_category', help='Target column name')
    train_parser.add_argument('--models', nargs='+', choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'decision_tree', 'all'], help='Models to train')
    train_parser.add_argument('--output', help='Output directory for models (default: models)')
    train_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    train_parser.add_argument('--report', action='store_true', help='Generate training report')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new data')
    predict_parser.add_argument('--model', required=True, help='Trained model file (.pkl)')
    predict_parser.add_argument('--scaler', help='Feature scaler file (.pkl)')
    predict_parser.add_argument('--input', required=True, help='Input CSV file with customer data')
    predict_parser.add_argument('--output', help='Output file for predictions (default: predictions.csv)')
    predict_parser.add_argument('--sample', type=int, help='Display N sample predictions')
    predict_parser.add_argument('--stats', action='store_true', help='Display prediction statistics')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'merge':
        merge_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'info':
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)
    
    print("\n" + "="*70)
    print(f" Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()