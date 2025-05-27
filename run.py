#!/usr/bin/env python
"""
Run Call Center Analysis
========================

This script runs the call center analysis pipeline using your CSV files.

Usage:
    python run_analysis.py

Make sure you have:
    - test.py (the main analysis code)
    - contact.csv
    - genesys.csv
    
All in the same directory.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Import the pipeline from test.py
try:
    from test import CallCenterAnalysisPipeline
except ImportError:
    print("Error: Could not import from test.py. Make sure test.py is in the same directory.")
    sys.exit(1)

def main():
    """
    Main function to run the analysis
    """
    print("\n" + "="*80)
    print("RUNNING CALL CENTER ANALYSIS")
    print("="*80 + "\n")
    
    # Define file paths
    genesys_file = 'genesys.csv'
    contact_file = 'contact.csv'
    output_directory = 'call_center_analysis_results'
    country = 'US'  # Change to 'UK' if needed
    
    # Check if files exist
    print("Checking for required files...")
    
    if not os.path.exists(genesys_file):
        print(f"❌ Error: Cannot find {genesys_file}")
        print("Please make sure genesys.csv is in the current directory")
        return
    else:
        print(f"✓ Found {genesys_file}")
    
    if not os.path.exists(contact_file):
        print(f"❌ Error: Cannot find {contact_file}")
        print("Please make sure contact.csv is in the current directory")
        return
    else:
        print(f"✓ Found {contact_file}")
    
    try:
        # Load the CSV files
        print("\nLoading data files...")
        
        # Try different encodings if needed
        try:
            genesys_df = pd.read_csv(genesys_file)
            print(f"✓ Loaded Genesys data: {len(genesys_df)} rows, {len(genesys_df.columns)} columns")
        except UnicodeDecodeError:
            print("Trying different encoding for Genesys file...")
            genesys_df = pd.read_csv(genesys_file, encoding='latin-1')
            print(f"✓ Loaded Genesys data with latin-1 encoding: {len(genesys_df)} rows")
        
        try:
            contact_df = pd.read_csv(contact_file)
            print(f"✓ Loaded Contact data: {len(contact_df)} rows, {len(contact_df.columns)} columns")
        except UnicodeDecodeError:
            print("Trying different encoding for Contact file...")
            contact_df = pd.read_csv(contact_file, encoding='latin-1')
            print(f"✓ Loaded Contact data with latin-1 encoding: {len(contact_df)} rows")
        
        # Show column names to verify correct loading
        print("\nGenesys columns:", list(genesys_df.columns)[:5], "...")
        print("Contact columns:", list(contact_df.columns)[:5], "...")
        
        # Initialize the pipeline
        print(f"\nInitializing analysis pipeline...")
        print(f"Output directory: {output_directory}")
        print(f"Country setting: {country}")
        
        pipeline = CallCenterAnalysisPipeline(
            country=country,
            output_dir=output_directory
        )
        
        # Load data into pipeline
        pipeline.load_data(genesys_df=genesys_df, contact_df=contact_df)
        
        # Run the complete pipeline
        print("\nStarting analysis pipeline...")
        print("This may take a few minutes depending on data size...\n")
        
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\nResults saved to: {output_directory}/")
            print("\nKey files created:")
            print(f"  📊 {output_directory}/merged_call_data.csv - Complete dataset with all features")
            print(f"  📈 {output_directory}/daily_summary.csv - Daily aggregated data")
            print(f"  🎯 {output_directory}/modeling_ready_data.csv - Ready for predictive modeling")
            print(f"  📄 {output_directory}/business_report.txt - Executive summary")
            print(f"  🖼️  {output_directory}/plots/ - All visualizations")
            print(f"  📝 {output_directory}/README.md - Documentation")
            
            # Show some quick stats
            print("\nQuick Statistics:")
            print(f"  - Total calls analyzed: {len(pipeline.merged_df):,}")
            print(f"  - Date range: {pipeline.merged_df['date'].min()} to {pipeline.merged_df['date'].max()}")
            print(f"  - Call centers: {pipeline.merged_df['call_centre'].nunique()}")
            
        else:
            print("\n❌ Analysis failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure both CSV files are in the current directory")
        print("2. Check that the CSV files are not open in Excel")
        print("3. Verify the CSV files have the expected columns")
        print("4. Try opening the CSV files in a text editor to check the format")
        
        # Print more detailed error info
        import traceback
        print("\nDetailed error:")
        traceback.print_exc()
        
    print("\n" + "="*80)
    print("Process complete")
    print("="*80)


if __name__ == "__main__":
    main()
