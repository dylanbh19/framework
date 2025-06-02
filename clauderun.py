"""
run_explainability_analysis.py
==============================
Simple runner script that automatically finds your data files and runs the explainability analysis.
No command-line parameters needed!

Just place this script in the same directory as your intent_explainability_data_only.py file
and run it with: python run_explainability_analysis.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Import the main analyzer
try:
    from intent_explainability_data_only import DataOnlyExplainabilityAnalyzer
except ImportError:
    print("ERROR: Could not find intent_explainability_data_only.py in the current directory!")
    print("Please ensure both files are in the same directory.")
    sys.exit(1)


def find_data_files():
    """
    Automatically find the augmented data files in common locations.
    Returns the best candidate file path.
    """
    print("🔍 Searching for augmented data files...")
    
    # Define search patterns and locations
    search_patterns = [
        # Most likely locations based on your script
        "augmentation_results_pro/best_augmented_data.csv",
        "augmentation_results/best_augmented_data.csv",
        "best_augmented_data.csv",
        
        # Alternative patterns
        "**/best_augmented_data.csv",
        "**/augmentation_results*/best_augmented_data.csv",
        
        # If the file has been renamed
        "augmented_data.csv",
        "*_augmented_data.csv",
        "**/augmented_data.csv",
        "**/*_augmented_data.csv",
        
        # Look for any CSV with 'intent_augmented' in recent results
        "augmentation_results*/*.csv",
        "*.csv"
    ]
    
    # Search in current directory and subdirectories
    current_dir = Path.cwd()
    found_files = []
    
    for pattern in search_patterns:
        if "**" in pattern:
            # Recursive search
            matches = list(current_dir.glob(pattern))
        else:
            # Direct path or simple pattern
            matches = list(current_dir.glob(pattern))
            if not matches and "/" in pattern:
                # Try as direct path
                direct_path = current_dir / pattern
                if direct_path.exists():
                    matches = [direct_path]
        
        for match in matches:
            if match.is_file() and match.suffix == '.csv':
                # Quick check if it's likely an augmented data file
                try:
                    with open(match, 'r', encoding='utf-8') as f:
                        header = f.readline().lower()
                        if 'intent_augmented' in header or 'intent' in header:
                            found_files.append(match)
                            print(f"✓ Found potential file: {match}")
                except:
                    pass
    
    # Remove duplicates and sort by modification time (newest first)
    found_files = list(set(found_files))
    found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not found_files:
        return None
    
    # Return the most recent file that contains 'best' or is in an augmentation results folder
    for file in found_files:
        if 'best' in file.name or 'augmentation_results' in str(file):
            return file
    
    # Otherwise return the most recent file
    return found_files[0]


def find_comparison_file(base_path):
    """
    Find the method comparison file if it exists.
    """
    if base_path is None:
        return None
        
    # Look for comparison file in the same directory as the main file
    parent_dir = base_path.parent
    
    comparison_patterns = [
        "method_comparison.csv",
        "comparison.csv",
        "*comparison*.csv"
    ]
    
    for pattern in comparison_patterns:
        matches = list(parent_dir.glob(pattern))
        if matches:
            print(f"✓ Found comparison file: {matches[0]}")
            return matches[0]
    
    return None


def main():
    """
    Main function to run the explainability analysis.
    """
    print("=" * 60)
    print("INTENT AUGMENTATION EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Find the augmented data file
    data_file = find_data_files()
    
    if data_file is None:
        print("❌ ERROR: Could not find augmented data file!")
        print("\nPlease ensure you have one of the following:")
        print("  - augmentation_results_pro/best_augmented_data.csv")
        print("  - augmentation_results/best_augmented_data.csv")
        print("  - Any CSV file with 'intent_augmented' column")
        print("\nSearched in:", Path.cwd())
        sys.exit(1)
    
    print(f"\n📊 Using data file: {data_file}")
    print(f"   File size: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Last modified: {datetime.fromtimestamp(data_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 2: Check for comparison file
    comparison_file = find_comparison_file(data_file)
    if comparison_file:
        print(f"\n📈 Also found comparison file: {comparison_file}")
    else:
        print("\n📝 No comparison file found (optional - analysis will continue)")
    
    # Step 3: Set output directory
    output_dir = Path("explainability_results")
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    
    # Step 4: Create analyzer and run
    print("\n" + "=" * 60)
    print("Starting analysis...")
    print("=" * 60 + "\n")
    
    try:
        # Initialize analyzer
        analyzer = DataOnlyExplainabilityAnalyzer(
            data_path=str(data_file),
            output_dir=str(output_dir)
        )
        
        # Run the analysis
        analyzer.run_full_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\nResults saved to: {output_dir.absolute()}")
        print("\n📋 Key outputs:")
        print(f"  - Executive report: {output_dir}/reports/explainability_report.html")
        print(f"  - Technical docs: {output_dir}/reports/explainability_report.md")
        print(f"  - Discovered rules: {output_dir}/reports/discovered_rules.txt")
        print(f"  - Case explanations: {output_dir}/data/case_explanations.csv")
        print(f"  - Interactive dashboards: {output_dir}/interactive/")
        print(f"  - Visualizations: {output_dir}/plots/")
        
        # Open the HTML report automatically if possible
        html_report = output_dir / "reports" / "explainability_report.html"
        if html_report.exists():
            print(f"\n🌐 Opening report in browser...")
            try:
                import webbrowser
                webbrowser.open(f"file://{html_report.absolute()}")
                print("   Report opened in default browser")
            except:
                print("   Could not open browser automatically")
                print(f"   Please open manually: {html_report.absolute()}")
        
    except Exception as e:
        print("\n❌ ERROR during analysis:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nPlease check:")
        print("  1. The data file has the expected columns (intent_augmented, etc.)")
        print("  2. You have the required packages installed:")
        print("     pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn")
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n✨ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("intent_explainability_data_only.py").exists():
        print("❌ ERROR: This script must be run from the same directory as intent_explainability_data_only.py")
        print(f"Current directory: {Path.cwd()}")
        print("\nPlease:")
        print("1. Save both files in the same directory")
        print("2. Navigate to that directory")
        print("3. Run: python run_explainability_analysis.py")
        sys.exit(1)
    
    # Run the analysis
    main()