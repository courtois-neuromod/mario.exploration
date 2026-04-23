import sys
import os
import pandas as pd
# Add the current directory to sys.path so we can import 'code' as a package
sys.path.append(os.getcwd())

# Define parameters
data_path = os.path.join(os.getcwd(), 'data')
metrics = ['dice', 'percent', 'kappa']
# Panel sizes to test
panel_sizes = [1, 3, 5, 9, 13, 15]

print(f"Running panel size sweep analysis on {data_path}...")
print(f"Testing panel sizes: {panel_sizes}")

try:
    # Try importing as package first
    try:
        from code.my_analysis.data_loader import load_and_aggregate
        from code.my_analysis.quality import check_quality
        from code.my_analysis.metrics import calculate_metrics_panel_sweep
        from code.my_analysis.viz import (plot_panel_size_sweep, 
                                          plot_scene_agreement_summary, 
                                          plot_global_agreement_summary, 
                                          plot_correlation_matrix, 
                                          plot_mean_correlation_matrix)
    except ImportError:
        # If that fails, add code to path and import
        print("Adjusting import paths...")
        sys.path.append(os.path.join(os.getcwd(), 'code'))
        from my_analysis.data_loader import load_and_aggregate
        from my_analysis.quality import check_quality
        from my_analysis.metrics import calculate_metrics_panel_sweep
        from my_analysis.viz import (plot_panel_size_sweep, 
                                     plot_scene_agreement_summary, 
                                     plot_global_agreement_summary, 
                                     plot_correlation_matrix, 
                                     plot_mean_correlation_matrix)
    
    output_dir = os.path.join(os.getcwd(), 'outputdata')
    
    # Load and clean data
    print("Loading data...")
    df = load_and_aggregate(data_path, save_data=True)
    
    #df = pd.read_csv('data/preload_data.csv')
    print("Checking data quality...")
    df_clean = check_quality(df, output_dir)
    
    # Calculate metrics across panel sizes
    print("Calculating metrics across panel sizes...")
    results = calculate_metrics_panel_sweep(df_clean, metrics, panel_sizes)
    
    # Generate plots
    print("Generating plots...")
    plots_output_path = os.path.join(output_dir, 'results')
    plots = plot_panel_size_sweep(results, output_path=plots_output_path)
    
    # Generate scene agreement summary plots
    print("Generating scene agreement summary plots...")
    plot_scene_agreement_summary(results, output_path=plots_output_path)
    
    # Generate global agreement summary plot
    print("Generating global agreement summary plot...")
    plot_global_agreement_summary(results, output_path=plots_output_path)
    
    # Generate correlation plots at scene level
    print("Generating correlation plots...")
    plot_correlation_matrix(df_clean, output_path=plots_output_path)
    
    # Generate mean correlation plot (Resume)
    print("Generating mean correlation plot...")
    plot_mean_correlation_matrix(df_clean, output_path=plots_output_path)
    
    print("Panel size sweep analysis finished successfully.")
except Exception as e:
    print(f"Analysis failed: {e}")
    import traceback
    traceback.print_exc()
