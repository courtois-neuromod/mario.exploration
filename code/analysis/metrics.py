import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from .utils import calculate_dice

def calculate_metrics(df, metrics, nb_panels=1, size_panel=None):
    """
    Backward-compatible wrapper for single configuration metrics.
    Translates to new unified sweep structure but returns simplified format.
    
    Args:
        df (pd.DataFrame): Filtered DataFrame containing judge answers.
        metrics (list): List of metrics to compute (['dice', 'percent', 'kappa']).
        nb_panels (int): Number of panels to form (effectively iterations if size_panel handled by sweep logic).
                         Wait, nb_panels in original was "number of panels to form for inter-panel agreement".
                         Actually in original code:
                         `n_judges < 2 * current_size_panel` check suggests we need 2 panels.
                         The loop `for i in range(100)` suggests bootstrapping 100 times.
                         'nb_panels' argument was seemingly unused for loop count, it was used for default size logic:
                         `current_size_panel = max(1, n_judges // nb_panels)` 
        size_panel (int): Number of judges per panel.

    Returns:
        dict: Standard simplified structure: scene -> player -> metric -> question -> mean_score
    """
    # Create a list for panel_sizes containing just the one requested size (or default logic)
    # But size_panel might be None, handled inside.
    # We'll use the new unified function but need to handle the size_panel logic first.
    
    # We cannot easily use the new sweep function directly because it iterates 1000 times and takes specific sizes.
    # The original calculate_metrics had unique logic for default size: `n_judges // nb_panels`.
    # This depends on n_judges which varies per group.
    # So we should probably keep the original logic or make the new function flexible enough.
    
    # Let's use a unified internal worker.
    return _calculate_metrics_unified(df, metrics, panel_sizes=None, 
                                      fixed_panel_size=size_panel, 
                                      nb_panels_heuristic=nb_panels,
                                      n_bootstraps=100, 
                                      return_sweep_format=False)

def calculate_metrics_panel_sweep(df, metrics, panel_sizes=None):
    """
    Calculates agreement metrics across multiple panel sizes.
    Wraps unified worker for sweep format.
    
    Args:
        df (pd.DataFrame): Data.
        metrics (list): Metrics.
        panel_sizes (list): Sizes to sweep.
        
    Returns:
        dict: Sweep structure: scene -> player -> metric -> question -> {size: score}
    """
    if panel_sizes is None:
        panel_sizes = [1, 2, 4, 6, 10, 15]
        
    return _calculate_metrics_unified(df, metrics, panel_sizes=panel_sizes, 
                                      n_bootstraps=1000, 
                                      return_sweep_format=True)

def _calculate_metrics_unified(df, metrics, panel_sizes=None, fixed_panel_size=None, 
                               nb_panels_heuristic=1, n_bootstraps=100, return_sweep_format=True):
    """
    Internal unified function for calculating metrics.
    
    Args:
        panel_sizes (list): List of sizes to test. If set, ignores fixed_panel_size logic.
        fixed_panel_size (int): Specific size to use if panel_sizes is None.
        nb_panels_heuristic (int): Used to calculate default size if both above are None.
        return_sweep_format (bool): If True, returns dict[size] = score. 
                                    If False, returns score (for backward compat).
    """
    results_data = {}
    
    # Group by Scene and Player
    grouped = df.groupby(['scene_id', 'player'])
    
    for (scene, player), group_df in grouped:
        if scene not in results_data:
            results_data[scene] = {}
        if player not in results_data[scene]:
            results_data[scene][player] = {}
            
        print(f"Processing Scene: {scene}, Player: {player}")
        
        available_judges = group_df['participant'].unique()
        n_judges = len(available_judges)
        
        # Determine sizes to test for this group
        current_panel_sizes = []
        if panel_sizes is not None:
            current_panel_sizes = panel_sizes
        else:
            # Legacy logic
            size = fixed_panel_size
            if size is None:
                size = max(1, n_judges // nb_panels_heuristic)
            current_panel_sizes = [size]
            
        # Initialize storage
        for m in metrics:
            results_data[scene][player][m] = {}
            
        questions = group_df['question'].unique()
        
        # Structure initialization
        for q in questions:
            for m in metrics:
                if return_sweep_format:
                    results_data[scene][player][m][q] = {}
                else:
                    # Will store just the value later
                    pass
        
        # Loop over sizes
        for size_panel in current_panel_sizes:
            if n_judges < 2 * size_panel:
                if panel_sizes is not None: # Only warn if in sweep mode where we expect many sizes
                    print(f"  Skipping panel size {size_panel}: not enough judges ({n_judges})")
                else:
                    print(f"Warning: Not enough judges ({n_judges}) to form 2 panels of size {size_panel}.")
                continue
                
            if panel_sizes is not None:
                print(f"  Testing panel size: {size_panel}")
            
            # Run Bootstrapping
            metric_scores = {m: {q: [] for q in questions} for m in metrics}
            
            for i in range(n_bootstraps):
                shuffled = np.random.permutation(available_judges)
                p1_judges = shuffled[:size_panel]
                p2_judges = shuffled[size_panel:2*size_panel]
                
                p1_df = group_df[group_df['participant'].isin(p1_judges)]
                p2_df = group_df[group_df['participant'].isin(p2_judges)]
                
                # Consensus
                p1 = p1_df.groupby(['question', 'clip'])['answer'].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )
                p2 = p2_df.groupby(['question', 'clip'])['answer'].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )
                
                for q in questions:
                    try:
                        v1_s = p1.loc[q] if q in p1.index else None
                        v2_s = p2.loc[q] if q in p2.index else None
                    except KeyError:
                        continue
                        
                    if v1_s is None or v2_s is None: continue
                    
                    # Align
                    common = v1_s.index.intersection(v2_s.index)
                    if len(common) == 0: continue
                    
                    v1 = v1_s.loc[common].astype(int)
                    v2 = v2_s.loc[common].astype(int)
                    
                    if 'dice' in metrics:
                        metric_scores['dice'][q].append(calculate_dice(v1, v2))
                    if 'percent' in metrics:
                        metric_scores['percent'][q].append((v1 == v2).mean())
                    if 'kappa' in metrics:
                        try:
                            k = cohen_kappa_score(v1, v2)
                            if not np.isnan(k): metric_scores['kappa'][q].append(k)
                        except: pass

            # Aggregate
            for m in metrics:
                for q in questions:
                    scores = metric_scores[m][q]
                    val = np.mean(scores) if scores else np.nan
                    
                    if return_sweep_format:
                        results_data[scene][player][m][q][size_panel] = val
                    else:
                        # For legacy format, we assume only one size loop
                        results_data[scene][player][m][q] = val
                        
    return results_data
