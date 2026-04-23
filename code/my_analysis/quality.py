import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path
import random
from sklearn.metrics import cohen_kappa_score
from .utils import calculate_dice
from .viz import plot_scene_quality

# TODO: 

def check_quality(df, output_dir):
    """
    Checks data quality and excludes participants based on criteria.
    Generates quality plots.
    
    Args:
        df (pd.DataFrame): Aggregated DataFrame with columns: 
                           ['judge_ID', 'scene_id', 'player', 'clip', 'visualisation_time', 'answer', ...]
        output_dir (str): Directory to save the exclusion files and plots.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    
    # Ensure output directory exists
    outliers_path = Path(os.path.join('outputdata', 'quality_check', 'outliers.txt'))
    outliers_path.parent.mkdir(parents=True, exist_ok=True)
    outliers_path.touch()

    excluded_judges = set()

    date_message = ['\n###########################\n',f'###   {datetime.datetime.now()}    ###\n', '###########################\n']
    outliers_message = []+date_message
    
    # --- 1. Identify Uncomplete Judges ---
    uncomplete_judges = df[~df['done']]['judge_ID'].unique()
    outliers_message = outliers_message + [f"{judge} had not FINISHED the study\n" for judge in uncomplete_judges]
    print("Uncomplete judges:", uncomplete_judges)
    excluded_judges.update(uncomplete_judges)
            
    df_analysis = df[df['done']].copy()
    
    # --- 2. Time Outliers (Visualisation Time < 80% Duration) ---
    df_analysis['clip_uid'] = df_analysis['scene_id'].astype(str) + "_" + df_analysis['player'].astype(str) + "_" + df_analysis['clip'].astype(str)
    
    time_outliers = {}
    
    for judge in df_analysis['judge_ID'].unique():
        judge_df = df_analysis[df_analysis['judge_ID'] == judge]
        
        # Group by clip to get max vis time for this judge/clip
        judge_clip_stats = judge_df.groupby('clip_uid').agg({
            'visualisation_time': 'max',
            'video_duration': 'first'
        })
        
        is_time_outlier = False
        outlier_points = []
        for clip_uid, row in judge_clip_stats.iterrows():
            max_vis = row['visualisation_time']
            duration = row['video_duration']
            
            if pd.notnull(duration) and duration > 0:
                if max_vis < 0.80 * duration:
                    is_time_outlier = True
                    outlier_points.append(clip_uid)

        if is_time_outlier:
            time_outliers[judge] = len(outlier_points)
            
    # --- 3. Agreement Metrics Outliers (< Mean - 1*SD) ---
    metric_outliers = set()
    judge_metrics = {j: {'kappa': []} for j in df_analysis['judge_ID'].unique()}
    
    for (scene, player), group in df_analysis.groupby(['scene_id', 'player']):
        # Pivot answers: Index=[question, clip], Columns=judge_ID

        pivoted = group.pivot_table(index=['question', 'clip'], columns='judge_ID', values='answer', aggfunc='first')
        pivoted_ref = w
        for judge in pivoted.columns:
            consensus = pivoted_ref[judge]
            judge_ans = pivoted[judge]
            valid_mask = judge_ans.notna() & consensus.notna()
            
            if not valid_mask.any(): continue
            
            v1 = judge_ans[valid_mask].astype(int)
            v2 = consensus[valid_mask].astype(int)
            
            try:
                k = cohen_kappa_score(v1, v2)
                if np.isnan(k): k = 1.0 
                judge_metrics[judge]['kappa'].append(k)
            except: pass

    # Average metrics per judge
    final_judge_metrics = []
    for judge, measurements in judge_metrics.items():
        row = {'judge_ID': judge}
        for m in ['kappa']:
            row[m] = np.mean(measurements[m]) if measurements[m] else np.nan
        final_judge_metrics.append(row)
        
    df_metrics = pd.DataFrame(final_judge_metrics).set_index('judge_ID')
    
    outliers = df_metrics[df_metrics['kappa'] < 1.5].index.tolist()
    metric_outliers.update(outliers)
        
    bad_boys = metric_outliers & set(time_outliers.keys())
    outliers_message = outliers_message+[f"{judge} saw less than 80% of video DURATION for {time_outliers[judge]} AND had extrem AGREEMENT measure: {df_metrics.loc[judge, :].to_list()} (percent, kappa, dice)\n" for judge in bad_boys]

    with open(outliers_path, 'a') as f:
        for line in outliers_message:
             f.write(line)

    # --- 4. Plotting (Per Scene) - Delegated to viz.py ---
    
    for (scene, player), group in df_analysis.groupby(['scene_id', 'player']):
        
        # --- A. Prepare Data for this Scene ---
        
        # 1. Time Data & Outliers
        time_data = []
        scene_time_outlier_judges = set()
        scene_time_outlier_events = set()

        
        for (judge, clip), subgroup in group.groupby(['judge_ID', 'clip']):
             max_vis = subgroup['visualisation_time'].max()
             duration = subgroup['video_duration'].iloc[0] 
             
             is_outlier_point = False
             if pd.notnull(duration) and duration > 0:
                 if max_vis < 0.75 * duration:
                     is_outlier_point = True
                     scene_time_outlier_judges.add(judge)
                     scene_time_outlier_events.add((judge, clip))
            
             time_data.append({
                 'judge_ID': judge,
                 'clip': clip,
                 'visualisation_time': max_vis,
                 'video_duration': duration,
                 'is_outlier': is_outlier_point
             })
             
        df_plot_time = pd.DataFrame(time_data)
        
        # 2. Metric Data & Outliers
        scene_metric_outlier_judges = set()
        scene_metric_outlier_events = set()
        
        pivoted = group.pivot_table(index=['question', 'clip'], columns='judge_ID', values='answer', aggfunc='first')
        judge_metrics_scene = {j: {'percent': [], 'kappa': [], 'dice': []} for j in group['judge_ID'].unique()}
        
        for judge in pivoted.columns:
            others = pivoted.drop(columns=[judge])
            if others.empty: continue
            consensus = others.mode(axis=1).iloc[:, 0]
            judge_ans = pivoted[judge]
            valid_mask = judge_ans.notna() & consensus.notna()
            
            if valid_mask.any():
                v1 = judge_ans[valid_mask].astype(int)
                v2 = consensus[valid_mask].astype(int)
                judge_metrics_scene[judge]['percent'].append((v1 == v2).mean())
                try:
                    k = cohen_kappa_score(v1, v2)
                    if np.isnan(k): k = 1.0
                    judge_metrics_scene[judge]['kappa'].append(k)
                except: pass
                judge_metrics_scene[judge]['dice'].append(calculate_dice(v1, v2))

        scene_final_metrics = []
        for judge, measurements in judge_metrics_scene.items():
            row = {'judge_ID': judge}
            for m in ['percent', 'kappa', 'dice']:
                row[m] = np.mean(measurements[m]) if measurements[m] else np.nan
            scene_final_metrics.append(row)
        
        df_metrics_scene_final = pd.DataFrame(scene_final_metrics)
        
        # Local Metric Outliers
        if not df_metrics_scene_final.empty:
            df_metrics_scene_final = df_metrics_scene_final.set_index('judge_ID')
            for metric in ['percent', 'kappa', 'dice']:
                mean_val = df_metrics_scene_final[metric].mean()
                std_val = df_metrics_scene_final[metric].std()
                threshold = mean_val - 3 * std_val
                
                current_outliers = df_metrics_scene_final[df_metrics_scene_final[metric] < threshold].index.tolist()
                for j in current_outliers:
                    scene_metric_outlier_judges.add(j)
                    scene_metric_outlier_events.add((j, metric))

        # Prepare outliers dict for viz
        outliers_for_viz = {
            'time': scene_time_outlier_judges,
            'metrics': scene_metric_outlier_events
        }
        
        # Call Visualization
        plot_scene_quality(scene, player, df_plot_time, df_metrics_scene_final, outliers_for_viz, outliers_path.parent)
        
    # --- 5. Return Filtered Data ---
    print(f"WARNING: {len(time_outliers)} Time Outliers detected but NOT excluded.")
    excluded_judges.update(bad_boys)
             
    df_filtered = df[~df['judge_ID'].isin(excluded_judges)]
    
    print(f"Excluded {len(excluded_judges)} judges.")
    print(f" - Uncomplete: {len(uncomplete_judges)}")
    print(f" - Time Outliers: {len(time_outliers)}")
    print(f" - Metric Outliers: {len(metric_outliers)}")
    
    return df_filtered
