import os
import pandas as pd
from .data_loader import load_and_aggregate
from .quality import check_quality
from .metrics import calculate_metrics
from .viz import plot_results

class Results:
    """
    Container for analysis results with attribute-style access.
    Structure: results.scene.player.metric -> value/plot
    """
    def __init__(self):
        self._data = {}

    def _set(self, scene, player, key, value):
        scene = str(scene)
        player = str(player)
        if scene not in self._data:
            self._data[scene] = ResultsNode(scene)
            setattr(self, scene, self._data[scene])
        
        scene_node = self._data[scene]
        if not hasattr(scene_node, player):
            player_node = ResultsNode(player)
            setattr(scene_node, player, player_node)
        
        player_node = getattr(scene_node, player)
        setattr(player_node, key, value)

    def _set_scene_attr(self, scene, key, value):
        scene = str(scene)
        if scene not in self._data:
            self._data[scene] = ResultsNode(scene)
            setattr(self, scene, self._data[scene])
        
        scene_node = self._data[scene]
        setattr(scene_node, key, value)
        
    def __repr__(self):
        return f"<Results object with scenes: {list(self._data.keys())}>"

    def __getitem__(self, key):
        return self._data[key]

class ResultsNode:
    """Helper class for nested attribute access."""
    def __init__(self, name):
        self._name = name
        self._data = {}

    def __repr__(self):
        return f"<ResultsNode: {self._name}>"

    def __getitem__(self, key):
        return getattr(self, key)


def run_pipeline(path_data, output_dir=None, metrics=['dice', 'percent', 'kappa'], nb_panels=1, size_panel=10, 
                 scene_id=None, comparason_id=None, plot=True):
    """
    Runs the complete analysis pipeline.
    
    Args:
        path_data (str): Path to data folder.
        output_dir (str): Path to save output results. If None, defaults to path_data.
        metrics (list): List of metrics to compute.
        nb_panels (int): Number of panels (for heuristic default size determination).
        size_panel (int): Size of each panel.
        scene_id (dict): Optional filter/mapping.
        comparason_id (dict): Optional filter/mapping.
        plot (bool): Whether to generate plots.
        
    Returns:
        Results: Object containing metrics and plots.
    """
    
    if output_dir is None:
        output_dir = path_data
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    print("Loading data...")
    df = load_and_aggregate(path_data)
    
    # 2. Quality Check
    print("Checking data quality...")
    df_clean = check_quality(df, output_dir)
    
    if scene_id:
        filtered_dfs = []
        for s_key, players in scene_id.items():
            sub_df = df_clean[df_clean['scene_id'] == str(s_key)]
            if players:
                sub_df = sub_df[sub_df['player'].isin(players)]
            filtered_dfs.append(sub_df)
            
        if filtered_dfs:
            df_clean = pd.concat(filtered_dfs)
        else:
            print("Warning: No data matched the provided scene_id filter.")
            
    # 3. Calculate Metrics
    print("Calculating metrics...")
    metrics_results = calculate_metrics(df_clean, metrics, nb_panels, size_panel)
    
    # 4. Generate Visualizations (if enabled)
    plots = {}
    if plot:
         print("Generating plots...")
         plots_output_path = os.path.join(output_dir, 'results')
         plots = plot_results(metrics_results, df_clean, output_path=plots_output_path, 
                             nb_panels=nb_panels, size_panel=size_panel)
    
    # 5. Build Results Object
    results = Results()
    
    # Add metrics
    for scene, players in metrics_results.items():
        for player, player_metrics in players.items():
            agreements_key = f"{scene}_{player}_agreements"
            if plot and agreements_key in plots:
                results._set(scene, player, "agreements", plots[agreements_key])
            
            for metric, scores in player_metrics.items():
                results._set(scene, player, metric, scores)
            
            viz_key = f"{scene}_{player}_visualisation_time"
            if plot and viz_key in plots:
                results._set(scene, player, "visualisation_time", plots[viz_key])
            
        corr_key = f"{scene}_corr"
        if plot and corr_key in plots:
            results._set_scene_attr(scene, "corr", plots[corr_key])

    return results
