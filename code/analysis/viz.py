import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns
import os

# TODO: - add a plot=True or False argument to use the plot functions in notebooks
#       - add a save= Treu or False to give the option to save new figures 

# --- Helper Functions ---
def _plot_metric_ax(ax, questions_data, title=None, show_legend=True, show_final_values=True, ylabel='means'):
    """
    Helper function to plot a single metric panel.
    """
    # Get all questions
    questions = list(questions_data.keys())
    
    # Generate summer colors for questions
    summer = cm.get_cmap('summer', len(questions))
    
    # Plot each question as a separate line
    for q_idx, question in enumerate(questions):
        panel_size_scores = questions_data[question]
        
        if not panel_size_scores:
            continue
        
        # Sort by panel size
        sorted_items = sorted(panel_size_scores.items())
        panel_sizes = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        # Filter out NaNs
        valid_data = [(ps, s) for ps, s in zip(panel_sizes, scores) if not np.isnan(s)]
        if not valid_data:
            continue
        
        panel_sizes_valid, scores_valid = zip(*valid_data)
        
        # Get color from summer colormap
        color = summer(q_idx)
        
        # Plot line
        ax.plot(panel_sizes_valid, scores_valid, 'o-', 
               linewidth=2, markersize=6, label=question, color=color, alpha=0.8)
    
    # Formatting
    if title:
        ax.set_title(title, fontweight='bold', fontsize=11)
    
    ax.set_xlabel('panel sizes', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_legend:
        ax.legend(title='questions', loc='upper left', framealpha=0.9)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add text box with final values (last panel size)
    if show_final_values:
        final_values_text = "Final values:\n"
        has_values = False
        for question in questions:
            panel_size_scores = questions_data[question]
            if panel_size_scores:
                # Get the last (largest) panel size
                max_panel_size = max(panel_size_scores.keys())
                final_value = panel_size_scores[max_panel_size]
                if not np.isnan(final_value):
                    final_values_text += f"{question}: {final_value:.3f}\n"
                    has_values = True
        
        if has_values:
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax.text(0.98, 0.02, final_values_text.strip(), transform=ax.transAxes,
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=props)

# --- Correlation Plots ---
def plot_correlation_matrix(df, output_path=None):
    """
    Generates a correlation matrix plot between questions at the scene level.
    """
    plots = {}
    grouped = df.groupby('scene_id')
    
    for scene, group_df in grouped:
        mean_resp = group_df.groupby(['clip', 'player', 'question'])['answer'].mean().unstack()
        
        if mean_resp.empty or mean_resp.shape[1] < 2:
            continue
            
        corr = mean_resp.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
                    cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(f"Question Correlation Matrix\nScene: {scene}", 
                     fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_key = f"{scene}_corr"
        plots[plot_key] = fig
        
        if output_path:
            scene_dir = os.path.join(output_path, scene)
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            
            save_path = os.path.join(scene_dir, f"{plot_key}.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
    return plots

def plot_mean_correlation_matrix(df, output_path=None):
    """
    Generates a mean correlation matrix plot across all scenes.
    """
    correlation_matrices = []
    for scene, group_df in df.groupby('scene_id'):
        mean_resp = group_df.groupby(['clip', 'player', 'question'])['answer'].mean().unstack()
        if mean_resp.empty or mean_resp.shape[1] < 2:
            continue
        correlation_matrices.append(mean_resp.corr())
    
    if not correlation_matrices:
        print("No valid correlation matrices found to average.")
        return {}
        
    all_questions = sorted(list(set().union(*[c.columns for c in correlation_matrices])))
    aligned_matrices = []
    for corr in correlation_matrices:
        aligned = corr.reindex(index=all_questions, columns=all_questions)
        aligned_matrices.append(aligned.values)
        
    mean_corr_values = np.nanmean(np.array(aligned_matrices), axis=0)
    mean_corr = pd.DataFrame(mean_corr_values, index=all_questions, columns=all_questions)
    
    mask = np.triu(np.ones_like(mean_corr, dtype=bool), k=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mean_corr, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn', 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title("Mean Question Correlation Matrix\n(Average across all scenes)", 
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_key = "questions_corr"
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fig.savefig(os.path.join(output_path, f"{plot_key}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    return {plot_key: fig}

# --- Agreement Plots ---
def plot_results(results_data, df, output_path=None, nb_panels=1, size_panel=10):
    """
    Generates single-panel agreement plots (Question vs Mean).
    Also includes visualisation time distributions.
    """
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
        
    plots = {}
    
    # 1. Combined Agreement Metrics Plots (Question vs Score)
    for scene, players in results_data.items():
        for player, metrics in players.items():
            metrics_to_plot = ['dice', 'kappa', 'percent']
            available_metrics = [m for m in metrics_to_plot if m in metrics and metrics[m]]
            
            if not available_metrics:
                continue
            
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
            if n_metrics == 1: axes = [axes]
            
            fig.suptitle(f'Agreement Metrics: {scene} - {player}\\nN= {size_panel}, panel = {nb_panels}', 
                        fontsize=14, fontweight='bold')
            
            for col_idx, metric in enumerate(available_metrics):
                questions_data = metrics[metric]
                questions = list(questions_data.keys())
                scores = list(questions_data.values())
                
                valid_data = [(q, s) for q, s in zip(questions, scores) if not np.isnan(s)]
                if not valid_data: continue
                q_valid, s_valid = zip(*valid_data)
                
                ax = axes[col_idx]
                positions = list(range(len(q_valid)))
                ax.plot(positions, s_valid, 'o-', linewidth=2, markersize=8, color='steelblue', alpha=0.7)
                
                ax.set_title(f'{metric.capitalize()} Coefficients', fontweight='bold', fontsize=12)
                ax.set_ylabel('Mean', fontsize=10)
                ax.set_xlabel('Questions', fontsize=10)
                ax.set_xticks(positions)
                ax.set_xticklabels(q_valid, rotation=45, ha='right')
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plot_key = f"{scene}_{player}_agreements"
            plots[plot_key] = fig
            
            if output_path:
                fig.savefig(os.path.join(output_path, f"{plot_key}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

    # 2. Visualisation Time Distribution (Box Plot)
    grouped = df.groupby(['scene_id', 'player'])
    for (scene, player), group_df in grouped:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='clip', y='visualisation_time', data=group_df, ax=ax, palette="Set3")
        ax.set_title(f"Visualisation Time Distribution per Clip\nScene: {scene}, Player: {player}")
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Clip Number")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_key = f"{scene}_{player}_visualisation_time"
        plots[plot_key] = fig
        if output_path:
            fig.savefig(os.path.join(output_path, f"{plot_key}.png"))
            plt.close(fig)

    # 3. Correlation Plot
    plots.update(plot_correlation_matrix(df, output_path=output_path))
            
    return plots

def plot_panel_size_sweep(results_data, output_path=None):
    """Generates panel size sweep plots (Panel Size vs Score)."""
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    plots = {}
    for scene, players in results_data.items():
        for player, metrics in players.items():
            metrics_to_plot = ['dice', 'kappa', 'percent']
            available_metrics = [m for m in metrics_to_plot if m in metrics and metrics[m]]
            if not available_metrics: continue
            
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
            if n_metrics == 1: axes = [axes]
            
            fig.suptitle(f'{scene} - {player}', fontsize=14, fontweight='bold')
            
            for col_idx, metric in enumerate(available_metrics):
                questions_data = metrics[metric]
                _plot_metric_ax(axes[col_idx], questions_data, 
                                title=f'{metric.capitalize()} mean by panel size',
                                show_legend=True, show_final_values=True)
            
            plt.tight_layout()
            plot_key = f"{scene}_{player}_agreements"
            plots[plot_key] = fig
            
            if output_path:
                scene_player_dir = os.path.join(output_path, scene, player)
                if not os.path.exists(scene_player_dir): os.makedirs(scene_player_dir)
                fig.savefig(os.path.join(scene_player_dir, f"{scene}_{player}_agreements.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
    return plots

def plot_scene_agreement_summary(results_data, output_path=None):
    """Generates a summary plot for each scene (Mean + Individual Players)."""
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    plots = {}
    metrics_to_plot = ['dice', 'kappa', 'percent']
    
    for scene, players_data in results_data.items():
        players_list = list(players_data.keys())
        if not players_list: continue
        first_player_metrics = players_data[players_list[0]]
        available_metrics = [m for m in metrics_to_plot if m in first_player_metrics and first_player_metrics[m]]
        if not available_metrics: continue
        
        n_metrics = len(available_metrics)
        n_rows = len(players_list) + 1
        fig, axes = plt.subplots(n_rows, n_metrics, figsize=(6 * n_metrics, 4 * n_rows), squeeze=False)
        fig.suptitle(f'{scene} - Agreement Summary (Mean + Players)', fontsize=16, fontweight='bold', y=0.99)
        
        # Mean Data Computation
        mean_data = {m: {} for m in available_metrics}
        for metric in available_metrics:
            temp_agg = {} 
            for player in players_list:
                p_metrics = players_data[player].get(metric, {})
                for q, size_dict in p_metrics.items():
                    if q not in temp_agg: temp_agg[q] = {}
                    for size, score in size_dict.items():
                        if size not in temp_agg[q]: temp_agg[q][size] = []
                        if not np.isnan(score): temp_agg[q][size].append(score)
            for q, sizes in temp_agg.items():
                mean_data[metric][q] = {size: np.mean(scores) for size, scores in sizes.items() if scores}
        
        # Plot Mean (Row 0)
        for col_idx, metric in enumerate(available_metrics):
            ax = axes[0, col_idx]
            _plot_metric_ax(ax, mean_data[metric], title=f'MEAN - {metric.capitalize()}',
                            show_legend=(col_idx==0), show_final_values=True)
            ax.set_title(f'MEAN - {metric.capitalize()}', fontweight='bold', fontsize=12, color='black')
            ax.set_facecolor('#f8f9fa')
            
        # Plot Players (Row 1..N)
        for row_idx, player in enumerate(players_list):
            for col_idx, metric in enumerate(available_metrics):
                ax = axes[row_idx + 1, col_idx]
                questions_data = players_data[player].get(metric, {})
                _plot_metric_ax(ax, questions_data, title=f'{player} - {metric.capitalize()}',
                                show_legend=False, show_final_values=True)
                                
        plt.tight_layout()
        plot_key = f"{scene}_agreement"
        plots[plot_key] = fig
        
        if output_path:
            scene_dir = os.path.join(output_path, scene)
            if not os.path.exists(scene_dir): os.makedirs(scene_dir)
            fig.savefig(os.path.join(scene_dir, f"{scene}_agreement.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
    return plots

def plot_global_agreement_summary(results_data, output_path=None):
    """Generates global agreement summary plot."""
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    plots = {}
    metrics_to_plot = ['dice', 'kappa', 'percent']
    
    scenes_list = list(results_data.keys())
    if not scenes_list: return {}
    available_metrics = metrics_to_plot
    
    scene_means_data = {}
    grand_mean_accumulator = {m: {} for m in available_metrics}
    
    for scene in scenes_list:
        players_data = results_data[scene]
        players_list = list(players_data.keys())
        if not players_list: continue
        scene_means_data[scene] = {m: {} for m in available_metrics}
        
        for metric in available_metrics:
            temp_agg = {} 
            for player in players_list:
                p_metrics = players_data[player].get(metric, {})
                for q, size_dict in p_metrics.items():
                    if q not in temp_agg: temp_agg[q] = {}
                    for size, score in size_dict.items():
                        if size not in temp_agg[q]: temp_agg[q][size] = []
                        if not np.isnan(score): temp_agg[q][size].append(score)
            
            for q, sizes in temp_agg.items():
                scene_means_data[scene][metric][q] = {}
                for size, scores in sizes.items():
                    if scores:
                        val = np.mean(scores)
                        scene_means_data[scene][metric][q][size] = val
                        if q not in grand_mean_accumulator[metric]: grand_mean_accumulator[metric][q] = {}
                        if size not in grand_mean_accumulator[metric][q]: grand_mean_accumulator[metric][q][size] = []
                        grand_mean_accumulator[metric][q][size].append(val)
                        
    grand_mean_data = {m: {} for m in available_metrics}
    for metric in available_metrics:
        for q, sizes in grand_mean_accumulator[metric].items():
            grand_mean_data[metric][q] = {size: np.mean(vals) for size, vals in sizes.items() if vals}
            
    n_metrics = len(available_metrics)
    n_rows = len(scenes_list) + 1
    fig, axes = plt.subplots(n_rows, n_metrics, figsize=(6 * n_metrics, 4 * n_rows), squeeze=False)
    fig.suptitle('Global Agreement Summary (Grand Mean + Scenes)', fontsize=16, fontweight='bold', y=0.995)
    
    for col_idx, metric in enumerate(available_metrics):
        ax = axes[0, col_idx]
        _plot_metric_ax(ax, grand_mean_data[metric], title=f'GRAND MEAN - {metric.capitalize()}',
                        show_legend=(col_idx==0), show_final_values=True)
        ax.set_title(f'GRAND MEAN - {metric.capitalize()}', fontweight='bold', fontsize=12, color='black')
        ax.set_facecolor('#e6f3ff')
        
    for row_idx, scene in enumerate(scenes_list):
        for col_idx, metric in enumerate(available_metrics):
            ax = axes[row_idx + 1, col_idx]
            data = scene_means_data.get(scene, {}).get(metric, {})
            _plot_metric_ax(ax, data, title=f'{scene} (Mean) - {metric.capitalize()}',
                            show_legend=False, show_final_values=True)
                            
    plt.tight_layout()
    plot_key = "global_agreement_summary"
    plots[plot_key] = fig
    
    if output_path:
        fig.savefig(os.path.join(output_path, "global_agreement_summary.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return plots

# --- Quality Check Plots ---
def plot_scene_quality(scene, player, df_plot_time, df_metrics, outliers, output_path=None):
    """
    Generates quality check plots for a specific scene and player.
    
    Args:
        scene (str): Scene ID.
        player (str): Player ID.
        df_plot_time (pd.DataFrame): Time data for plotting.
        df_metrics (pd.DataFrame): Metrics data for plotting.
        outliers (dict): Dict with 'time' (set of judges) and 'metrics' (set of (judge, metric) tuples).
        output_path (str): Directory to save plots.
    """
    # Prepare color map
    time_outlier_judges = outliers.get('time', set())
    metric_outlier_events = outliers.get('metrics', set())
    
    # Need judges from metric_outlier_events
    metric_outlier_judges = {j for j, m in metric_outlier_events}
    
    all_problem_judges = time_outlier_judges.union(metric_outlier_judges)
    unique_problems = sorted(list(all_problem_judges))
    
    palette = sns.color_palette("hsv", len(unique_problems) if unique_problems else 1)
    judge_color_map = {judge: palette[i] for i, judge in enumerate(unique_problems)}
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Quality Control - Scene: {scene}, Player: {player}', fontsize=16)
    
    # Plot 1: Time vs Clip
    ax1 = axes[0]
    if not df_plot_time.empty:
        # Separate data
        df_plot_time['is_outlier'] = df_plot_time['is_outlier'].fillna(False)
        valid_points = df_plot_time[~df_plot_time['is_outlier']]
        outlier_points = df_plot_time[df_plot_time['is_outlier']]
        
        # Plot Valid (Grey, zorder 1)
        if not valid_points.empty:
            ax1.scatter(valid_points['clip'], valid_points['visualisation_time'], 
                        c='lightgrey', s=30, zorder=1, alpha=0.8, edgecolors='none', label='Valid')
            
        # Plot Outliers (Colored, zorder 5)
        if not outlier_points.empty:
            # We need to color each point
            colors = [judge_color_map.get(j, 'red') for j in outlier_points['judge_ID']]
            ax1.scatter(outlier_points['clip'], outlier_points['visualisation_time'], 
                        c=colors, s=50, zorder=5, alpha=0.8, edgecolors='none')
        
        # Add threshold line (Mean * 0.75 or similar)
        # Assuming clips are numeric
        clip_durations = df_plot_time.groupby('clip')['video_duration'].mean()
        if not clip_durations.empty:
            ax1.plot(clip_durations.index, clip_durations.values * 0.75, 'k:', label='75% Duration Threshold', alpha=0.5)

    ax1.set_xlabel('Clip Number')
    ax1.set_ylabel('Visualisation Time (s)')
    ax1.set_title('Max Visualisation Time per Judge/Clip')

    # Plot 2: Agreement Metrics
    ax2 = axes[1]
    if df_metrics is not None and not df_metrics.empty:
        df_melt = df_metrics.reset_index().melt(id_vars='judge_ID', 
                                               value_vars=['percent', 'kappa', 'dice'], 
                                               var_name='Metric', value_name='Score')
        
        metrics_map = {'percent': 0, 'kappa': 1, 'dice': 2}
        df_melt['x_base'] = df_melt['Metric'].map(metrics_map)
        rng = np.random.RandomState(42)
        df_melt['x_jittered'] = df_melt['x_base'] + rng.uniform(-0.1, 0.1, size=len(df_melt))
        
        # Identify outliers in melted data
        # An event is (judge, metric)
        df_melt['is_outlier'] = df_melt.apply(lambda row: (row['judge_ID'], row['Metric']) in metric_outlier_events, axis=1)
        
        valid_metrics = df_melt[~df_melt['is_outlier']]
        outlier_metrics = df_melt[df_melt['is_outlier']]
        
        # Plot Valid
        if not valid_metrics.empty:
            ax2.scatter(valid_metrics['x_jittered'], valid_metrics['Score'], 
                        c='lightgrey', s=30, zorder=1, alpha=0.8)
            
        # Plot Outliers
        if not outlier_metrics.empty:
            colors = [judge_color_map.get(j, 'red') for j in outlier_metrics['judge_ID']]
            ax2.scatter(outlier_metrics['x_jittered'], outlier_metrics['Score'], 
                        c=colors, s=50, zorder=5, alpha=0.8)
                    
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['Percent', 'Kappa', 'Dice'])
        ax2.set_ylabel('Score')
        ax2.set_title('Agreement Metrics')
        
        # Threshold lines
        for i, metric in enumerate(['percent', 'kappa', 'dice']):
             vals = df_metrics[metric]
             mean_val = vals.mean()
             std_val = vals.std()
             thresh = mean_val - 3 * std_val
             ax2.hlines(thresh, i-0.3, i+0.3, colors='red', linestyles=':', label='-3 SD' if i==0 else "")

    plt.tight_layout()
    
    plot_key = f"quality_{scene}_{player}"
    if output_path:
        if not os.path.exists(output_path): os.makedirs(output_path)
        fig.savefig(os.path.join(output_path, f"{plot_key}.png"))
        plt.close(fig)
        
    return fig
