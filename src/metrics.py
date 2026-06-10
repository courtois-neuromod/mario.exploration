import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from .utils import calculate_dice
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


def metrics_by_panel_sizes(df, metrics, by, panel_sizes=None, n_bootstraps=100, verbose=False):
    """
    Calculate inter-rater agreement metrics using bootstrapped panel splits.

    Args:
        df (pd.DataFrame): Input data with columns scene_id, player, participant,
                           question, clip, answer.
        metrics (list[str]): Metrics to compute. Supported: 'dice', 'percent', 'kappa'.
        panel_sizes (list[int]): Panel sizes to sweep over. Cannot be None or empty.
        n_bootstraps (int): Number of bootstrap / swap iterations per panel size.

    Returns:
        pd.DataFrame: Flat table with columns:
            scene_id | player | metric | question | panel_size | swap_0 | … | swap_{n_bootstraps-1}
            Each row holds the n_bootstraps raw scores for one
            (scene, player, metric, question, panel_size) combination.
    """
    if not panel_sizes:
        raise ValueError("panel_sizes cannot be None or empty.")

    rows = []
    groups = df.groupby(by)
    pbar = tqdm(groups)

    for keys, group_df in pbar:
        pbar.set_description(f"Subset: {keys}")

        available_judges = group_df['judge_ID'].unique()
        n_judges = len(available_judges)

        for size_panel in panel_sizes:
            if n_judges < 2 * size_panel:
                print(f"  Skipping panel size {size_panel}: not enough judges ({n_judges})")
                continue
            
            if verbose:
                print(f"  Testing panel size: {size_panel}")
            metric_scores = {m: {keys: []} for m in metrics}

            # Collect n_bootstraps raw scores for every (metric, question)

            for i in range(n_bootstraps):
                shuffled = np.random.permutation(available_judges)
                p1_judges = shuffled[:size_panel]
                p2_judges = shuffled[size_panel:2 * size_panel]
                indexs = ['scene_id', 'question', 'clip', 'player']
                for col in by:
                    indexs.remove(col)

                pivoted = group_df.pivot_table(
                    index=indexs,
                    columns='judge_ID',
                    values='answer'
                    )
                v1 = (
                    pivoted[p1_judges]
                    .mode(axis=1)[0]
                    .dropna()
                    .astype(int)
                )
                v2 = (
                    pivoted[p2_judges]
                    .mode(axis=1)[0]
                    .dropna()
                    .astype(int)
                )
                common_idx = v1.index.intersection(v2.index)

                v1 = v1.loc[common_idx]
                v2 = v2.loc[common_idx]

                for metric in metrics:
                    if metric == 'dice':
                        value = calculate_dice(v1, v2)
                    elif metric == 'percent':
                        value = (v1 == v2).mean()
                    elif metric == 'kappa' in metrics:
                        try:
                            k = cohen_kappa_score(v1, v2)
                            if not np.isnan(k):
                                value = k
                        except Exception:
                            pass
                    else:
                        value = np.nan
                    row ={by[i]:keys[i] for i in range(len(by))}
                    row['metric'] = metric
                    row['panel_size'] = size_panel
                    row['swap']= f'swap_{i}'
                    row['value']=value
                    rows.append(row)

    id_cols = ['scene_id', 'player',  'question', 'metric', 'panel_size', 'swap', 'value']
    df_results = pd.DataFrame(rows, columns=id_cols)
    return df_results

def metrics_split_panel(df, metrics= ['dice'], n_bootstraps=100):

    rows = []
    groups = df.groupby('question')

    for key, group_df in groups:
        available_judges = group_df['judge_ID'].unique()
        n_judges_panel = group_df['judge_ID'].nunique()//2 

        print(f"Processing subset: ", key)

        for i in range(n_bootstraps):
            shuffled = np.random.permutation(available_judges)
            p1_judges = shuffled[:n_judges_panel]
            group_df['half'] = np.where(
                group_df['judge_ID'].isin(p1_judges),
                "First half",
                "Second half")
            df_score = (
                group_df
                .groupby(['clip_uid', 'half'])
                .agg({'answer': 'mean',
                    'scene_player': 'first',
                    'clip': 'first'
                })
                .reset_index()
                )
            df_score['answer_Low'] = df_score['answer'].apply(lambda x: True if x < 0.2 else False)
            df_score['answer_Hight'] = df_score['answer'].apply(lambda x: True if x > 0.8 else False)
            
            for target in ['Low', 'Hight']:

                pivoted = df_score.pivot_table(
                        index='clip_uid',
                        columns='half',
                        values=f'answer_{target}'
                        )
             
                v1 = pivoted['First half'].astype(int)
                v2 = pivoted['Second half'].astype(int)

                common_idx = v1.index.intersection(v2.index)

                v1 = v1.loc[common_idx]
                v2 = v2.loc[common_idx]
                
                
                for metric in metrics:
                    if metric == 'dice':
                        value = calculate_dice(v1, v2)
                    elif metric == 'percent':
                        value = (v1 == v2).mean()
                    elif metric == 'kappa' in metrics:
                        try:
                            k = cohen_kappa_score(v1, v2)
                            if not np.isnan(k):
                                value = k
                        except Exception:
                            pass
                    else:
                        value = np.nan

                    rows.append({'question':key,
                                 'target':target,
                                'swap': f'swap_{i}',
                                'metric': metric,
                                'value':value})
                    
    id_cols = ['question', 'target','metric', 'swap', 'value']

    return pd.DataFrame(rows, columns=id_cols )

def corr_from_split(df):
    results= []
    for i in range(100):
        for question, df_score in df.groupby(['question']):
            # Assigner chaque réponse à la moitié 1 ou 2 à l'intérieur de chaque scène
            df_score = df_score.sample(frac=1).reset_index(drop=True)
            df_score['rank'] = df_score.groupby('scene_player').cumcount()
            df_score['n'] = df_score.groupby('scene_player')['answer'].transform('size')

            df_score['half'] = np.where(
                df_score['rank'] < df_score['n'] / 2,
                "First half",
                "Second half"
            )

            # Moyenne par clip dans chaque moitié
            df_score = (
                df_score
                .groupby(['clip_uid', 'half'])
                .agg({
                    'answer': 'mean',
                    'scene_player': 'first',
                    'clip': 'first'
                })
                .reset_index()
            )

            for scene, df_scene in df_score.groupby("scene_player"):

                # une ligne par clip
                pivot = df_scene.pivot(
                    index="clip_uid",
                    columns="half",
                    values="answer"
                ).dropna()

                r, p = pearsonr(
                    pivot["First half"],
                    pivot["Second half"]
                )
                results.append({
                    'swap_n':i,
                    'question':question[0],
                    'scene': scene,
                    'r': r,
                    'p':p
                })

    results = pd.DataFrame(results)
    results_grp = results.groupby(['question', 'scene'], as_index=False).agg({
                    'r': 'mean',
                    'p':'mean'
                })

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
            print('no panel size was given')
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