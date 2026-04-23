import pandas as pd
import json
import os
import re
import glob
import datetime
from pathlib import Path
from .utils import get_clip_duration

def file_newer_than(path, cutoff):
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    if mtime < cutoff:
        print(f"File {path} is older than cutoff date: {cutoff}")
        return False
    else:
        return True

def check_consent(df, consent_path):
    if df['consent_form.block_1/question5'].dropna().iloc[0] == True:
        return True
    else:
        with open(consent_path, 'a') as f:
            f.write(f"{df['PROLIFIC_PID'].dropna().iloc[0]}/{df['participant'].dropna().iloc[0]}: No consent given\n")
        return False

def this_sub_is_clean(df, list_outlier):
    participant_ID = df['PROLIFIC_PID'].iloc[0]
    return not participant_ID in list_outlier

def file_is_ok(df, filename, list_outlier, cutoff, output_dir):
    try:
        is_clean = False
        consent_ok = False
        if file_newer_than(filename, cutoff):
            consent_ok = check_consent(df, output_dir)
            is_clean = this_sub_is_clean(df, list_outlier)

        return is_clean and consent_ok
        
    except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return False


def load_and_aggregate(path_data, cutoff=datetime.datetime(2026, 3, 25), save_data=False):
    """
    Loads all CSV files in the specified directory and aggregates them into a single DataFrame.
    
    Args:
        path_data (str): Path to the directory containing the data CSV files.
        
    Returns:
        pd.DataFrame: A long-format DataFrame containing the aggregated data.
    """
    all_files = glob.glob(os.path.join(path_data, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {path_data}")
        
    aggregated_data = []
    outliers = []
    outliers_path = os.path.join('outputdata', 'quality_check', 'outliers.txt')
    consent_path = Path(os.path.join('outputdata', 'consent_check', 'no_consent.txt'))
    consent_path.parent.mkdir(parents=True, exist_ok=True)
    consent_path.touch()

    with open(outliers_path) as file:
        for line in file:
            IDs = line.split(' ')[0]
            outliers.append(IDs)
    
    for filename in all_files:

        df = pd.read_csv(filename)

        if file_is_ok(df, filename, outliers, cutoff, consent_path):
            print(f"Processing file {filename}")
            try:
                # Extract Judge ID
                judge_id = df['PROLIFIC_PID'].dropna().iloc[0] if not df['PROLIFIC_PID'].dropna().empty else "UNKNOWN"
                participant = df['participant'].dropna().iloc[0]
                
                # Extract Scene ID
                raw_scene_id = re.search(r'w(\d+)l(\d+)_scene-(\d+)', df['clip_path'].dropna().iloc[0]).group(0)
                scene_id = str(raw_scene_id).replace("_scene-", "s")
                scene_id = scene_id.strip("'").strip('"').strip("’").strip("‘")
                
                # Check completed status
                done = False
                if 'thank_s.started' in df.columns and 'thank_s.stopped' in df.columns:
                    if not df['thank_s.started'].dropna().empty and not df['thank_s.stopped'].dropna().empty:
                        done = True
                
                # Process answers
                answers_col = 'anwsers' if 'anwsers' in df.columns else 'answers'
                
                if answers_col in df.columns and not df[answers_col].dropna().empty:
                    answers_json = df[answers_col].dropna().iloc[-1]
                    try:
                        answers_dict = json.loads(answers_json)
                        
                        # Structure: player -> question -> clip -> answer (bool)
                        for player, questions in answers_dict.items():
                            for question, clips in questions.items():
                                for clip_key, answer in clips.items():
                                    try:
                                        clip_num = int(clip_key.split('_')[1])-1
                                    except (IndexError, ValueError):
                                        continue 

                                    if player=='ppo':
                                        path_pattern = 'ppo_mario_ep-'
                                    elif 'im' in player:
                                        path_pattern = player[4:]+'_epoch='
                                    else:
                                        path_pattern = player
                                    
                                    clip_row = df[
                                        (df['clip_num'] == clip_num) &
                                        (df['clip_path'].str.split('/').str[1].str.contains(path_pattern))
                                        ]
                                    if not clip_row.empty:
                                        vis_times = []
                                        clip_path_rel = clip_row['clip_path'].iloc[0]
                                        json_path_rel = clip_row['clip_path'].iloc[0].replace('videos', 'infos').replace('.mp4', '.json')
                                        
                                        for idx in clip_row.index:
                                            v_start = df.at[idx, 'display_clips.started']
                                            v_stop = df.at[idx, 'display_clips.stopped']
                                            
                                            if pd.notnull(v_start) and pd.notnull(v_stop):
                                                vis_times.append(v_stop - v_start)
                                            
                                        vis_time = max(vis_times) if vis_times else None

                                        with open(json_path_rel) as json_data:
                                            d = json.load(json_data)
                                            cleared = d['Cleared']

                                        video_duration = get_clip_duration(clip_path_rel, cleared)

                                        aggregated_data.append({
                                            'participant': participant,
                                            'judge_ID': judge_id,
                                            'scene_id': scene_id,
                                            'player': player,
                                            'question': question,
                                            'clip': clip_num,
                                            'answer': answer,
                                            'visualisation_time': vis_time,
                                            'clip_path': clip_path_rel,
                                            'video_duration': video_duration,
                                            'done': done,
                                        })
                                        
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for file {filename}")
                        continue

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

    data = pd.DataFrame(aggregated_data)
    print(data['judge_ID'].nunique(),' were collected')

    if save_data:
        data_path = os.path.join('data', 'preload_data.csv')
        data.to_csv(data_path)

    return data