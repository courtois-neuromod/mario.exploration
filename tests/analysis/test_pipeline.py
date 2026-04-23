import unittest
import pandas as pd
import numpy as np
import shutil
import tempfile
import os
import json
import matplotlib.pyplot as plt
from my_analysis.data_loader import load_and_aggregate
from my_analysis.quality import check_quality
from my_analysis.metrics import calculate_metrics
from my_analysis.pipeline import run_pipeline, Results

class TestAnalysisPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for data
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample CSV data
        self.sample_data = {
            'PROLIFIC_PID': ['JUDGE1', 'JUDGE2', 'JUDGE3'],
            'scene_id': ['1-1-1', '1-1-1', '1-1-1'],
            'art_sub': ['AI_1', 'AI_1', 'AI_1'],
            'hum_sub': ['HUM_1', 'HUM_1', 'HUM_1'],
            'thank_s.started': [1, 1, 1],
            'thank_s.stopped': [2, 2, 2],
            'anwsers': [
                json.dumps({"player1": {"q1": {"Clip_1": True, "Clip_2": False}, "q2": {"Clip_1": True, "Clip_2": True}}}),
                json.dumps({"player1": {"q1": {"Clip_1": True, "Clip_2": False}, "q2": {"Clip_1": False, "Clip_2": True}}}),
                json.dumps({"player1": {"q1": {"Clip_1": False, "Clip_2": True}, "q2": {"Clip_1": True, "Clip_2": True}}})
            ],
            'clip_num': [1, 1, 1], # Placeholder, logic in loader is complex regarding matching
            'display_clips.started': [10, 10, 10],
            'display_clips.stopped': [15, 12, 15], # Judge 2 is faster
            'clip_path': ['clip1.mp4', 'clip1.mp4', 'clip1.mp4'],
            'video_duration': [5, 5, 5]
        }
        
        # We need separate files for each judge to mimic real data structure
        # data_loader expects one file per participant
        
        for i in range(3):
            judge_id = f"JUDGE{i+1}"
            
            # Construct a minimal DF for this judge
            # We need rows corresponding to clips.
            # Let's say 2 clips.
            
            # Row 1: Clip 1
            row1 = {
                'PROLIFIC_PID': judge_id,
                'scene_id': '1-1-1',
                'art_sub': 'AI_1',
                'hum_sub': 'HUM_1',
                'thank_s.started': 1,
                'thank_s.stopped': 2,
                'anwsers': json.dumps({"player1": {"q1": {"Clip_1": True, "Clip_2": False}, "q2": {"Clip_1": True, "Clip_2": False}}}) if i < 2 else json.dumps({"player1": {"q1": {"Clip_1": False, "Clip_2": True}, "q2": {"Clip_1": False, "Clip_2": True}}}),
                'clip_num': 1,
                'display_clips.started': 10,
                'display_clips.stopped': 15,
                'clip_path': 'clip1.mp4',
                'video_duration': 5
            }
             # Row 2: Clip 2
            row2 = row1.copy()
            row2['clip_num'] = 2
            row2['clip_path'] = 'clip2.mp4'
            
            # Create DF
            df = pd.DataFrame([row1, row2])
            
            # Save
            df.to_csv(os.path.join(self.test_dir, f"{judge_id}.csv"), index=False)
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_load_and_aggregate(self):
        df = load_and_aggregate(self.test_dir)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 3 * 2 * 2) # 3 judges * 2 questions * 2 clips
        self.assertIn('judge_ID', df.columns)
        self.assertIn('visualisation_time', df.columns)
        
    def test_check_quality(self):
        df = load_and_aggregate(self.test_dir)
        # Mock some bad data
        # Judge 3 has different answers, check agreement
        # But limited data.
        df_clean = check_quality(df, self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'uncomplete_data.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'time_outliers.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'agreement_outliers.txt')))
        
    def test_calculate_metrics(self):
        df = load_and_aggregate(self.test_dir)
        metrics = calculate_metrics(df, ['dice', 'percent'], nb_panels=1, size_panel=1)
        self.assertIn('1-1-1', metrics)
        self.assertIn('player1', metrics['1-1-1'])
        self.assertIn('dice', metrics['1-1-1']['player1'])
        
    def test_run_pipeline(self):
        # Full run
        results = run_pipeline(path_data=self.test_dir, metrics=['dice'], nb_panels=1, size_panel=1, plot=True)
        
        # Check attribute access
        # results.1-1-1 is not valid python syntax for attribute but getattr works
        # The scene id is '1-1-1'.
        # Python attributes cannot start with number or contain dashes usually.
        # But specific implementation uses setattr/getattr.
        # So `results.1-1-1` is invalid syntax. 
        # User requested: `print(results.1-1-1.ppo.dice)`
        # If scene ID is "1-1-1", python parser interprets it as subtraction.
        # This is a user requirement issue!
        # "1-1-1" cannot be an attribute name accessed via dot notation directly in source code.
        # But `getattr(results, "1-1-1")` works.
        # User example code might be pseudo-code or they expect to use getattr or dictionary access?
        # "print(results.1-1-1.ppo.dice)" -> This will definitely fail in Python script.
        # Unless user means `results.s1_1_1` or similar?
        # But instruction says: "scene_id is a string like '1-1-1'".
        # I will document this limitation or maybe map to safe names?
        # But the user specifically asked for this. 
        # I will implement as requested (using setattr), and user will have to use getattr or interactive shell that supports it (none?)
        # Or maybe they use `vars(results)['1-1-1']`?
        
        scene_node = getattr(results, '1-1-1')
        player_node = getattr(scene_node, 'player1')
        self.assertTrue(hasattr(player_node, 'dice'))
        
        # Check plots generated
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'results')))
        
if __name__ == '__main__':
    unittest.main()
