import numpy as np
import os
import pickle
import argparse

def merge_demo_output(input_files, output_file):
    
    trackers, pose_history, times = [], [], []

    for i in range(len(input_files)):
        with open(os.path.expanduser(input_files[i]), 'rb') as f:
            pickle_data = pickle.load(f)
            tr, ph, ti = pickle_data
            trackers.append(tr)
            pose_history.append(ph)
            times.append(ti)
            
    trackers_res = trackers[0]
    pose_history_res = pose_history[0]
    times_res = times[0]
    
    for i in range(1, len(input_files)):
        for seg in trackers[i].segment_nursery + trackers[i].segments \
            + trackers[i].inactive_segments + trackers[i].segment_graveyard:
            seg.id += trackers_res.segment_graveyard[-1].id + int(1e5)

        trackers_res.segments += trackers[i].segments
        trackers_res.segment_nursery += trackers[i].segment_nursery
        trackers_res.inactive_segments += trackers[i].inactive_segments
        trackers_res.segment_graveyard += trackers[i].segment_graveyard
        
        pose_history_res += pose_history[i]
        times_res += times[i]
        
    with open(os.path.expanduser(output_file), 'wb') as f:
        pickle.dump((trackers_res, pose_history_res, times_res), f)
    
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Merge tracking results')
    parser.add_argument('--input', '-i', nargs='+', type=str, help='Input files')
    parser.add_argument('--output', '-o', type=str, help='Output file')
    args = parser.parse_args()
    
    merge_demo_output(args.input, args.output)