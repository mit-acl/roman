import numpy as np
import os
import pickle
import argparse

from roman.map.map import load_roman_map, ROMANMap

def merge_demo_output(input_files, output_file):
    
    roman_maps = []
    for input_file in input_files:
        roman_maps.append(load_roman_map(input_file))
            
    merged = ROMANMap.concatenate(roman_maps)
        
    with open(os.path.expanduser(output_file), 'wb') as f:
        pickle.dump(merged, f)
    
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Merge tracking results')
    parser.add_argument('--input', '-i', nargs='+', type=str, help='Input files')
    parser.add_argument('--output', '-o', type=str, help='Output file')
    args = parser.parse_args()
    
    merge_demo_output(args.input, args.output)