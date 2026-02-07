#!/usr/bin/env python3
"""
Preprocess MIMIC-CXR dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    if pd.isna(text): 
        return ""
    return ' '.join(str(text).split())

def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    # Adjust input path to where kagglehub likely dumped it or where we moved it
    # We moved it to data/raw/mimic_cxr
    data_dir = project_root / 'data' / 'raw' / 'mimic_cxr'
    out_dir = project_root / 'data' / 'processed_mimic'
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Data Dir: {data_dir}")
    
    # We expect a metadata.csv or similar. 
    # Since the structure is {subject_id, study_id, dicom_id, text, ViewPosition}
    # We might need to find the CSV file.
    
    # MIMIC-CXR generally has 'mimic-cxr-2.0.0-metadata.csv' and a separate text file or 'mimic-cxr-2.0.0-negbio.csv' etc.
    # However, for this hackathon context, if the user downloads a specific version that simplifies it to "subject_id, ... text",
    # we should look for that.
    
    # We will try to load 'mimic_cxr_labels.csv' or similar if it exists, or look for standard MIMIC files.
    # Strategy: 
    # 1. Look for a large CSV that might contain 'text' or 'report'.
    # 2. If not found, look for metadata.csv and reports. Not implemented here due to complexity without seeing files.
    # We will proceed with the assumption of a unified CSV as per user description.
    
    potential_files = list(data_dir.glob('*.csv'))
    target_file = None
    for f in potential_files:
        # Heuristic: file > 10MB likely contains reports or full index
        if f.stat().st_size > 10 * 1024 * 1024: 
             target_file = f
             break
             
    if not target_file and potential_files:
        target_file = potential_files[0]
        
    if not target_file:
        print("No suitable CSV found.")
        return

    print(f"Processing {target_file}...")
    df = pd.read_csv(target_file)
    
    # Map columns based on user spec
    # "subject_id", "study_id", "dicom_id", "text", "ViewPosition"
    
    rename_map = {}
    if 'dicom_id' in df.columns: rename_map['dicom_id'] = 'filename_base'
    if 'text' in df.columns: rename_map['text'] = 'report'
    
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure filename has extension
    if 'filename_base' in df.columns:
        df['filename'] = df['filename_base'].astype(str) + '.jpg'
    elif 'image_id' in df.columns:
        df['filename'] = df['image_id'].astype(str) + '.jpg'
    
    # Ensure attributes exist
    if 'report' not in df.columns:
        print("CRITICAL: 'report' or 'text' column not found.")
        # Fallback for compilation: create empty
        df['report'] = "No report available."
        
    if 'indication' not in df.columns:
        df['indication'] = "Chest X-ray."
        
    # Filter ViewPosition
    if 'ViewPosition' in df.columns:
        df = df[df['ViewPosition'].isin(['PA', 'AP'])].copy()

    # Select and save
    keep_cols = ['filename', 'indication', 'report', 'subject_id']
    # Filter only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    df = df[keep_cols]
    
    # Split
    # Split by patient (subject_id) to avoid leakage
    patients = df['subject_id'].unique()
    train_pts, temp_pts = train_test_split(patients, test_size=0.2, random_state=42)
    val_pts, test_pts = train_test_split(temp_pts, test_size=0.5, random_state=42)
    
    train_df = df[df['subject_id'].isin(set(train_pts))]
    val_df = df[df['subject_id'].isin(set(val_pts))]
    test_df = df[df['subject_id'].isin(set(test_pts))]
    
    train_df.to_csv(out_dir / 'train.csv', index=False)
    val_df.to_csv(out_dir / 'val.csv', index=False)
    test_df.to_csv(out_dir / 'test.csv', index=False)
    
    print(f"Processed MIMIC-CXR: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

if __name__ == "__main__":
    main()
