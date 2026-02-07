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
    
    # Map 'dicom_id' to 'filename' (usually needs .jpg appended)
    # CRITICAL: MIMIC-CXR images are often nested (files/p10/p100.../view.jpg)
    # We need to map the ID to the ACTUAL relative path.
    
    print("Mapping image paths (this may take a minute)...")
    image_map = {}
    # Walk through the data dir to find all images
    # Supports jpg, png, dcm
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Key: filename without extension (often dicom_id)
                key = Path(file).stem
                # Value: Relative path from data_dir
                rel_path = Path(root) / file
                rel_path = rel_path.relative_to(data_dir)
                image_map[key] = str(rel_path)
    
    print(f"Found {len(image_map)} images.")
    
    def get_rel_path(row):
        # file_id could be in 'dicom_id', 'image_id', or 'filename_base'
        # We renamed to 'filename_base' or 'filename' above or below?
        # The rename happens *before* this hook.
        
        # Check potential ID columns
        candidates = [row.get('filename_base'), row.get('dicom_id'), row.get('image_id'), row.get('subject_id')]
        
        for cand in candidates:
            if pd.notna(cand):
                s_cand = str(cand)
                # Try exact match
                if s_cand in image_map:
                    return image_map[s_cand]
                # Try with .jpg
                if s_cand.replace('.jpg','') in image_map:
                    return image_map[s_cand.replace('.jpg','')]
                    
        return None

    # Apply mapping
    # We need to ensure we have a column to map from.
    # The previous rename block:
    # if 'dicom_id' in df.columns: rename_map['dicom_id'] = 'filename_base'
    # df.rename...
    
    # So we look at 'filename_base'
    if 'filename_base' in df.columns:
        df['filename'] = df.apply(get_rel_path, axis=1)
    elif 'image_id' in df.columns:
         df['filename'] = df.apply(lambda x: image_map.get(str(x['image_id']), None), axis=1)
    
    # Drop rows where image not found
    len_before = len(df)
    df = df.dropna(subset=['filename'])
    print(f"Matched {len(df)}/{len_before} images.")
    if len(df) == 0:
        print("WARNING: No images matched! Checking first 5 map keys vs ids...")
        print(f"Map keys: {list(image_map.keys())[:5]}")
        print(f"IDs: {df['filename_base'].head().tolist() if 'filename_base' in df.columns else 'N/A'}")
        
    # Standardize columns
    # We need: 'filename', 'indication', 'report'
    
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
