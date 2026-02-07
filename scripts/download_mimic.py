import kagglehub
import shutil
import os
from pathlib import Path

def main():
    print("Downloading MIMIC-CXR dataset using kagglehub...")
    # Download dataset
    try:
        path = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")
        print("Dataset downloaded to:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Define target directory
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / 'data' / 'raw' / 'mimic_cxr'
    
    print(f"Target directory: {target_dir}")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files
    print("Moving files to target directory...")
    source_path = Path(path)
    
    # Iterate over files in the downloaded path and move them
    for item in source_path.iterdir():
        dest = target_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                os.remove(dest)
        
        print(f"Moving {item} to {dest}")
        shutil.move(str(item), str(dest))
        
    print("✓ MIMIC-CXR dataset successfully placed in data/raw/mimic_cxr")

if __name__ == "__main__":
    main()
