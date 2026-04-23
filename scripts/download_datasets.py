#!/usr/bin/env python
"""
Car Parts Dataset Download Script

This script downloads and prepares multiple car parts datasets for training.
Datasets included:
- CompCars (27,618 car parts images)
- Carparts-seg (3,833 images)
- Stanford Cars (16,185 images)
- Roboflow Car Parts (optional)

Usage:
    python scripts/download_datasets.py
"""

import subprocess
import zipfile
from pathlib import Path

# Dataset configuration
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)


def run_command(cmd, cwd=None):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def download_compcars():
    """Download CompCars dataset from Kaggle."""
    print("\n" + "="*50)
    print("Downloading CompCars Dataset...")
    print("="*50)
    
    compcars_dir = DATASETS_DIR / "compcars"
    if compcars_dir.exists():
        print("CompCars already exists. Skipping...")
        return True
    
    # Download from Kaggle
    cmd = "kaggle datasets download -d renancostaalencar/compcars"
    if not run_command(cmd, cwd=str(DATASETS_DIR)):
        print("Failed to download CompCars. Make sure Kaggle CLI is configured.")
        return False
    
    # Extract
    zip_path = DATASETS_DIR / "compcars.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(compcars_dir)
        zip_path.unlink()
        print(f"Extracted CompCars to {compcars_dir}")
    
    return True


def download_carparts_seg():
    """Download Carparts-seg dataset from Ultralytics."""
    print("\n" + "="*50)
    print("Downloading Carparts-seg Dataset...")
    print("="*50)
    
    carparts_dir = DATASETS_DIR / "carparts-seg"
    if carparts_dir.exists():
        print("Carparts-seg already exists. Skipping...")
        return True
    
    # Use Ultralytics to download
    try:
        import urllib.request
        
        # Direct download URL
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
        zip_path = DATASETS_DIR / "carparts-seg.zip"
        
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(carparts_dir)
        zip_path.unlink()
        print(f"Extracted Carparts-seg to {carparts_dir}")
        
        return True
    except Exception as e:
        print(f"Error downloading Carparts-seg: {e}")
        return False


def download_stanford_cars():
    """Download Stanford Cars dataset from Kaggle."""
    print("\n" + "="*50)
    print("Downloading Stanford Cars Dataset...")
    print("="*50)
    
    stanford_dir = DATASETS_DIR / "stanford_cars"
    if stanford_dir.exists():
        print("Stanford Cars already exists. Skipping...")
        return True
    
    # Download from Kaggle
    cmd = "kaggle datasets download -d eduardo4jesus/stanford-cars-dataset"
    if not run_command(cmd, cwd=str(DATASETS_DIR)):
        print("Failed to download Stanford Cars.")
        return False
    
    # Extract
    zip_path = DATASETS_DIR / "stanford-cars-dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(stanford_dir)
        zip_path.unlink()
        print(f"Extracted Stanford Cars to {stanford_dir}")
    
    return True


def download_roboflow_carparts():
    """
    Download Roboflow car parts datasets.
    Note: Requires manual download from Roboflow Universe.
    """
    print("\n" + "="*50)
    print("Roboflow Datasets (Manual Download Required)")
    print("="*50)
    
    roboflow_dir = DATASETS_DIR / "roboflow_carparts"
    roboflow_dir.mkdir(exist_ok=True)
    
    print("""
    Please manually download from Roboflow Universe:
    
    1. Create account at https://app.roboflow.com
    2. Go to these URLs and fork the datasets:
    
       - https://universe.roboflow.com/motork/car-parts-yolov8-unir
       - https://universe.roboflow.com/team-data/car-parts-ybiev
       - https://universe.roboflow.com/car-oil/motor-oil
    
    3. Export each as YOLOv8 format
    4. Extract to: datasets/roboflow_carparts/
    
    Directory structure:
    datasets/roboflow_carparts/
    ├── car_parts_motork/
    ├── car_parts_team/
    └── motor_oil/
    """)
    
    return True


def create_combined_dataset():
    """Create combined dataset configuration."""
    print("\n" + "="*50)
    print("Creating Combined Dataset Configuration...")
    print("="*50)
    
    combined_dir = DATASETS_DIR / "combined_car_parts"
    combined_dir.mkdir(exist_ok=True)
    
    # Create images and labels directories
    (combined_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (combined_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (combined_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (combined_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    yaml_content = """# Combined Car Parts Dataset Configuration
# Generated by download_datasets.py

# Dataset paths
path: ./datasets/combined_car_parts
train: images/train
val: images/val

# Classes
names:
  # Car body parts
  0: front_bumper
  1: rear_bumper
  2: hood
  3: trunk
  4: front_door
  5: rear_door
  6: front_light
  7: rear_light
  8: front_windshield
  9: rear_windshield
  10: wheel
  11: mirror
  12: grille
  
  # Auto parts products
  13: engine_oil
  14: oil_filter
  15: air_filter
  16: brake_pad
  17: brake_disc
  18: spark_plug
  19: battery
  20: tire
  
  # General categories
  21: car_part
  22: car
"""
    
    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created configuration at {yaml_path}")
    print("\nNote: You need to merge the datasets manually or use the merge script.")
    
    return True


def main():
    """Main download function."""
    print("="*60)
    print("   CAR PARTS DATASET DOWNLOAD SCRIPT")
    print("="*60)
    
    print("\nThis script will download the following datasets:")
    print("  1. CompCars (27,618 images)")
    print("  2. Carparts-seg (3,833 images)")
    print("  3. Stanford Cars (16,185 images)")
    print("  4. Roboflow datasets (manual)")
    print("\nTotal: ~47,000+ images")
    
    input("\nPress Enter to start downloading...")
    
    # Download datasets
    results = {
        "CompCars": download_compcars(),
        "Carparts-seg": download_carparts_seg(),
        "Stanford Cars": download_stanford_cars(),
        "Roboflow": download_roboflow_carparts(),
    }
    
    # Create combined config
    create_combined_dataset()
    
    # Summary
    print("\n" + "="*60)
    print("   DOWNLOAD SUMMARY")
    print("="*60)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {dataset}: {status}")
    
    print("\nNext steps:")
    print("  1. Check datasets/ directory for downloaded files")
    print("  2. Download Roboflow datasets manually (see instructions above)")
    print("  3. Run merge script to combine datasets (optional)")
    print("  4. Train with: python scripts/train_model.py")


if __name__ == "__main__":
    main()
