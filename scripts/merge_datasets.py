#!/usr/bin/env python
"""
Dataset Merger Script

Merges existing car parts datasets into a unified format for YOLO training.

Existing datasets:
- carparts-seg (images/ labels/): 3,833 images, 23 classes
- car-parts-1 (Roboflow): 10,239 images, 50 classes

Total: 14,072 images

Usage:
    cd ml_service
    python scripts/merge_datasets.py
"""

import shutil
from pathlib import Path


class DatasetMerger:
    """Merge multiple datasets into a unified YOLO format."""
    
    def __init__(self, output_dir: str = "datasets/combined_car_parts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        self.train_count = 0
        self.val_count = 0
        
        # Unified classes - combining both datasets
        # carparts-seg classes (0-22) will map to unified 0-12
        # car-parts-1 classes (0-49) will map to unified 13-62
        self.unified_classes = [
            # From carparts-seg (remapped)
            "front_bumper",      # 0
            "rear_bumper",       # 1
            "hood",              # 2
            "trunk",             # 3
            "front_door",        # 4
            "rear_door",         # 5
            "front_light",       # 6
            "rear_light",        # 7
            "front_windshield",  # 8
            "rear_windshield",   # 9
            "wheel",             # 10
            "mirror",            # 11
            "grille",            # 12
        ]
        # Add car-parts-1 classes (keep original numbering, offset by 13)
        for i in range(50):
            self.unified_classes.append(f"part_{i}")
        
        # carparts-seg class remapping
        self.carparts_seg_mapping = {
            0: 1,   # back_bumper -> rear_bumper
            1: 5,   # back_door -> rear_door
            2: 9,   # back_glass -> rear_windshield
            3: 5,   # back_left_door -> rear_door
            4: 7,   # back_left_light -> rear_light
            5: 7,   # back_light -> rear_light
            6: 5,   # back_right_door -> rear_door
            7: 7,   # back_right_light -> rear_light
            8: 0,   # front_bumper
            9: 4,   # front_door
            10: 8,  # front_glass -> front_windshield
            11: 4,  # front_left_door -> front_door
            12: 6,  # front_left_light -> front_light
            13: 6,  # front_light
            14: 4,  # front_right_door -> front_door
            15: 6,  # front_right_light -> front_light
            16: 2,  # hood
            17: 11, # left_mirror -> mirror
            18: -1, # object (skip)
            19: 11, # right_mirror -> mirror
            20: 3,  # tailgate -> trunk
            21: 3,  # trunk
            22: 10, # wheel
        }
        
        # car-parts-1 offset (add 13 to all class IDs)
        self.car_parts_1_offset = 13
    
    def merge_carparts_seg(self, images_dir: str, labels_dir: str):
        """
        Merge carparts-seg dataset.
        
        Structure:
            images/train/  -> 3156 images
            images/val/    -> 401 images
            images/test/   -> 276 images
            labels/train/
            labels/val/
            labels/test/
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        if not images_dir.exists():
            print(f"  Carparts-seg images not found at {images_dir}")
            return 0
        
        print("Merging carparts-seg...")
        count = 0
        
        for split in ['train', 'val', 'test']:
            img_split_dir = images_dir / split
            lbl_split_dir = labels_dir / split
            
            if not img_split_dir.exists():
                continue
            
            # Use existing split (train -> train, val -> val, test -> val)
            target_split = "val" if split in ['val', 'test'] else "train"
            
            for img_file in img_split_dir.glob("*.*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                
                # Copy image
                img_dest = self.output_dir / "images" / target_split / f"carparts_{img_file.name}"
                shutil.copy2(img_file, img_dest)
                
                # Process label
                label_file = lbl_split_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_dest = self.output_dir / "labels" / target_split / f"carparts_{img_file.stem}.txt"
                    self._remap_labels_carparts(label_file, label_dest)
                
                count += 1
                if target_split == "train":
                    self.train_count += 1
                else:
                    self.val_count += 1
        
        print(f"  Added {count} images from carparts-seg")
        return count
    
    def merge_car_parts_1(self, dataset_dir: str):
        """
        Merge car-parts-1 (Roboflow) dataset.
        
        Structure:
            train/images/  -> 6216 images
            train/labels/
            valid/images/  -> 2134 images
            valid/labels/
            test/images/   -> 1889 images
            test/labels/
        """
        dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            print(f"  car-parts-1 not found at {dataset_dir}")
            return 0
        
        print("Merging car-parts-1 (Roboflow)...")
        count = 0
        
        for split in ['train', 'valid', 'test']:
            img_split_dir = dataset_dir / split / "images"
            lbl_split_dir = dataset_dir / split / "labels"
            
            if not img_split_dir.exists():
                continue
            
            # valid/test -> val, train -> train
            target_split = "val" if split in ['valid', 'test'] else "train"
            
            for img_file in img_split_dir.glob("*.*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                
                # Copy image
                img_dest = self.output_dir / "images" / target_split / f"parts1_{img_file.name}"
                shutil.copy2(img_file, img_dest)
                
                # Process label (offset class IDs by 13)
                label_file = lbl_split_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_dest = self.output_dir / "labels" / target_split / f"parts1_{img_file.stem}.txt"
                    self._offset_labels(label_file, label_dest, self.car_parts_1_offset)
                
                count += 1
                if target_split == "train":
                    self.train_count += 1
                else:
                    self.val_count += 1
        
        print(f"  Added {count} images from car-parts-1")
        return count
    
    def _remap_labels_carparts(self, src_file: Path, dest_file: Path):
        """Remap carparts-seg class IDs to unified classes."""
        with open(src_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            if class_id in self.carparts_seg_mapping:
                new_class_id = self.carparts_seg_mapping[class_id]
                if new_class_id >= 0:
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts) + '\n')
        
        with open(dest_file, 'w') as f:
            f.writelines(new_lines)
    
    def _offset_labels(self, src_file: Path, dest_file: Path, offset: int):
        """Add offset to all class IDs in label file."""
        with open(src_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            parts[0] = str(class_id + offset)
            new_lines.append(' '.join(parts) + '\n')
        
        with open(dest_file, 'w') as f:
            f.writelines(new_lines)
    
    def create_data_yaml(self):
        """Create data.yaml for the combined dataset."""
        yaml_content = f"""# Combined Car Parts Dataset
# Generated by merge_datasets.py
# Total classes: {len(self.unified_classes)}

path: ./datasets/combined_car_parts
train: images/train
val: images/val

nc: {len(self.unified_classes)}
names:
"""
        for i, name in enumerate(self.unified_classes):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nCreated: {yaml_path}")
    
    def print_summary(self):
        """Print merge summary."""
        print("\n" + "="*50)
        print("   MERGE COMPLETE")
        print("="*50)
        print(f"  Train images: {self.train_count}")
        print(f"  Val images: {self.val_count}")
        print(f"  Total: {self.train_count + self.val_count}")
        print(f"  Classes: {len(self.unified_classes)}")
        print(f"\nOutput directory: {self.output_dir}")


def main():
    print("="*50)
    print("   DATASET MERGER")
    print("="*50)
    print()
    
    merger = DatasetMerger()
    
    # Merge carparts-seg (images/ and labels/ folders)
    merger.merge_carparts_seg("datasets/images", "datasets/labels")
    
    # Merge car-parts-1 (Roboflow format)
    merger.merge_car_parts_1("datasets/car-parts-1")
    
    # Create data.yaml
    merger.create_data_yaml()
    
    # Print summary
    merger.print_summary()
    
    print("\n" + "="*50)
    print("   NEXT STEPS")
    print("="*50)
    print("  1. Verify dataset: ls datasets/combined_car_parts/images/train")
    print("  2. Train model:")
    print("     python scripts/train_model.py --data datasets/combined_car_parts/data.yaml --epochs 100")


if __name__ == "__main__":
    main()
