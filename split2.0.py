import os, shutil
import random

# Input full dataset
all_images = 'images'
all_annots = 'annotations'

# Output folders
splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test

base_out = 'split_data'
for s in splits:
    os.makedirs(os.path.join(base_out, s, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_out, s, 'annotations'), exist_ok=True)

# Matching base filenames only
all_files = [f[:-4] for f in os.listdir(all_images) if f.endswith('.jpg')]
random.shuffle(all_files)

# Split into train/val/test
train_end = int(len(all_files) * split_ratio[0])
val_end = train_end + int(len(all_files) * split_ratio[1])

file_splits = {
    'train': all_files[:train_end],
    'val': all_files[train_end:val_end],
    'test': all_files[val_end:]
}

# Move files
for split, files in file_splits.items():
    for base in files:
        shutil.copy(os.path.join(all_images, base + '.jpg'), os.path.join(base_out, split, 'images', base + '.jpg'))
        shutil.copy(os.path.join(all_annots, base + '.xml'), os.path.join(base_out, split, 'annotations', base + '.xml'))

print("✅ Dataset split into train/val/test.")
