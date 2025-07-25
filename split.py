import os, random, shutil

input_folder = 'images'        # Folder where all your .jpg and .xml files are
output_base = 'split-data'        # Output root folder

# Create output directories
for split in ['train', 'test', 'val']:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# Get all .jpg files (each must have a matching .xml)
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
random.shuffle(image_files)

# Calculate split indices
total = len(image_files)
train_end = int(0.7 * total)
test_end = train_end + int(0.2 * total)

train_files = image_files[:train_end]
test_files = image_files[train_end:test_end]
val_files = image_files[test_end:]

# Function to copy .jpg and .xml files
def copy_pair(files, target_dir):
    for file in files:
        base = os.path.splitext(file)[0]
        jpg_src = os.path.join(input_folder, f"{base}.jpg")
        xml_src = os.path.join(input_folder, f"{base}.xml")

        jpg_dst = os.path.join(output_base, target_dir, f"{base}.jpg")
        xml_dst = os.path.join(output_base, target_dir, f"{base}.xml")

        if os.path.exists(jpg_src) and os.path.exists(xml_src):
            shutil.copy(jpg_src, jpg_dst)
            shutil.copy(xml_src, xml_dst)
        else:
            print(f"Skipping {base}: missing .jpg or .xml")

# Copy to each split folder
copy_pair(train_files, 'train')
copy_pair(test_files, 'test')
copy_pair(val_files, 'val')
