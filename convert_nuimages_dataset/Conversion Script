import os
from nuimages import NuImages
import shutil

def collect_all_classes(dataroot):
    splits = ['train', 'val']
    all_class_names = set()
    for split in splits:
        nuim = NuImages(version=f'v1.0-{split}', dataroot=dataroot, verbose=False)
        categories = nuim.category
        class_names = [cat['name'] for cat in categories]
        all_class_names.update(class_names)
    return sorted(all_class_names)

def convert_nuimages_to_yolo(nuim, split, output_dir, class_to_id):
    images_dir = os.path.join(output_dir, 'images', split)
    labels_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for image in nuim.image:
        img_token = image['token']
        filepath = image['filepath']
        img_width = image['width']
        img_height = image['height']
        img_filename = os.path.basename(filepath)
        
        # Copy image
        src_path = os.path.join(nuim.dataroot, filepath)
        dest_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)  # Use copy instead of symlink

        # Process annotations
        annotations = nuim.get_annotations(img_token)
        label_content = []
        for ann in annotations:
            category = nuim.get('category', ann['category_token'])
            class_name = category['name']
            if class_name not in class_to_id:
                continue
            class_id = class_to_id[class_name]

            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            label_content.append(label_line)
        
        # Write label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_content))

    print(f"Converted {split} split with {len(nuim.image)} images.")

# Main Execution
output_dir = 'path/to/yolo_dataset'
dataroot = 'path/to/nuimages/dataset'

# Collect all classes
all_class_names = collect_all_classes(dataroot)
class_to_id = {name: idx for idx, name in enumerate(all_class_names)}

# Save classes.txt
with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
    for name in all_class_names:
        f.write(f"{name}\n")

# Process each split
for split in ['train', 'val']:
    nuim = NuImages(version=f'v1.0-{split}', dataroot=dataroot, verbose=False)
    convert_nuimages_to_yolo(nuim, split, output_dir, class_to_id)
