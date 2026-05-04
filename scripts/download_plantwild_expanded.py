import json
import zipfile
import io
from pathlib import Path
import requests

# Load label map
label_map = json.load(open('models/label_map.json', encoding='utf-8'))
valid_labels = list(label_map['label_to_id'].keys())
print(f'All valid labels ({len(valid_labels)}):')
for label in valid_labels:
    print(f'  - {label}')

# Download ZIP with streaming
url = 'https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip'
print(f'\nDownloading PlantWild ZIP for {len(valid_labels)} labels...')
print('This will take several minutes (approx 2.7GB)...')

response = requests.get(url, timeout=600)

# Create output directory
output_dir = Path('data/controlled_feedback_dataset_expanded')
if output_dir.exists():
    import shutil
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Extract up to 5 images per label
images_per_label = {label: 0 for label in valid_labels}
max_per_label = 5

total_extracted = 0
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_info in zip_ref.infolist():
        if not file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        for label in valid_labels:
            # PlantWild path format: plantwild/images/{label}/{image}.jpg
            if f'/images/{label}/' in file_info.filename:
                if images_per_label[label] < max_per_label:
                    zip_ref.extract(file_info, output_dir)
                    images_per_label[label] += 1
                    total_extracted += 1
                    if total_extracted % 10 == 0:
                        print(f'  Extracted {total_extracted} images...')
                break
        
        # Stop when all labels have enough images
        if all(count >= max_per_label for count in images_per_label.values()):
            break

# Reorganize structure: move from plantwild/images/label/ to output_dir/label/
plantwild_images = output_dir / 'plantwild' / 'images'
if plantwild_images.exists():
    for label_dir in plantwild_images.iterdir():
        if label_dir.is_dir():
            target = output_dir / label_dir.name
            if target.exists():
                import shutil
                shutil.rmtree(target)
            label_dir.rename(target)
    # Remove empty plantwild folder
    shutil.rmtree(output_dir / 'plantwild')

print(f'\nExtraction complete: {total_extracted} images total')
print('Images per label:')
for label in valid_labels:
    count = images_per_label[label]
    print(f'  {label}: {count}')

# Verify final structure
print(f'\nFinal structure in {output_dir}:')
for item in sorted(output_dir.iterdir()):
    if item.is_dir():
        img_count = len(list(item.rglob('*.jpg')) + list(item.rglob('*.jpeg')) + list(item.rglob('*.png')))
        print(f'  {item.name}/: {img_count} images')
