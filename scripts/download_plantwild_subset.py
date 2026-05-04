import json
import zipfile
import io
from pathlib import Path
import requests

# Load label map
label_map = json.load(open('models/label_map.json', encoding='utf-8'))
valid_labels = set(label_map['label_to_id'].keys())

# Select 3 labels for small simulation
selected_labels = ['tomato late blight', 'apple rust', 'citrus canker']
print(f'Selected labels: {selected_labels}')

# Download ZIP with streaming
url = 'https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip'
print('Downloading ZIP (this may take a while)...')

response = requests.get(url, timeout=600)

# Create output directory
output_dir = Path('data/controlled_feedback_dataset')
output_dir.mkdir(parents=True, exist_ok=True)

# Extract only selected folders (limit to 5 images per label)
images_per_label = {label: 0 for label in selected_labels}
max_per_label = 5

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_info in zip_ref.infolist():
        # Check if file is in selected label folders
        for label in selected_labels:
            if f'/{label}/' in file_info.filename and file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if images_per_label[label] < max_per_label:
                    zip_ref.extract(file_info, output_dir)
                    print(f'Extracted: {file_info.filename}')
                    images_per_label[label] += 1
                    break

total_images = sum(images_per_label.values())
print(f'Extracted {total_images} images total:')
for label, count in images_per_label.items():
    print(f'  {label}: {count}')
