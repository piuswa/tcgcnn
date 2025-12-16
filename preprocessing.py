import re
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

# clean csv

INPUT = "Yugi_db.csv"
OUTPUT = "Yugi_db_with_categories_v2.csv"

df = pd.read_csv(INPUT)

card_type_col = None
types_col = None
for c in df.columns:
    if c.lower().strip() in ['type', 'card type', 'card-type']:
        card_type_col = c
    if c.lower().strip() in ['types', 'type(s)', 'types ']:
        types_col = c
if card_type_col is None:
    for c in df.columns:
        if 'type' in c.lower():
            card_type_col = c
            break
if types_col is None:
    for c in df.columns:
        if 'type' in c.lower() and c != card_type_col:
            types_col = c
            break

def clean_text(x):
    if pd.isna(x):
        return ''
    s = str(x).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

patterns = [
    ('Link Monster', r'\blink\b'),
    ('Pendulum Monster', r'\bpendulum\b'),
    ('Fusion Monster', r'\bfusion\b'),
    ('Ritual Monster', r'\britual\b'),
    ('Synchro Monster', r'\bsynchro\b'),
    ('Xyz Monster', r'\bxyz\b'),
    ('Effect Monster', r'\beffect\b'),
    ('Normal Monster', r'\bnormal\b'),
]

def classify_row(row):
    card_type_text = clean_text(row.get(card_type_col, ''))
    types_text = clean_text(row.get(types_col, '')) if types_col else ''
    prop_text = clean_text(row.get('Property', '')) if 'Property' in df.columns else ''
    effect_types_text = clean_text(row.get('Effect types', '')) if 'Effect types' in df.columns else ''

    if re.search(r'\bspell\b', card_type_text) or re.search(r'\bspell\b', prop_text):
        return 'Spell'
    if re.search(r'\btrap\b', card_type_text) or re.search(r'\btrap\b', prop_text):
        return 'Trap'

    search_text = ' '.join([types_text, prop_text, effect_types_text])
    for label, pat in patterns:
        if re.search(pat, search_text):
            return label

    return 'Normal Monster'

df['CardCategory'] = df.apply(classify_row, axis=1)
df.to_csv(OUTPUT, index=False)
print(f"Saved updated CSV to: {OUTPUT}")

# resize images

input_folder = "Yugi_images"
output_folder = "Yugi_images_processed"

os.makedirs(output_folder, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((180, 128)),
])
print("Starting loop")
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    try:
        with Image.open(input_path) as img:
            img_resized = transform(img)
            img_resized.save(output_path)
    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("Done")