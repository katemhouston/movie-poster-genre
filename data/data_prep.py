import os
import argparse
import yaml
import pandas as pd
from datasets import load_dataset
from PIL import Image

parser = argparse.ArgumentParser(description="Download and prepare movie poster dataset")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
   config = yaml.safe_load(f)

save_dir = config["data"]["save_dir"]
metadata_path = config["data"]["metadata_path"]
img_size = config["data"]["img_size"]
quality = config["data"]["quality"]

os.makedirs(save_dir, exist_ok=True)

print("Loading dataset...")

ds = load_dataset("skvarre/movie_posters-100k", streaming=True, split='train')

def clean_example(example):
   genres = example.get("genres")
   release_date = example.get("release_date")

   if not genres or not release_date:
       return None

   year = int(release_date[:4])

   return {
       "primary_genre": genres[0]["name"],
       "year": year,
       "decade": (year//10) * 10
   }

rows = []
saved_idx = 0
skipped = 0

print("Starting cleaning loop...")

for i, example in enumerate(ds):
   cleaned_info = clean_example(example)

   if cleaned_info is None:
       skipped += 1
       continue

   img_path = os.path.join(save_dir, f'{saved_idx}.jpg')

   if not os.path.exists(img_path):
       img = example['image'].convert("RGB").resize((img_size, img_size))
       img.save(img_path, quality=quality)

   rows.append({
       "idx": saved_idx,
       "name": example.get("title"),
       "primary_genre": cleaned_info["primary_genre"],
       "release_year": cleaned_info["year"],
       "release_decade": cleaned_info["decade"],
       "img_path": img_path
   })

   saved_idx += 1

   if saved_idx % 1000 == 0:
       pd.DataFrame(rows).to_csv(metadata_path, index=False)
       print(f'Saved {saved_idx} examples')

df = pd.DataFrame(rows)
df.to_csv(metadata_path, index=False)
