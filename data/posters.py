from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def label_encoding(df):
    genres = sorted(df["primary_genre"].unique())
    label_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    idx_to_label = {idx: genre fpr genre, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label

class PosterDataset(Dataset):
    def __init__(self, df, label_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform
        
        self.image_paths = [Path(p) for p in self.df["img_path"]]
        self.labels = [label_to_idx[g] for g in self.df["primary_genre"]]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, torch.tensor(label)

    def get_splits(df, random_state=42):
        train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['primary_genre'], random_state=random_state)

        train_df, val_df = train_test_split(train_val_df, test_size=1/9, stratify=df['primary_genre'], random_state=random_state)

        print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
        return train_df, val_df, test_df

    def get_temporal_split(df, train_decades, test_decades):
        train_df = df[df['release_decade'].isin(train_decades)]
        test_df = df[df['release_decade'].isin(test_decades)]

        print(f'Train Decades: {train_decades} ({len(train_df)} samples)')')
        print(f'Test Decades: {test_decades} ({len(test_df)} samples)')')

        return train_df, test_df

    def get_dataloader(df, label_to_idx, transform, batch_size=64, shuffle=True, num_workers=4):
        dataset = PosterDataset(df, label_to_idx, transform)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)