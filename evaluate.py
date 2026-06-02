import torch
import pandas as pd
import yaml
import argparse
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader.data_loaders import label_encoding, get_dataloader, get_within_decade_split, get_transfer_split, transform
from models.cnn import PosterCNN

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["metadata_path"])
label_to_idx, idx_to_label = label_encoding(df)
num_classes = len(label_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_name = config["experiment"]["name"]
os.makedirs(f"results/{run_name}", exist_ok=True)

experiments = {
    "within_1980s": get_within_decade_split(df, 1980),
    "within_2010s": get_within_decade_split(df, 2010),
    "forward_1980s_1990s": get_transfer_split(df, 1980, 1990),
    "large_gap_1980s_2020s": get_transfer_split(df, 1980, 2020),
    "backward_2010s_1980s": get_transfer_split(df, 2010, 1980),
}

for exp_name,(train_df, val_df, test_df) in experiments.items():
    print(f'\n--- {exp_name} ---')

    model = PosterCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(f'checkpoints/{exp_name}_best.pt', weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = get_dataloader(test_df, label_to_idx, transform, batch_size=64, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.append(predictions)
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()

    report = classification_report(all_labels, all_predictions, target_names=list(label_to_idx.keys()), zero_division=0)
    print(report)

    with open(f"results/{run_name}/{exp_name}_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, 
                annot=True,
                fmt='d',
                xticklabels=label_to_idx.keys(),
                yticklabels=label_to_idx.keys()
    )
    plt.title(f'Confusion Matrix: {exp_name}')
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"results/{run_name}/{exp_name}_confusion.png", dpi=150)
    plt.close()

    print(f"Saved results to results/{run_name}/")