import torch
import torch.optim as optim
import pandas as pd
import yaml
import argparse
from data_loader.data_loaders import transform, label_encoding, get_dataloader, get_within_decade_split, get_transfer_split
from models.cnn import PosterCNN
from trainer.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["metadata_path"])

label_to_idx, idx_to_label = label_encoding(df)
num_classes = len(label_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

experiments = [
    ("within_1980s", *get_within_decade_split(df, 1980)),
    ("within_2010s", *get_within_decade_split(df, 2010)),
    ("forward_1980s_1990s", *get_transfer_split(df, 1980, 1990)),
    ("large_gap_1980s_2020s", *get_transfer_split(df, 1980, 2020)),
    ("backward_2010s_1980s", *get_transfer_split(df, 2010, 1980)),
]

results = {}

for exp_name, train_df, val_df, test_df in experiments:
    print(f'\n--- Running experiment: {exp_name} ---')

    train_loader = get_dataloader(train_df, label_to_idx, transform, batch_size=config['training']['batch_size'])
    val_loader = get_dataloader(val_df, label_to_idx, transform, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = get_dataloader(test_df, label_to_idx, transform, batch_size=config['training']['batch_size'], shuffle=False)

    model = PosterCNN(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path=f'checkpoints/{exp_name}_best.pt',
        exp_name=exp_name
    )

    trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])

    model.load_state_dict(torch.load(f"checkpoints/{exp_name}_best.pt"))
    test_f1 = trainer.evaluate(test_loader)

    print(f'Test F1 ({exp_name}): {test_f1:.4f}')

    results[exp_name] = test_f1

print('\n--- Final Results ---')

for exp, f1 in results.items():
    print(f'{exp}: {f1:.4f}')

results_df = pd.DataFrame(results.items(), columns=['experiment', 'test_f1'])
results_df['run'] = config['experiment']['name']
results_df['notes'] = config['experiment']['notes']

# append to master results file
import os
master_path = 'all_results.csv'
if os.path.exists(master_path):
    existing = pd.read_csv(master_path)
    results_df = pd.concat([existing, results_df], ignore_index=True)
results_df.to_csv(master_path, index=False)