import os
import pandas as pd
import numpy as np
import librosa
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb 
import argparse
from torchinfo import summary 
from torchlibrosa.augmentation import SpecAugmentation

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_TIME_STEPS = 400
random = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=(2, 4), padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=(2, 2), padding=2, bias=False) 
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flattened_size = 64 
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.flattened_size, 64) 
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes) 

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.dataframe = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.samples = []

        for index, row in self.dataframe.iterrows():
            flac_file = row["flacfile"]
            label = row["buzzlabel"]
            file_path = os.path.join(self.audio_dir, flac_file)
            self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # Resize mel_spec to MAX_TIME_STEPS (width) and N_MELS (height)
            mel_spec = cv2.resize(mel_spec, (MAX_TIME_STEPS, N_MELS), interpolation=cv2.INTER_AREA)
            mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)
            feature_data = torch.from_numpy(mel_spec).float().unsqueeze(0) 
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            feature_data = torch.zeros((1, N_MELS, MAX_TIME_STEPS), dtype=torch.float32)
            label = -1

        return feature_data, torch.tensor(label, dtype=torch.long)

def create_dataloaders(csv_file_path, audio_folder_path, batch_size, random=random):
    full_dataset_no_transform = AudioDataset(csv_file=csv_file_path, audio_dir=audio_folder_path)

    valid_indices = [i for i, (data, label) in enumerate(full_dataset_no_transform) if label != -1]
    labels = [full_dataset_no_transform.samples[i][1] for i in valid_indices]

    train_val_indices, test_indices, _, _ = train_test_split(
        valid_indices, labels, test_size=0.001, stratify=labels, random_state=random
    )
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, [labels[i] for i in train_val_indices], test_size=0.075, stratify=[labels[i] for i in train_val_indices], random_state=random
    )

    train_subset = torch.utils.data.Subset(full_dataset_no_transform, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset_no_transform, val_indices)
    test_subset = torch.utils.data.Subset(full_dataset_no_transform, test_indices)

    print(f"Shape of train_subset (length): {len(train_subset)}")
    print(f"Shape of val_subset (length): {len(val_subset)}")
    print(f"Shape of test_subset (length): {len(test_subset)}")

    train_labels_for_sampler = [full_dataset_no_transform.samples[i][1] for i in train_indices]
    class_counts = np.bincount(train_labels_for_sampler)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_labels_for_sampler]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, class_weights


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, class_weights_tensor, chkpt_path, freq_mask_param, time_mask_param, run_id, early_stopping_patience):
    wandb.watch(model, criterion, log="all", log_freq=100)

    best_val_loss = float('inf')
    patience_counter = 0
    
    spec_augment_transform = SpecAugmentation(
        time_drop_width=time_mask_param, 
        time_stripes_num=2,              
        freq_drop_width=freq_mask_param, 
        freq_stripes_num=2               
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    os.makedirs(chkpt_path, exist_ok=True)
    best_model_path = os.path.join(chkpt_path, run_id + '_best_model.pt')

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            inputs = spec_augment_transform(inputs) 
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                wandb.log({"Train Batch Loss": loss.item()})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train

        model.eval() 
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probabilities.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val

        fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs)
        roc_auc = auc(fpr, tpr)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}, "
              f"Val AUC: {roc_auc:.4f}")

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": epoch_train_loss,
            "Train Accuracy": epoch_train_accuracy,
            "Validation Loss": epoch_val_loss,
            "Validation Accuracy": epoch_val_accuracy,
            "Validation ROC AUC": roc_auc
        })

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[{epoch+1}/{num_epochs}] Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"[{epoch+1}/{num_epochs}] Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break 


    cm = confusion_matrix(all_val_labels, all_val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    wandb.log({"Validation Confusion Matrix": wandb.Image(plt)})
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    wandb.log({"Validation ROC Curve": wandb.Image(plt)})
    plt.close()

def evaluate_model(test_loader, model, criterion):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_test_labels = []
    all_test_preds = [] 
    all_test_probs = [] 

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_probs.extend(probabilities.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct_test / total_test

    fpr, tpr, thresholds = roc_curve(all_test_labels, all_test_probs)
    roc_auc = auc(fpr, tpr)

    
    class_labels = [0, 1] 
    
    print(f"\n--- Test Results ---")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test AUC: {roc_auc:.4f}")

    wandb.log({
        "Test Loss": avg_test_loss,
        "Test Accuracy": test_accuracy,
        "Test ROC AUC": roc_auc,
    })

    metrics_table_data = []
    for i, class_label in enumerate(class_labels):
        if i == 1: 
            precision = precision_score(all_test_labels, all_test_preds, pos_label=class_label, zero_division=0)
            recall = recall_score(all_test_labels, all_test_preds, pos_label=class_label, zero_division=0)
            f1 = f1_score(all_test_labels, all_test_preds, pos_label=class_label, zero_division=0)
            pr_auc = average_precision_score(all_test_labels, all_test_probs) # PR-AUC still uses probabilities

            print(f"\n--- Metrics for Class {class_label} (Positive Class) ---")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall (Sensitivity): {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  PR-AUC: {pr_auc:.4f}")

            wandb.log({
                f"Class {class_label} Precision": precision,
                f"Class {class_label} Recall": recall,
                f"Class {class_label} F1-Score": f1,
                f"Class {class_label} PR-AUC": pr_auc
            })
            metrics_table_data.append([class_label, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", "-", f"{pr_auc:.4f}"])
        else:

            precision_neg = precision_score(all_test_labels, all_test_preds, pos_label=class_label, zero_division=0)
            recall_neg = recall_score(all_test_labels, all_test_preds, pos_label=class_label, zero_division=0)
            f1_neg = f1_score(all_test_labels, all_test_preds, pos_label=class_label, zero_division=0)

            print(f"\n--- Metrics for Class {class_label} (Negative Class) ---")
            print(f"  Precision: {precision_neg:.4f}")
            print(f"  Recall (Specificity): {recall_neg:.4f}")
            print(f"  F1-Score: {f1_neg:.4f}")

            wandb.log({
                f"Class {class_label} Precision": precision_neg,
                f"Class {class_label} Recall": recall_neg,
                f"Class {class_label} F1-Score": f1_neg,
            })
            metrics_table_data.append([class_label, f"{precision_neg:.4f}", f"{recall_neg:.4f}", f"{f1_neg:.4f}", "-", "-"])

    metrics_table = wandb.Table(
        columns=["Class", "Precision", "Recall", "F1-Score", "ROC AUC", "PR AUC"],
        data=metrics_table_data
    )
    wandb.log({"Class-wise Metrics": metrics_table})

    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix (Standard Threshold)") # Updated title
    wandb.log({"Test Confusion Matrix": wandb.Image(plt)})
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    wandb.log({"Test ROC Curve": wandb.Image(plt)})
    plt.close()

def main(args):
    print("code started")

    CSV_FILE = "train.csv"
    AUDIO_FOLDER = "audio"

    run = wandb.init(project="BumbleBuzz CNN", name="with proper site split", config=vars(args)) # Changed run name
    config = wandb.config

    print("wandb project created")

    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(CSV_FILE, AUDIO_FOLDER, config.batch_size, random)

    print("data loaded")

    class_weights_tensor = class_weights.to(DEVICE)
    print(f"Class weights for loss function: {class_weights_tensor}")

    # Create the new SmallCNN model
    model = SmallCNN(num_classes=2, input_channels=1).to(DEVICE)

    print(f"\nCreated SmallCNN Model:")
    print("\nModel Summary:")
    try:
        summary(model, input_size=(1, 1, N_MELS, MAX_TIME_STEPS))
    except Exception as e:
        print(f"Could not generate model summary: {e}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Train the model
    train_model(
        train_loader, val_loader, model, criterion, optimizer,
        config.epochs, class_weights_tensor, config.chkpt_path,
        freq_mask_param=config.spec_augment_freq_mask,
        time_mask_param=config.spec_augment_time_mask,
        run_id=run.id,
        early_stopping_patience=config.early_stopping_patience
    )

    best_model_path = os.path.join(config.chkpt_path, run.id + '_best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nLoaded best model from {best_model_path} for testing.")
        evaluate_model(test_loader, model, criterion)
    else:
        print(f"Warning: Best model not found at {best_model_path}. Evaluation will use the last trained model state.")
        evaluate_model(test_loader, model, criterion)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument("-m", "--momentum", type=float, default=0.0, help="Momentum for optimizer (not used by Adam)")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization) for the optimizer")

    parser.add_argument("--chkpt_path", type=str, default="./checkpoints", help="Path to save model checkpoints")

    parser.add_argument("--spec_augment_freq_mask", type=int, default=15, help="Frequency mask parameter (freq_drop_width) for SpecAugmentation")
    parser.add_argument("--spec_augment_time_mask", type=int, default=30, help="Time mask parameter (time_drop_width) for SpecAugmentation")
    parser.add_argument("--early_stopping_patience", type=int, default=12, help="Number of epochs to wait for improvement before early stopping")
    
    args = parser.parse_args()
    main(args)