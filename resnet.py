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
import torchaudio.transforms as T
import argparse
from torchinfo import summary 

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_TIME_STEPS = 400
random = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# https://gitlab.imt-atlantique.fr/m20leona/explore/-/blob/master/models/resnet.py?ref_type=heads
class ResNetBlock(nn.Module):
    def __init__(self, ifm, ofm, stride=1, groups=[1,1]):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(ifm, ofm, kernel_size=3, stride=stride, padding=1, groups=groups[0], bias=False)
        self.bn1 = nn.BatchNorm2d(ofm)
        self.conv2 = nn.Conv2d(ofm, ofm, kernel_size=3, stride=1, padding=1, groups=groups[1], bias=False)
        self.bn2 = nn.BatchNorm2d(ofm)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or ifm != ofm:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ifm, ofm, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ofm)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=2, input_channels=1):
        super(ResNet, self).__init__()
        self.ifm = blocks[0][0]
        
        # modified to accept mel-spectrogram
        self.conv1 = nn.Conv2d(input_channels, self.ifm, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ifm)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        blocks_array = []
        previous_fmaps = blocks[0][0]
        
        for i, (fmaps, stride, groups) in enumerate(blocks):
            if i == 0:
                blocks_array.append(ResNetBlock(previous_fmaps, fmaps, stride=1, groups=groups))
            else:
                blocks_array.append(ResNetBlock(previous_fmaps, fmaps, stride, groups))
            previous_fmaps = fmaps
        
        self.blocks = nn.ModuleList(blocks_array)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(blocks[-1][0], num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Initial convolution and pooling
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        # ResNet blocks
        for block in self.blocks:
            out = block(out)
        
        # Global average pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        # Classifier
        out = self.dropout(out)
        out = self.linear(out)
        return out

def create_resnet_model(depth=18, num_classes=2):
    if depth == 8:

        blocks = [
            (64, 1, [1, 1]),   
            (128, 2, [1, 1]),  
            (256, 2, [1, 1]), 
            (512, 2, [1, 1]),  
        ]
    elif depth == 14:
        blocks = [
            (64, 1, [1, 1]),   
            (64, 1, [1, 1]),   
            (128, 2, [1, 1]),  
            (128, 1, [1, 1]),  
            (256, 2, [1, 1]),  
            (256, 1, [1, 1]),  
            (512, 2, [1, 1]),  
        ]
    elif depth == 18:
        blocks = [
            (64, 1, [1, 1]),  
            (64, 1, [1, 1]),   
            (128, 2, [1, 1]),  
            (128, 1, [1, 1]),  
            (256, 2, [1, 1]), 
            (256, 1, [1, 1]), 
            (512, 2, [1, 1]), 
            (512, 1, [1, 1]), 
        ]
    else:
        # Default ResNet-18 
        blocks = [
            (64, 1, [1, 1]),
            (64, 1, [1, 1]),
            (128, 2, [1, 1]),
            (128, 1, [1, 1]),
            (256, 2, [1, 1]),
            (256, 1, [1, 1]),
            (512, 2, [1, 1]),
            (512, 1, [1, 1]),
        ]
    
    return ResNet(blocks, num_classes=num_classes, input_channels=1)

# dataset class
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

#specaugment class
#https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html
#https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.FrequencyMasking.html
class SpecAugment:
    def __init__(self, freq_mask_param, time_mask_param):
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)

    def __call__(self, spec):
        spec = self.freq_masking(spec)
        spec = self.time_masking(spec)
        return spec

#dataloader class
def create_dataloaders(csv_file_path, audio_folder_path, batch_size, random=random):
    full_dataset_no_transform = AudioDataset(csv_file=csv_file_path, audio_dir=audio_folder_path)

    valid_indices = [i for i, (data, label) in enumerate(full_dataset_no_transform) if label != -1]
    labels = [full_dataset_no_transform.samples[i][1] for i in valid_indices]

    train_val_indices, test_indices, _, _ = train_test_split(
        valid_indices, labels, test_size=0.35, stratify=labels, random_state=random
    )
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, [labels[i] for i in train_val_indices], test_size=0.075, stratify=[labels[i] for i in train_val_indices], random_state=random
    )

    train_subset = torch.utils.data.Subset(full_dataset_no_transform, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset_no_transform, val_indices)
    test_subset = torch.utils.data.Subset(full_dataset_no_transform, test_indices)

    train_labels_for_sampler = [full_dataset_no_transform.samples[i][1] for i in train_indices]
    class_counts = np.bincount(train_labels_for_sampler)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_labels_for_sampler]

    #https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, class_weights


#training loop
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, class_weights_tensor, chkpt_path, freq_mask_param, time_mask_param, run_id, early_stopping_patience):
    wandb.watch(model, criterion, log="all", log_freq=100)

    best_val_loss = float('inf')
    patience_counter = 0
    spec_augment_transform = SpecAugment(freq_mask_param=freq_mask_param, time_mask_param=time_mask_param)
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

#evalution function
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

    youden_j = tpr - fpr
    optimal_threshold_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_threshold_idx]

    optimal_threshold_preds = (np.array(all_test_probs) >= optimal_threshold).astype(int)

    class_labels = [0, 1] 
    
    print(f"\n--- Test Results ---")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test AUC: {roc_auc:.4f}")
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")

    wandb.log({
        "Test Loss": avg_test_loss,
        "Test Accuracy": test_accuracy,
        "Test ROC AUC": roc_auc,
        "Optimal Threshold": optimal_threshold
    })

    metrics_table_data = []
    for i, class_label in enumerate(class_labels):
        if i == 1:
            precision = precision_score(all_test_labels, optimal_threshold_preds, pos_label=class_label, zero_division=0)
            recall = recall_score(all_test_labels, optimal_threshold_preds, pos_label=class_label, zero_division=0)
            f1 = f1_score(all_test_labels, optimal_threshold_preds, pos_label=class_label, zero_division=0)
            pr_auc = average_precision_score(all_test_labels, all_test_probs) # PR-AUC is typically for the positive class

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
            precision_neg = precision_score(all_test_labels, optimal_threshold_preds, pos_label=class_label, zero_division=0)
            recall_neg = recall_score(all_test_labels, optimal_threshold_preds, pos_label=class_label, zero_division=0)
            f1_neg = f1_score(all_test_labels, optimal_threshold_preds, pos_label=class_label, zero_division=0)

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


    cm = confusion_matrix(all_test_labels, optimal_threshold_preds) # Use optimal threshold predictions for CM
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix (Optimal Threshold: {optimal_threshold:.2f})")
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

    CSV_FILE = "merged_wo_false.csv"
    AUDIO_FOLDER = "audio"

    run = wandb.init(project="BumbleBuzz ResNet",name="resnet_18", config=vars(args))
    config = wandb.config

    print("wandb project created")

    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(CSV_FILE, AUDIO_FOLDER, config.batch_size, random)

    print("data loaded")

    class_weights_tensor = class_weights.to(DEVICE)
    print(f"Class weights for loss function: {class_weights_tensor}")

    # Create ResNet model
    model = create_resnet_model(depth=config.resnet_depth, num_classes=2).to(DEVICE)

    print(f"\nCreated ResNet-{config.resnet_depth} Model:")
    print("\nModel Summary:")
    try:
        summary(model, input_size=(1, 1, N_MELS, MAX_TIME_STEPS))
    except:
        print("Could not generate model summary")

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

    parser.add_argument("--spec_augment_freq_mask", type=int, default=15, help="Frequency mask parameter for SpecAugment")
    parser.add_argument("--spec_augment_time_mask", type=int, default=30, help="Time mask parameter for SpecAugment")
    parser.add_argument("--early_stopping_patience", type=int, default=12, help="Number of epochs to wait for improvement before early stopping")
    
    # ResNet specific arguments
    parser.add_argument("--resnet_depth", type=int, default=18, choices=[8, 14, 18], help="ResNet depth (8, 14, or 18)")

    args = parser.parse_args()
    main(args)