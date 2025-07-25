import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, HfApi
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class SmallCNN(nn.Module, PyTorchModelHubMixin):
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
        self.num_classes = num_classes
        self.input_channels = input_channels

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

try:
    dummy_state_dict = torch.load('model2.pt', map_location='cpu')
    print("Successfully loaded")
except FileNotFoundError:
    print("Error")

model_to_upload = SmallCNN(num_classes=2, input_channels=1)
model_to_upload.load_state_dict(dummy_state_dict)
model_to_upload.to(DEVICE) 
model_to_upload.eval()

repo_id = "purvalb/BumbleBuzz-Buzz-Detector"

print(f"\nAttempting to push model to: {repo_id}")


model_to_upload.push_to_hub(
        repo_id,
        commit_message="Initial upload of BumbleBuzz Buzz Detector"
)
print(f"Model '{repo_id}' successfully uploaded!")