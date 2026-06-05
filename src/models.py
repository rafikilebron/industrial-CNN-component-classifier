# Deep Learning
import torch
import torch.nn as nn

# CNN Architectures
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        # Load architecture
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Freeze parameters (feature extractor)
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace output layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, n_classes)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        # Set model to evaluation mode
        self.model.eval()

        # Predict
        with torch.no_grad():
            outputs     = self.model(x)
            prediction  = torch.argmax(outputs, 1).item()

        return prediction