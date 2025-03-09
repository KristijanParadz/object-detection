import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2


class ReIDModel:
    def __init__(self, device):
        self.device = device
        # Load a pre-trained ResNet50 model and remove its final classification layer
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # output shape: [B, 2048, 1, 1]
        self.model = nn.Sequential(*(list(base_model.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)
        # Typical pre-processing for ImageNet models; reid models may use different sizes
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # common size for reid
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        """
        Given a BGR image crop (numpy array), return a normalized embedding vector.
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_tensor)  # shape: [1, 2048, 1, 1]
        embedding = embedding.view(
            embedding.size(0), -1)  # flatten to [1, 2048]
        embedding = embedding.cpu().numpy()[0]
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
