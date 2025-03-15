import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Any
from numpy.typing import NDArray


class ReIDModel(nn.Module):
    """
    ReIDModel encapsulates a feature extractor based on ResNet50,
    converting input images to normalized feature embeddings.
    """

    def __init__(self, device: torch.device) -> None:
        super(ReIDModel, self).__init__()
        self.device = device
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            *(list(base_model.children())[:-1]))
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_embedding(self, image_bgr: NDArray[Any]) -> NDArray[Any]:
        """
        Converts a BGR image to a normalized feature embedding.

        Args:
            image_bgr: A BGR image as a NumPy array.

        Returns:
            A 1D normalized embedding as a NumPy array.
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError(
                "Invalid input: image_bgr must be a valid NumPy array.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.feature_extractor(inp)

        feat = feat.view(feat.size(0), -1).cpu().numpy()[0]
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm

        return feat
