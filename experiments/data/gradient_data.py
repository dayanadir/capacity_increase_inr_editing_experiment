import re

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from pathlib import Path

from experiments.data.image_processing import style_edit

from typing import Tuple


class GradientDatasetForEditing(Dataset):
    # todo: support data augmentation
    # todo: can merge with GradientDatasetForClassification, or at least have a common base class
    def __init__(
        self,
        data_dir: str,
        image_data_path: str,
        train: bool = True,
        transform=None,
        target_transform=style_edit("dilate"),
    ):
        """
        Args:
            data_dir (str): Path to the data directory
            transform (callable, optional): Optional transform to be applied on input
            target_transform (callable, optional): Optional transform for targets
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # Support both flat .pth files and nested checkpoint layouts.
        self.grad_paths = sorted(Path(data_dir).rglob("*.pth"))
        # todo: this is MNIST specific
        self.image_dataset = MNIST(
            root=image_data_path,
            download=True,
            train=train,
            transform=None
        )
        self.img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                target_transform,
                transforms.ToTensor(),
            ]
        )

        self.orig_img_transform = transforms.Compose(
            [
                transforms.Lambda(np.array),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.grad_paths)

    @staticmethod
    def _extract_mnist_index(grad_path: Path) -> int:
        # New layout example: mnist_png_training_1_14234/checkpoints/model_final.pth
        # Legacy layout example: mnist_png_training_1_14234_xxx.pth
        candidates = [grad_path.parent.name, grad_path.parent.parent.name, grad_path.stem]
        for candidate in candidates:
            match = re.search(r"(?:training|testing)_\d+_(\d+)", candidate)
            if match:
                return int(match.group(1))
        raise ValueError(f"Could not extract MNIST index from path: {grad_path}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor], int]:
        """
        Args:
            idx (int): Index of the data item
        Returns:
            tuple: (image, label)
        """
        # Load image
        grad_path = self.grad_paths[idx]
        sample = torch.load(grad_path, map_location=lambda storage, loc: storage, weights_only=False)
        if isinstance(sample, dict) and "neuron_features" in sample and "sd_path" in sample:
            # Legacy gradient data format.
            neuron_features = sample["neuron_features"]
            state_dict = torch.load(sample["sd_path"], map_location=lambda storage, loc: storage, weights_only=True)
        else:
            # Checkpoint-only format for non-gradient experiments.
            neuron_features = tuple()
            state_dict = sample

        # todo: this is MNIST specific
        orig_target, _ = self.image_dataset[self._extract_mnist_index(grad_path)]

        # load weights and biases
        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        # Apply transforms
        if self.transform:
            neuron_features = self.transform(neuron_features)

        if self.target_transform:
            target = self.img_transform(orig_target)

        orig_target = self.orig_img_transform(orig_target)

        return neuron_features, weights, biases, target, orig_target
