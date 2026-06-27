"""
PyTorch Dataset that loads face images from a JSON manifest.
Used by train_vit.py, eval_vit.py, and eval_video_level.py.
"""

import json
from PIL import Image
from torch.utils.data import Dataset


class JsonImageDataset(Dataset):		# reads a json list of {path, label} entries and serves images to the dataloader

    def __init__(self, json_path: str, transform=None, class_to_idx: dict = None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.transform = transform

        # if no mapping is passed (training set), build one from the data
        # for val/test, always pass the training set's mapping so indices stay consistent
        if class_to_idx is None:
            labels = sorted(set(s["label"] for s in self.samples))
            self.class_to_idx = {lab: i for i, lab in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[sample["label"]]
        return img, label
