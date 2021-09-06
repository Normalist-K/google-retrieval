from torchvision.datasets.folder import default_loader

import glob
import os

import albumentations as A
import numpy as np
import pandas as pd
import torchvision.datasets as datasets


class GLDataset(datasets.VisionDataset):
    def __init__(self, root, split_df_root, fold_no=0, seed=42, split="train", transform=None):
        super().__init__(root, transform=transform)
        assert split in ["train", "val", "test", "index"]
        self.loader = default_loader
        self.split = split
        self.seed = seed

        self.label_map = dict(map(lambda x: (x[1], x[0]), enumerate(pd.read_csv(
            os.path.join(root, "train.csv"))["landmark_id"].unique())))

        if split == "train":
            df = pd.read_csv(os.path.join(
                split_df_root, f"train_df_fold{fold_no}.csv"))
            self.data = list(zip(df["id"], df["landmark_id"].map(self.label_map)))
        elif split == "val":
            df = pd.read_csv(os.path.join(
                split_df_root, f"val_df_fold{fold_no}.csv"))
            self.data = list(zip(df["id"], df["landmark_id"].map(self.label_map)))
        else:
            self.data = glob.glob(os.path.join(root, f"{split}/*/*/*/*.jpg"))

    def __getitem__(self, index):
        if self.split in ["train", "val"]:
            path, target = self.data[index]
            path = os.path.join(self.root, "train",
                                path[0], path[1], path[2], f"{path}.jpg")
        else:
            path = self.data[index]

        img = self.loader(path)
        img = np.array(img)

        if self.transform is not None:
            if type(self.transform) == A.Compose:
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)

        if self.split in ["train", "val"]:
            return img, target
        else:
            return img, path.split('/')[-1].replace(".jpg", "")

    def __len__(self):
        return len(self.data)