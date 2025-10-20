import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple
import glob
from desam.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import torch
from torchvision.transforms.functional import resize, to_pil_image
all_center = ['A-ISBI', 'B-ISBI_1.5', 'C-I2CVB', 'D-UCL', 'E-BIDMC', 'F-HK']

class PretrainDataset(Dataset):

    def __init__(self,
                 root,
                 split,
                 split_file,
                 transforms=None,
                 cp_transforms=None,
                 center_index=0
                 ):
        super().__init__()
        self.data_root = root
        self.transforms = transforms
        self.cp_transforms = cp_transforms
        self.split = split
        self.images = []
        self.masks = []
        self.gts = []
        self.used_patientid = []
        self.split_file = split_file

        if not os.path.exists(self.split_file) and self.split_file.startswith('../../'):
            self.split_file = self.split_file[3:]
        if not os.path.exists(self.data_root) and self.data_root.startswith('../../'):
            self.data_root = self.data_root[3:]

        if self.split_file is not None:
            patient_id = pd.read_csv(self.split_file)

        all_patientid = patient_id[patient_id.center == all_center[center_index]]['patientid'].values.tolist()

        print(all_patientid)
        if self.split == 'train':
            train_patientid = all_patientid[:int(len(all_patientid) * 0.7)]
            label_patientid = train_patientid[:int(len(train_patientid) * 0.2)]
            self.patientid = label_patientid
            print("labeled patient id:", self.patientid)
        elif self.split == 'val':
            used_patientid = all_patientid[int(len(all_patientid) * 0.7):]
            val_patientid = used_patientid[:int(len(used_patientid) * 0.3)]
            self.patientid = val_patientid
        elif self.split == 'test':
            used_patientid = all_patientid[int(len(all_patientid) * 0.7):]
            test_patientid = used_patientid[int(len(used_patientid) * 0.3):]
            self.patientid = test_patientid
        elif self.split == 'out':
            self.patientid = all_patientid

        self.png_paths = [x for x in os.listdir(self.data_root) if int(x[9:12]) in self.patientid]

    def __getitem__(self, idx)-> Tuple[Any, Any]:
        image_name = self.png_paths[idx]
        npz_data = np.load(os.path.join(self.data_root, image_name), allow_pickle=True)
        image = npz_data['imgs'][:, :, 0]
        mask = npz_data['gts']
        mask = cv2.resize(np.uint8(mask), (image.shape[0], image.shape[1]))
        mask[np.where(mask > 1)] = 1
        gt = 1

        assert image.shape[0] == mask.shape[0]
        assert len(image.shape) == len(mask.shape)

        self.images.append(image)
        self.masks.append(mask)
        self.gts.append(gt)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        if self.cp_transforms is not None:
            cp_transformed = self.cp_transforms({'image': image, 'mask': mask, })
            image, mask = cp_transformed["image"], cp_transformed["mask"]
        image = np.expand_dims(image, axis=0)

        return image, mask, np.array([gt])

    def __len__(self) -> int:
        return len(self.png_paths)

class TrainProstateDataset(Dataset):

    def __init__(self,
                 root,
                 split,
                 split_file,
                 transforms=None,
                 cp_transforms=None,
                 center_index=1
                 ):
        super().__init__()
        self.data_root = root
        self.transforms = transforms
        self.cp_transforms = cp_transforms
        self.split = split
        self.images = []
        self.masks = []
        self.gts = []
        self.used_patientid = []
        self.unlabel_patientid = []
        self.split_file = split_file
        self.sam_transform = ResizeLongestSide(256)

        if not os.path.exists(self.split_file) and self.split_file.startswith('../../'):
            self.split_file = self.split_file[3:]
        if not os.path.exists(self.data_root) and self.data_root.startswith('../../'):
            self.data_root = self.data_root[3:]

        if self.split_file is not None:
            patient_id = pd.read_csv(self.split_file)

        all_patientid = patient_id[patient_id.center == all_center[center_index]]['patientid'].values.tolist()
        if self.split == 'train':
            train_patientid = all_patientid[:int(len(all_patientid) * 0.7)]
            self.label_patientid = train_patientid[:int(len(train_patientid) * 0.2)]
            self.unlabel_patientid = train_patientid[int(len(train_patientid) * 0.2):]
            self.patientid = train_patientid
        elif self.split == 'val':
            used_patientid = all_patientid[int(len(all_patientid) * 0.7):]
            val_patientid = used_patientid[:int(len(used_patientid) * 0.3)]
            self.patientid = val_patientid
        elif self.split == 'test':
            used_patientid = all_patientid[int(len(all_patientid) * 0.7):]
            test_patientid = used_patientid[int(len(used_patientid) * 0.3):]
            self.patientid = test_patientid
        elif self.split == 'pseudo':
            train_patientid = all_patientid[:int(len(all_patientid) * 0.7)]
            unlabel_patientid = train_patientid[int(len(train_patientid) * 0.2):]
            self.patientid = unlabel_patientid

        self.png_paths = sorted([x for x in os.listdir(self.data_root) if int(x[9:12]) in self.patientid])

    def __getitem__(self, idx)-> Tuple[Any, Any]:
        image_name = self.png_paths[idx]
        npz_data = np.load(os.path.join(self.data_root, image_name), allow_pickle=True)
        image = npz_data['imgs'][:, :, 0]
        mask = npz_data['gts']
        mask = cv2.resize(np.uint8(mask), (image.shape[0], image.shape[1]))
        mask[np.where(mask > 1)] = 1
        if int(image_name[9:12]) in self.unlabel_patientid:
            gt = 0
        else:
            gt = 1

        assert image.shape[0] == mask.shape[0]
        assert len(image.shape) == len(mask.shape)

        self.images.append(image)
        self.masks.append(mask)
        self.gts.append(gt)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if self.cp_transforms is not None:
            cp_transformed = self.cp_transforms({'image': image, 'mask': mask, })
            image, mask = cp_transformed["image"], cp_transformed["mask"]
        img = np.repeat(image[None, :, :], 3, axis=0)

        image = np.expand_dims(image, axis=0)

        return image, mask, np.array([gt]), img,  image_name

    def __len__(self) -> int:
        return len(self.png_paths)

