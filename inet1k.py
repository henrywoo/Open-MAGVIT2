from io import BytesIO
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset, DatasetDict
from PIL import Image
import numpy as np


DS_PATH_IMAGENET1K = "imagenet-1k"
IN_MEAN = np.array([0.485, 0.456, 0.406])
IN_STD = np.array([0.229, 0.224, 0.225])

class ImageLabelDataSet(Dataset):
    def __init__(self, dataset, transform=None, return_type='dict', split='train', image_size=224, convert_rgb=True,
                 img_key=None):
        if isinstance(dataset, DatasetDict) and (split is None or split in dataset):
            split = split or "train"
            self.dataset = dataset[split]
        else:
            self.dataset = dataset
        self.transform = transform
        self.return_type = return_type
        self.image_size = image_size
        if img_key is None:
            self.img_key = 'image' if 'image' in self.dataset.column_names else 'img'
        else:
            self.img_key = img_key
        self.label_key = 'label' if 'label' in self.dataset.column_names else 'lbl'
        self.convert_rgb = convert_rgb
        if self.label_key not in self.dataset.column_names:
            self.label_key = None
        if transform is None:
            if isinstance(self.image_size, int):
                self.resize_transform = transforms.Resize((image_size, image_size))
            self.to_tensor_transform = transforms.ToTensor()
            self.normalize_transform = transforms.Normalize(mean=IN_MEAN, std=IN_STD)
        else:
            # Check if self.transform contains a resize operation
            contains_resize = False
            if self.transform:
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        contains_resize = True
                        break
            if not contains_resize and image_size is not None:
                # Check the size of the first image in the dataset
                first_image = self.dataset[self.img_key][0]
                if isinstance(first_image, torch.Tensor):
                    first_image = transforms.ToPILImage()(first_image)
                first_image_size = first_image.size  # (width, height)

                # Add a resize transform if image size does not match
                if first_image_size != (self.image_size, self.image_size):
                    resize_transform = transforms.Resize((self.image_size, self.image_size))
                    if self.transform:
                        # Insert the resize transform at the beginning
                        self.transform.transforms.append(resize_transform)
                    else:
                        self.resize_transform = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.img_key]
        if self.convert_rgb and img.mode != 'RGB':
            img = img.convert('RGB')
        with BytesIO() as output:
            img.save(output, format='JPEG')
            img_bytes = output.getvalue()
        img = Image.open(BytesIO(img_bytes))
        if self.transform is None:
            if isinstance(self.image_size, int):
                img = self.resize_transform(img)
            img = self.to_tensor_transform(img)
            img = self.normalize_transform(img)
        else:
            img = self.transform(img)

        if self.return_type == 'image_only':
            return img
        elif self.return_type == 'pair':
            return img, item[self.label_key] if self.label_key else ''
        else:
            del item[self.img_key]
            item['image'] = img
            return item


def get_cv_dataset(path=DS_PATH_IMAGENET1K,
                   name=None,  # "full_size"
                   batch_size=1,
                   image_size=None,  # original size
                   split=None,  # 'train'
                   shuffle=True,
                   num_workers=4,
                   transform=None,
                   return_loader=False,
                   return_type='pair',
                   convert_rgb=False,
                   img_key=None,
                   **loader_params):
    if return_type not in ['image_only', 'pair', 'dict']:
        raise ValueError("return_type must be 'image_only' or 'pair' or 'dict'")

    dataset = load_dataset(path, trust_remote_code=True, split=split)

    if isinstance(split, str):
        custom_dataset = ImageLabelDataSet(dataset,
                                           transform=transform,
                                           return_type=return_type,
                                           split=split,
                                           image_size=image_size,
                                           convert_rgb=convert_rgb,
                                           img_key=img_key
                                           )
        if return_loader:
            return DataLoader(custom_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              **loader_params)
        else:
            return custom_dataset
    else:
        datasets_or_loader = {}
        for split_name in dataset.keys():
            datasets_or_loader[split_name] = ImageLabelDataSet(dataset,
                                                               transform=transform,
                                                               return_type=return_type,
                                                               split=split_name,
                                                               image_size=image_size,
                                                               convert_rgb=convert_rgb,
                                                               img_key=img_key)
        if return_loader:
            for split_name in datasets_or_loader:
                datasets_or_loader[split_name] = DataLoader(dataset=datasets_or_loader[split_name],
                                                            batch_size=batch_size,
                                                            shuffle=shuffle,
                                                            num_workers=num_workers,
                                                            **loader_params)
        return datasets_or_loader