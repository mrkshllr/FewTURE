# Copyright (c) Markus Hiller and Rongkai Ma -- 2022
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Dataloader build upon the DeepEMD repository, available under https://github.com/icoz69/DeepEMD/tree/master/Models/dataloader
"""
#
#
import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetLoader(Dataset):

    def __init__(self, setname, args, train_augmentation=None):

        DATASET_DIR = os.path.join(args.data_path, 'cifar_fs')

        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'meta-train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'meta-test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'meta-val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Incorrect set name. Please check!')

        data = []
        label = []


        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if setname == 'train' and train_augmentation is not None:
            self.transform = train_augmentation

        elif (setname == 'val' or setname == 'test') and train_augmentation is None:
            image_size = args.image_size
            if image_size == 224:
                img_resize = 256
            elif image_size == 84:
                img_resize = 92
            else:
                ValueError('Image size not supported at the moment.')
            self.transform = transforms.Compose([
                transforms.Resize([img_resize, img_resize]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)) # Differs from ImageNet standard!
            ])
        else:
            ValueError("Set name or train augmentation corrupt. Please check!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass