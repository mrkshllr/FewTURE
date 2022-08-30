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
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class MiniImageNet(Dataset):

    def __init__(self, setname, args, train_augmentation=None):
        IMAGE_PATH = os.path.join(args.data_path, 'miniimagenet_224/images')
        SPLIT_PATH = os.path.join(args.data_path, 'miniimagenet_224/split')

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))

        if (setname == 'val' or setname == 'test' or setname == 'train') and train_augmentation is None:
            image_size = args.image_size
            if image_size == 224:
                img_resize = 256
            elif image_size == 84:
                img_resize = 92
            else:
                ValueError('Image size not supported at the moment.')
            self.transform = transforms.Compose([
                # transforms.Resize([92, 92]),
                transforms.Resize([img_resize, img_resize]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                # transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                #                      np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
                ])
        elif setname == 'train' and train_augmentation is not None:
            self.transform = train_augmentation
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