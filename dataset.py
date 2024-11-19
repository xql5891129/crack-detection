import os

import cv2
import numpy as np
import torch.utils.data

#处理数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids#图像的文件名列表
        self.img_dir = img_dir#存储图像的目录路径
        self.mask_dir = mask_dir#存储标签掩码的目录路径
        self.img_ext = img_ext#图像文件的扩展名
        self.mask_ext = mask_ext#标签掩码文件的扩展名
        self.num_classes = num_classes#标签的类别数
        self.transform = transform#数据增强的转换函数

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        #数组沿深度方向进行拼接。
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}



