"""
    这里我们假设一张图片中，只有前景和背景之分。如果一个一张图片中有多个目标，建议进行目标分割，
也就是分开为多张图片。
"""


import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from skimage import transform
from torch.utils.data import Dataset
from torchkeras.plots import joint_imgs_col, joint_imgs_row


class ISIC2016Dataset(Dataset):
    def __init__(
            self, img_files, img_transform=None, 
            mask_transform=None, transforms=None,
            use_point=False, use_bbox=True, use_random=True
    ):
        self.img_files = img_files
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.transforms = transforms

        self.use_point = use_point
        self.use_bbox = use_bbox
        self.use_random = use_random

    def get(self, index):
        img_path = self.img_files[index]
        mask_path = img_path.replace('Images', 'Masks').replace('.jpg', '_Segmentation.png')
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 制作 bbox
        bbox, point = None, None
        if self.use_bbox or self.use_point:
            mask_arr = np.array(mask)
            coords = np.argwhere(mask_arr > 0)
            # 找到最大最小值坐标
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)

            if self.use_bbox:
                # 坐标随机增加或减少 10%
                if self.use_random:
                    sign = random.choice([-1, 1])
                    x1 += sign * random.randint(0, int(0.1 * (x2 - x1)))
                    x1 = max(0, x1)
                    sign = random.choice([-1, 1])
                    x2 += sign * random.randint(0, int(0.1 * (x2 - x1)))
                    x2 = min(mask_arr.shape[1], x2)
                    sign = random.choice([-1, 1])
                    y1 += sign * random.randint(0, int(0.1 * (y2 - y1)))
                    y1 = max(0, y1)
                    sign = random.choice([-1, 1])
                    y2 += sign * random.randint(0, int(0.1 * (y2 - y1)))
                    y2 = min(mask_arr.shape[0], y2)

                bbox = np.array([
                    [x1, y1, x2, y2]
                ])

            if self.use_point:
                # 随机抽点
                if self.use_random:
                    point = np.array([
                        tuple(random.choice(coords))
                    ])
                else:
                    point = np.array([
                        tuple(coords.mean(axis=0))
                    ])

                # 更换 xy 坐标
                point = point[:, ::-1]

        prompt = {
            'bbox': bbox,
            'point': point
        }
        return img, mask, prompt

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img, mask, prompt = self.get(idx)

        img_arr = np.array(img)
        mask_arr = np.clip(np.array(mask), 0, 1)

        sample = {
            "image": img_arr,
            "mask": mask_arr,
        }

        cls_labels = None
        if self.use_point or self.use_bbox:
            cls_labels = np.array([1])

            if self.use_point:
                sample['keypoints'] = prompt['point']
            if self.use_bbox:
                sample['bboxes'] = prompt['bbox']

        if self.transforms:
            sample = self.transforms(**sample, class_labels=cls_labels)

        if self.img_transform:
            sample['image'] = self.img_transform(sample['image'])
        if self.mask_transform:
            sample['mask'] = self.mask_transform(sample['mask'])

        sample['mask'] = sample['mask'][None, ...]
        sample['idx'] = np.array([idx])

        if self.use_point or self.use_bbox:
            sample['cls_labels'] = cls_labels

            if self.use_point:
                sample['keypoints'] = np.array(sample['keypoints'])
            if self.use_bbox:
                sample['bboxes'] = np.array(sample['bboxes'])
        return sample

    def show_sample(self, index):
        image, mask, prompt = self.get(index)

        draw = ImageDraw.Draw(image)
        if prompt['bbox'] is not None:
            for coords in prompt['bbox']:
                draw.rectangle(tuple(coords), outline='red', width=2)
        if prompt['point'] is not None:
             for coords in prompt['point']:
                x, y = coords
                draw.ellipse(
                    [x - 5, y - 5, x + 5, y + 5], fill='red'
                )
        # image = resize_and_pad_image(image, self.img_size, self.img_size)
        # mask = resize_and_pad_image(mask, self.img_size, self.img_size)
        joint_imgs = joint_imgs_row(image, mask)
        return joint_imgs

