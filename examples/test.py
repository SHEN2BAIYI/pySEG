# %% [markdown]
# # Segment Anything 自定义推理

# %% [markdown]
# # 一、环境准备

# %% [markdown]
# ## 1.检查 CUDA 状态

# %% [markdown]
# 多卡需禁用，或者运行后重启内核。

# %%
import torch


def check_cuda():
    flag = torch.cuda.is_available()
    if flag:
        print("CUDA可使用")
    else:
        print("CUDA不可用")

    # 获取GPU数量
    ngpu = torch.cuda.device_count()
    print("GPU数量：", ngpu)
    # Decide which device we want to run on
    device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("驱动为：", device)
    print("GPU型号： ", torch.cuda.get_device_name(0))


check_cuda()

# %% [markdown]
# ## 2.环境超参

# %%
from argparse import Namespace
import sys

sys.path.insert(0, '..')

config = Namespace(
    img_size=1024,
    batch_size=1,
    num_workers=2,
    model_name="vit_b",
    model_cpt_path="../checkpoints/sam/sam_vit_b_01ec64.pth",
    random_prompt=False,

    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
)

# %% [markdown]
# ## 3.可视化方法

# %%
import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def trans_ori_comparision(
        idxs, images, masks, ori_dataset,
        points=None, bboxes=None, cls_labels=None,
        mean=None, std=None, max_show_num=1,
):
    assert images is not None and masks is not None and idxs is not None, "image, label, id 不可缺少"
    assert max_show_num <= idxs.shape[0], "展示长度大于图片集长度"

    # tensor -> numpy
    idxs = idxs.detach().cpu().numpy()
    imgs_arr = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    masks_arr = masks.detach().cpu().permute(0, 2, 3, 1).numpy()
    masks_arr = masks_arr > 0

    if points is not None or bboxes is not None:

        cls_labels_arr = cls_labels.detach().cpu().numpy()

        if points is not None:
            points_arr = points.detach().cpu().numpy()

        if bboxes is not None:
            bboxs_arr = bboxes.detach().cpu().numpy()

    if mean is not None and std is not None:
        imgs_arr = imgs_arr * std + mean

    for i in range(max_show_num):
        idx = idxs[i][0]
        img_arr = imgs_arr[i]
        mask_arr = masks_arr[i].squeeze()

        # 创建一个包含两个子图的图形
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # 显示第一个图像
        axs[0].imshow(img_arr)
        axs[0].axis('on')
        show_mask(mask_arr, axs[0])

        # 显示第二个图像
        img_ori, mask_ori, prompt = ori_dataset.get(idx)
        img_ori = np.array(img_ori)
        mask_ori = np.array(mask_ori) > 0
        point_ori = prompt.get('point', None)
        bbox_ori = prompt.get('bbox', None)

        axs[1].imshow(img_ori)
        axs[1].axis('on')
        show_mask(mask_ori, axs[1])

        # 统一画线
        if points is not None or bboxes is not None:
            cls_label_arr = cls_labels_arr[i]
            if points is not None:
                point_arr = points_arr[i]
                show_points(point_arr, cls_label_arr, axs[0])
                show_points(point_ori, cls_label_arr, axs[1])

            if bboxes is not None:
                bbox_arr = bboxs_arr[i]
                for bbox_1, bbox_2 in zip(bbox_arr, bbox_ori):
                    show_box(bbox_1, axs[0])
                    show_box(bbox_2, axs[1])

        # 显示图形
        plt.show()


# %% [markdown]
# # 二、推理

# %% [markdown]
# ## 1.定义模型

# %%
from segment_anything import sam_model_registry

sam = sam_model_registry[config.model_name](checkpoint=config.model_cpt_path)
config.img_size = sam.image_encoder.img_size

# %% [markdown]
# ## 2.定义数据

# %% [markdown]
# ### 1.定义数据增强方式

# %%
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_transform():
    train_transforms = A.Compose(
        [
            # A.OneOf([
            #     A.HorizontalFlip(p=0.5),
            #     A.VerticalFlip(p=0.5),
            # ]),
            A.LongestMaxSize(
                max_size=config.img_size, p=1.0
            ),
            A.Normalize(
                mean=config.mean,
                std=config.std,
                max_pixel_value=1,
                p=1.0
            ),
            A.PadIfNeeded(
                min_height=config.img_size, min_width=config.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                position='top_left',
                value=0, p=1.0
            ),
            ToTensorV2(p=1),
        ],
        p=1.0,
        keypoint_params=A.KeypointParams(format='xy'),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )
    val_transforms = A.Compose(
        [
            A.LongestMaxSize(
                max_size=config.img_size, p=1.0
            ),
            A.PadIfNeeded(
                min_height=config.img_size, min_width=config.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0, p=1.0
            ),
            A.Normalize(
                mean=config.mean,
                std=config.std,
                max_pixel_value=1,
                p=1.0
            ),
            ToTensorV2(p=1),
        ],
        keypoint_params=A.KeypointParams(format='xy'),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )
    return train_transforms, val_transforms


# %% [markdown]
# ### 2.定义 dataset 和 dataloader

# %%
from custom.sam.datasets.isic2016 import ISIC2016Dataset
from pathlib import Path
import torch
import torch.utils
import torchvision

from torch.utils.data import DataLoader, ConcatDataset


def create_dataset():
    train_imgs = [
        '/home/zijieshen/new_disk/datasets/ISIC2016/P1/Train/Images/ISIC_0000000.jpg'
    ]
    val_imgs = [
        str(x) for x in Path("/home/zijieshen/new_disk/datasets/ISIC2016/P1/Val").rglob("*.jpg")
        if "checkpoint" not in str(x)
    ]
    train_ts, val_ts = create_transform()
    train_ds = ISIC2016Dataset(
        img_files=train_imgs, transforms=train_ts,
        use_bbox=True, use_point=True, use_random=config.random_prompt
    )
    val_ds = ISIC2016Dataset(
        img_files=val_imgs, transforms=val_ts,
        use_bbox=True, use_point=True, use_random=False
    )
    return train_ds, val_ds


def create_dataloader(train_ds=None, val_ds=None):
    if train_ds is None or val_ds is None:
        train_ds, val_ds = create_dataset()

    train_dl = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    return train_dl, val_dl


train_dataset, val_dataset = create_dataset()
train_dataloader, val_dataloader = create_dataloader(train_dataset, val_dataset)

# %% [markdown]
# ### 3.数据可视化

# %%
import matplotlib.pyplot as plt
import numpy as np

for batch in train_dataloader:
    images = batch.get('image', None)
    masks = batch.get('mask', None)
    idxs = batch.get('idx', None)

    points = batch.get('keypoints', None)
    bboxes = batch.get('bboxes', None)
    cls_labels = batch.get('cls_labels', None)

    trans_ori_comparision(
        idxs, images, masks, train_dataset, points, bboxes, cls_labels,
        mean=config.mean, std=config.std, max_show_num=1
    )

    break

# %% [markdown]
# ## 3.推理

# %% [markdown]
# ### 1.点推理

# %% [markdown]
# #### 1.坐标合并

# %%
point_coords = (points, cls_labels)

# %% [markdown]
# #### 2.推理

# %%
with torch.no_grad():
    features = sam.image_encoder(images)

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=point_coords,
        boxes=None,
        masks=None
    )

    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=features,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    masks = sam.postprocess_masks(
        low_res_masks, images.shape[-2:], images.shape[-2:]
    )

    masks = masks > 0

# %% [markdown]
# #### 3.可视化

# %%
trans_ori_comparision(
    idxs, images, masks, train_dataset, points, bboxes, cls_labels,
    mean=config.mean, std=config.std, max_show_num=1
)

# %% [markdown]
# ### 2.框推理

# %% [markdown]
# #### 1.推理

# %%
with torch.no_grad():
    features = sam.image_encoder(images)

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=bboxes,
        masks=None
    )

    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=features,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    masks = sam.postprocess_masks(
        low_res_masks, images.shape[-2:], images.shape[-2:]
    )

# %% [markdown]
# #### 2.可视化

# %%
trans_ori_comparision(
    idxs, images, masks, train_dataset, points, bboxes, cls_labels,
    mean=config.mean, std=config.std, max_show_num=1
)


