import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchkeras.data import resize_and_pad_image
from torchkeras.plots import joint_imgs_col, joint_imgs_row


class CustomDataset(Dataset):
    def __init__(self, img_files, img_size, transforms = None):
        self.__dict__.update(locals())

    def __len__(self):
        return len(self.img_files)

    def get(self, index):
        img_path = self.img_files[index]
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return image, mask

    def __getitem__(self, index):
        image, mask = self.get(index)

        # resize and pad
        image = resize_and_pad_image(image, self.img_size, self.img_size)
        mask = resize_and_pad_image(mask, self.img_size, self.img_size)

        # img -> array
        image_arr = np.array(image, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32)
        mask_arr = np.where(mask_arr > 100, 1, 0).astype(np.int64)

        sample = {
            "image": image_arr,
            "mask": mask_arr
        }
        if self.transforms is not None:
            sample = self.transforms(**sample)

        sample['mask'] = sample['mask'][None, ...]

        return sample

    def show_sample(self, index):
        image, mask = self.get(index)
        # image = resize_and_pad_image(image, self.img_size, self.img_size)
        # mask = resize_and_pad_image(mask, self.img_size, self.img_size)
        joint_imgs = joint_imgs_row(image, mask)
        return joint_imgs