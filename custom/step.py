class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None, optimizer=None, lr_scheduler=None):
        self.__dict__.update(locals())
        if self.stage == "train":
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch, *args, **kwargs):
        features, labels = batch['image'], batch['mask']

        # compute loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward
        if self.optimizer and self.accelerator and self.stage == 'train':
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # lr schedule
            if self.lr_scheduler:
                self.lr_scheduler.step()

        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # losses
        step_losses = {
            self.stage + "_loss": all_loss.item()
        }
        # metrics
        step_metrics = {
            self.stage + "_" + k: v(all_preds, all_labels).item() for k, v in self.metrics_dict.items()
        }
        if self.stage == "train":
            if self.optimizer:
                step_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            else:
                step_metrics['lr'] = 0

        return step_losses, step_metrics



import copy
from os.path import join

import monai
import torch
from matplotlib import pyplot as plt
# from peft import LoraConfig, get_peft_model, PeftModel
from peft import inject_adapter_in_model, LoraConfig, get_peft_model
from tqdm import tqdm

from Dataset import train_dataloader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

device = "cuda:0"
# 原始sam模型
ori_sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
print("mask_decoder.parameters",sum(p.numel() for p in ori_sam.mask_decoder.parameters()))
# print(ori_sam)
target_modules = ["image_encoder.blocks.{}.attn.qkv".format(i) for i in range(12)]
config = LoraConfig(target_modules=target_modules, r=8)
# 复制的sam加上lora
lora_sam = get_peft_model(ori_sam, config)
# lora_sam = inject_adapter_in_model(config, ori_sam)
# print(lora_sam)
# 只训练编码器的lora qkv部分和解码器
for param in lora_sam.base_model.model.mask_decoder.parameters():
    param.requires_grad = True
for param in lora_sam.base_model.model.image_encoder.parameters():
    param.requires_grad = True

print("lora_sam.parameters",sum(p.numel() for p in lora_sam.parameters()))

# merge_model = lora_sam.merge_and_unload()
chp_path = join('./ckp', "./0119")
lora_sam.to(device)
lora_sam.train()
for name, param in lora_sam.named_parameters():
    if param.requires_grad:
        print(name)
# print(lora_sam)
# 只传递可以优化的参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_sam.parameters()))
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# focal_loss=monai.losses.FocalLoss()
losses = []
best_loss = 1e10
# torch.cuda.empty_cache()

for epoch in range(800):
    epoch_loss = 0
    mem_before = torch.cuda.memory_allocated(device)
    for step, (name, image, label, point, point_label) in enumerate(tqdm(train_dataloader)):
        sam_trans = ResizeLongestSide(1024)
        coords_torch = torch.as_tensor(point, dtype=torch.float, device=device)
        # image=sam_trans.apply_image_torch(image)
        coords_torch = sam_trans.apply_coords_torch(coords_torch, (label.shape[-2], label.shape[-1]))
        labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=device)
        pt = (coords_torch, labels_torch)
        image = image.to(device)
        a = 1
        # mem_after = torch.cuda.memory_allocated(device)
        # mem_used = mem_after - mem_before
        # print(f"111Memory used for forward pass: {mem_used / 1024 ** 3} GiB")
        # mem_before = torch.cuda.memory_allocated(device)
        # for name, param in lora_sam.base_model.model.image_encoder.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        image_embedding = lora_sam.base_model.model.image_encoder(image)
        # mem_after = torch.cuda.memory_allocated(device)
        # mem_used = mem_after - mem_before
        # print(f"222Memory used for forward pass: {mem_used / 1024 ** 3} GiB")
        # print("---20/24.276")
        sparse_embeddings, dense_embeddings = lora_sam.base_model.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        binary_result, iou_predictions = lora_sam.base_model.model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=lora_sam.base_model.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # binary_result = binary_result.requires_grad_()
        # binary_result = F.interpolate(binary_result, size=label.shape[-2:], mode='bilinear', align_corners=False)
        label = label.to(device)
        loss = seg_loss(binary_result, label)
        print("开始loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the model checkpoint
    torch.save(lora_sam.state_dict(), join(chp_path, 'sam_model_latest.pth'))
    tqdm.write(f'EPOCH: {epoch}, Loss: {epoch_loss}')

    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(lora_sam.state_dict(), join(chp_path, 'sam_model_best.pth'))

    # %% plot loss
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show() # comment this line if you are running on a server
    plt.savefig(join(chp_path, 'train_loss.png'))
    plt.close()

