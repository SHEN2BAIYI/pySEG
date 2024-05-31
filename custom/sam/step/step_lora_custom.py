import torch
from segment_anything.utils.transforms import ResizeLongestSide

class SamLoraCustomStepRunner:
    def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None, optimizer=None, lr_scheduler=None):
        self.__dict__.update(locals())
        if self.stage == "train":
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch, *args, **kwargs):
        
        features = batch.get('image', None)
        labels = batch.get('mask', None)
        points = batch.get('keypoints', None)
        bboxes = batch.get('bboxes', None)
        cls_labels = batch.get('cls_labels', None)

        if points is not None:
            points = (points, cls_labels)

        # ---> image encoder <---
        image_embeddings = self.net.sam.image_encoder(features)
        # ---> prompt encoder <---
        sparse_embeddings, dense_embeddings = self.net.sam.prompt_encoder(
            points=points,
            boxes=bboxes,
            masks=None
        )

        # ---> mask decoder <---
        low_res_masks, iou_predictions = self.net.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.net.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # 对 binary_result 进行线性插值 256 -> 1024
        trans = ResizeLongestSide(
            self.net.sam.image_encoder.img_size
        )
        low_res_masks = trans.apply_image_torch(low_res_masks)

        loss = self.loss_fn(low_res_masks, labels)

        # backward
        if self.optimizer and self.accelerator and self.stage == 'train':
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()    
            # lr schedule
            if self.lr_scheduler:
                self.lr_scheduler.step()
            

        all_preds = self.accelerator.gather(low_res_masks)
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
