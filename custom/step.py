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

