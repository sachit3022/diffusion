from torchvision.models import resnet18
import torch
from datasets import ColoredMNIST,CounterfactualColoredMNIST
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import set_seed
import hydra
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


class FairPlModule(pl.LightningModule):
    def __init__(self,lr=1e-3):
        super().__init__()
        self.model =  resnet18(pretrained=False)
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        


    def training_step(self, batch, batch_idx):

        x, y,c =  batch['image'],batch['label'],batch['color']

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_acc', self.accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_acc', self.accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        e_opt = torch.optim.AdamW( list(self.model.parameters()), lr=self.lr)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False
        }
        return [e_opt],[e_scheduler]
    def accuracy(self, y_hat, y):
        return (y_hat.argmax(1) == y).float().mean()
    

@hydra.main(config_path='configs', config_name='fairness')
def main(cfg):

    set_seed(42)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tensorboard_logger = TensorBoardLogger(save_dir = 'lightning_logs', name='fairness')

    train_dataset = CounterfactualColoredMNIST(root_dir='/research/hal-gaudisac/Diffusion/image_gen/outputs/2024-06-10/20-04-10',train=True,threshold=1)#
    val_dataset = ColoredMNIST(root_dir='/research/hal-gaudisac/Diffusion/image_gen/data', download=False, threshold=0.1, train=False)
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,**cfg.dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,shuffle=False,**cfg.dataloader)

    pl_module = FairPlModule()
    trainer = pl.Trainer(callbacks=[lr_monitor], logger=tensorboard_logger, **cfg.trainer)
    trainer.fit(pl_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()


