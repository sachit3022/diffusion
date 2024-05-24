import hydra
from torch.utils.data import DataLoader
from datasets import GaussianMixtureDataset
from utils import plot_density_from_samples,set_seed
from models import TimeInputMLP,MLPEncoder,MLPDecoder,LabelEncoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import torch
from encoders import VAE,LabelCLIP
from score import DDPMProcess
from EMA import EMA


@hydra.main(config_path='configs', config_name='gmm')
def main(cfg):


    print(cfg)

    """

    ########## Hyperparameters and settings ##########
    set_seed(42)
    num_epochs = 200
    lr_monitor = LearningRateMonitor(logging_interval='step')
    diff_logger =TensorBoardLogger('lightning_logs', name='DDPM')
    vae_logger = TensorBoardLogger('lightning_logs', name='VAE')
    clip_logger = TensorBoardLogger('lightning_logs', name='CLIP')
    plots_dir= 'plots'
    d_latent = 16
    d_input=2
    num_classes= 8
    num_layers = 2
    mlp_ratio = 2.0
    dropout = 0.1
    beta = 0.0025*4 # beta used in the stable diffusion paper
    accelerator= 'gpu'
    latent_encoding_checkpoint = None#'lightning_logs/VAE/version_36/checkpoints/epoch=199-step=8000.ckpt'
    label_encoding_checkpoint = None#'lightning_logs/CLIP/version_9/checkpoints/epoch=99-step=3200.ckpt'
    diffusion_checkpoint = None
    ########## Dataset ##########
    
    
    train_dataset = GaussianMixtureDataset(n=20000,k=num_classes)
    val_dataset = GaussianMixtureDataset(n=1000,k=num_classes)
    val_dataset.alphas = train_dataset.alphas
    val_dataset.samples,val_dataset.labels = val_dataset.sample(1000)

    fig,axes = plt.subplots(2,2,figsize=(10,5))


    plot_density_from_samples(val_dataset.samples.detach().numpy(), filepath=f"{plots_dir}/val_input_samples.png", show=False, save=False,title="Val input samples density", ax=axes[0,0])
    plot_density_from_samples(train_dataset.samples.detach().numpy(), filepath=f"{plots_dir}/train_input_samples.png", show=False, save=False,title="Train input samples density", ax=axes[0,1])


    #normalise the data
    data_mean, data_std = train_dataset.samples.mean(dim=0), train_dataset.samples.std(dim=0)
    train_dataset.samples = (train_dataset.samples - data_mean) / data_std
    val_dataset.samples = (val_dataset.samples - data_mean) / data_std


    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4,persistent_workers=True)

    if latent_encoding_checkpoint is None:
        encoder = MLPEncoder(d_input,2*d_latent,dropout=dropout) #mu and sigma
        decoder = MLPDecoder(d_latent,d_input,dropout=dropout)
        vae = VAE(encoder,decoder,d_input,d_latent,beta=beta)
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=accelerator, log_every_n_steps=5, logger=vae_logger,gradient_clip_val=0.5,callbacks=[lr_monitor,EMA(decay=0.9999,validate_original_weights=True,every_n_steps=10)])# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
        trainer.fit(vae, train_dataloader, val_dataloader)
    else:
        vae = VAE.load_from_checkpoint(latent_encoding_checkpoint,d_in=d_input,d_latent=d_latent)

    ######## Extract the encoder and decoder ########
    
    image_encoder = vae.encoder
    image_decoder = vae.decoder

    ################## Train Label encoder ########

    if label_encoding_checkpoint is None:
        label_encoder= LabelEncoder(num_classes+1,d_latent)
        label_clip = LabelCLIP(vae.encoder,label_encoder, d_latent, num_classes)
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=accelerator, log_every_n_steps=5, logger=clip_logger,gradient_clip_val=0.5,callbacks=[lr_monitor,EMA(decay=0.9999,validate_original_weights=True,every_n_steps=10)])# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
        trainer.fit(label_clip, train_dataloader, val_dataloader) 
    else:
        label_clip = LabelCLIP.load_from_checkpoint(label_encoding_checkpoint,encoder=vae.encoder,d_latent=d_latent,num_classes=num_classes)

    ########## Extract the label encoder ########

    label_encoder = label_clip.label_encoder  

    ########## SANITY CHECK  before the diffusion model #############
    
    vae.eval()

    all_x = []
    for batch in val_dataloader:
        x,_ = batch
        x= x.to(vae.device)

        x_recon, mean, log_var = vae(x)
        x_recon = x_recon*data_std.to(vae.device) + data_mean.to(vae.device)
        all_x.append(x_recon)
    all_x = torch.cat(all_x)
    
    plot_density_from_samples(all_x.detach().cpu().numpy(), filepath=f"{plots_dir}/vae_reconstructions.png", show=False, save=False,title="vae_reconstructions", ax=axes[1,0])
    
    x_hat = vae.sample(n=1000)*data_std.to(vae.device) + data_mean.to(vae.device)
    plot_density_from_samples(x_hat.detach().cpu().numpy(), filepath=f"{plots_dir}/vae_samples.png", show=False, save=False, title="vae_samples",ax=axes[1,1])


    #test label encoder and check for posterior colapse.
    label_clip.eval()
    z = torch.arange(8).to(vae.device)
    z_latent = label_clip.label_encoder(z)
    x_hat = vae.decoder(z_latent)*data_std.to(vae.device) + data_mean.to(vae.device) 
    #plot the centers of x_hat
    for i in range(8):
        axes[1,0].scatter(x_hat[i,0].item(),x_hat[i,1].item(),c='g',s=100,marker=f'${i}$')
    
    plt.savefig(f"{plots_dir}/vae_samples.png")
    

    ########## Test the diffusion ##########
    # image_encoder = nn.Identity()
    # label_encoder = nn.Identity()
    # image_decoder = nn.Identity()
    # diffusion_model = DiffMLP(d_latent,num_layers=num_layers, mlp_ratio=mlp_ratio)
    ############ Train diffusion model ########
    
    diffusion_model = TimeInputMLP(d_latent,num_layers=num_layers, mlp_ratio=mlp_ratio)

    if diffusion_checkpoint is None:
        diff_process  = DDPMProcess(d_latent = d_latent, image_encoder = image_encoder,label_encoder = label_encoder, diffusion_model = diffusion_model,image_decoder = image_decoder)
        diff_trainer = pl.Trainer(max_epochs=num_epochs,devices=1,accelerator='gpu', log_every_n_steps=2,gradient_clip_val=0.5, logger=diff_logger,callbacks=[lr_monitor,EMA(decay=0.999,validate_original_weights=True,every_n_steps=5)])# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
        diff_trainer.fit(diff_process, train_dataloader, val_dataloader)
    else:
        diff_model = DDPMProcess.load_from_checkpoint(diffusion_checkpoint,d_latent=d_latent,image_encoder=image_encoder,label_encoder=label_encoder,diffusion_model=diffusion_model,image_decoder=image_decoder)

    ########## Generate samples ##########
    for y in range(num_classes):
        x_gen = diff_process.generate(N=1000,y=y)
        x_gen = x_gen*data_std.to(diff_process.device) + data_mean.to(diff_process.device)
        plot_density_from_samples(x_gen.detach().cpu().numpy(), filepath=f"{plots_dir}/ddpm_samples_{y}.png", show=False, save=True,title=f"Samples for class {y}")
    """

if __name__ == "__main__":
    main()

