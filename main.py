import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split
from datasets import GaussianMixtureDataset,ColoredMNIST
from utils import plot_samples,set_seed,compute_mean_std,inverse_transform
from models import TimeInputMLP,MLPEncoder,MLPDecoder,LabelEncoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import torch
from encoders import VAE,LabelCLIP
from score import DDPMProcess
from EMA import EMA
from torchvision import transforms
from torch import nn
from fairness import FairPlModule
import tqdm

@hydra.main(config_path='configs', config_name='colored_mnist')#colored_mnist # cg
def main(cfg):
    
    ########## Hyperparameters and settings ##########
    set_seed(42)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ema_callback = EMA(decay=0.9999,validate_original_weights=True,every_n_steps=10)

    diff_logger =TensorBoardLogger(save_dir = 'lightning_logs', name='DDPM')
    vae_logger = TensorBoardLogger(save_dir = 'lightning_logs', name='VAE')
    clip_logger = TensorBoardLogger(save_dir = 'lightning_logs', name='CLIP')
    fair_logger = TensorBoardLogger(save_dir = 'lightning_logs', name='fairness')
    

    ########## Dataset ##########
        
    train_dataset = instantiate(cfg.dataset,train=True)
    val_dataset = instantiate(cfg.dataset,train=False)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,**cfg.dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,shuffle=False,**cfg.dataloader)
    
    
    latent_model = instantiate(cfg.diffusion.latent_encoding)
  
    if cfg.diffusion.latent_encoding_checkpoint:
        encoder = latent_model.encoder
        decoder = latent_model.decoder
        latent_model = VAE.load_from_checkpoint(cfg.diffusion.latent_encoding_checkpoint,encoder=encoder,decoder=decoder)
    latent_trainer = pl.Trainer(callbacks=[lr_monitor,ema_callback],logger=vae_logger,**cfg.diffusion.train_latent_encoding)    # StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
    if cfg.diffusion.resume_latent_encoding: 
        latent_trainer.fit(latent_model, train_dataloader, val_dataloader)

    ################## Train Label encoder #################
    label_clip_partial = instantiate(cfg.diffusion.label_encoding)
    label_clip = label_clip_partial(latent_model=latent_model)
    if cfg.diffusion.label_encoding_checkpoint:
        label_clip = LabelCLIP.load_from_checkpoint(cfg.diffusion.label_encoding_checkpoint,latent_model=latent_model,label_encoder=label_clip.label_encoder)
    trainer = pl.Trainer(callbacks=[lr_monitor,ema_callback],logger=clip_logger, **cfg.diffusion.train_label_encoding)# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
    if cfg.diffusion.resume_label_encoding:
        trainer.fit(label_clip, train_dataloader, val_dataloader) 

    ########## Extract the label encoder ########

    label_encoder = label_clip.label_encoder
    label_encoder.eval()
    latent_model.eval() 
    
    image_encoder = nn.Identity() # we are not doing latent diffusion
    image_decoder = nn.Identity()

    #image_encoder = latent_model.encoder
    #image_decoder = latent_model.decoder

    ########## SANITY CHECK  before the diffusion model #############
    #verify(val_dataset,latent_model,label_encoder)


    ########## Train the diffusion model #############

    diffusion_model= instantiate(cfg.diffusion.sampling.diffusion_model)    
    diff_process  = DDPMProcess(d_latent = cfg.diffusion.sampling.d_latent,diffusion_model =diffusion_model, image_encoder=image_encoder, label_encoder=label_encoder, image_decoder=image_decoder,lr = cfg.diffusion.sampling.lr,inverse_transform=train_dataset.inverse_transform)
    if cfg.diffusion.sampling_checkpoint:
        diff_process = DDPMProcess.load_from_checkpoint(cfg.diffusion.sampling_checkpoint,d_latent = cfg.diffusion.sampling.d_latent,diffusion_model =diffusion_model, image_encoder=image_encoder, label_encoder=label_encoder, image_decoder=image_decoder,lr = cfg.diffusion.sampling.lr)
    diff_trainer = pl.Trainer(callbacks=[lr_monitor,ema_callback],logger=diff_logger,**cfg.diffusion.train_sampling) #   # StochasticxzWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
    if cfg.diffusion.resume_sampling:
         diff_trainer.fit(diff_process, train_dataloader, val_dataloader)
    
    ############### SANITY CHECK AFTER THE DIFFUSION MODEL #############

    #interpolate(val_dataloader,diff_process)




def interpolate(val_dataloader,diff_process):
    
    
    #from val
    val_samples = val_dataloader.__iter__().__next__()

    x = val_samples['X']
    y = val_samples['label']
    s = val_samples['sensitive']

    #interpolate bettween (0,0) -> (1,1)

    
    condition_1 = torch.logical_and(y == 0,s == 0) 
    condition_2 = torch.logical_and(y == 1,s == 1)

    # min count of condition 1 and condition 2
    min_count = min(condition_1.sum(),condition_2.sum())

    x,y,s = x.to(diff_process.device),y.to(diff_process.device),s.to(diff_process.device)
    x_1 = x[condition_1][:min_count]
    x_2 = x[condition_2][:min_count]

    diff_process.eval()


    x_gen = diff_process.interpolate(x_1,x_2)
    plot_samples(x_gen.detach().cpu(),how='density',title='interpolated_samples',save=True,show=False,filepath=f'inverse_interpolated_samples.png',ax=None)

   
    
    


def generate(val_dataset,diff_process):
    
    #plot_samples(val_sample.detach().cpu(),how=plot_style,title='val_samples',save=True,show=False,filepath=f'inverse_val_samples_0.png',ax=None)
    #val_s = torch.randint(0,10,(32,),device=val_sample.device,dtype=val_y.dtype)
    #val_y = val_s.clone()
    x_gen = diff_process.generate(N=val_y.size(0),samples=val_sample,y=val_y,s=val_s)
    
    
    plt.scatter(x_gen[:,0].detach().cpu(),x_gen[:,1].detach().cpu(),c=val_y.detach().cpu(),s=100)



    plt.savefig('generated_samples.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_samples(x_gen,how='density',title='generated_samples',save=False,show=False,filepath=f'inverse_generated_samples_test.png',ax=ax)
    plt.savefig('label_encoder.png')





def verify(val_dataset,latent_model,label_encoder):
    
    #print(compute_mean_std_of_latents(train_dataloader,latent_model))

    #plot tsne



    fig,axes = plt.subplots(2,2,figsize=(10,5))

    val_sample = val_dataset.sample(16)
    val_sample = val_sample.to(latent_model.device)
    
    val_recon, _, _ = latent_model(val_sample)
    x = latent_model.sample(16)
    


    plot_style = 'density' if len(val_recon.shape) == 2 else 'image'

    #check if x is a 1d,2d or 3d tensor
    
    
    plot_samples(val_sample.detach().cpu(), filepath=f"val_input_samples.png", how=plot_style,show=False, save=True, title="Val input samples",ax=axes[0,0])
    plot_samples(val_recon.detach().cpu(), filepath=f"vae_reconstructions.png", how=plot_style,show=False, save=True, title="Val Reconstructions",ax=axes[1,0])
    plot_samples(x.detach().cpu(), filepath=f"vae_samples.png", how=plot_style,show=False, save=True, title="Samples from the VAE",ax=axes[0,1])

    #test label encoder and check for posterior colapse.

    #same labels for all the samples can be color or
    label_samples = val_dataset.sample_labels() 
    z_latent = label_encoder(label_samples.to(label_encoder.device)) #bring labels to the same latent a space as the data
    x_hat = latent_model.decoder(z_latent)
    if plot_style == 'image':
        x_hat = x_hat.view(-1,3,28,28)
        plot_samples(x_hat.detach().cpu(), filepath=f"label_vae_samples.png", how=plot_style,show=False, save=True, title="Samples for each label",ax=axes[1,1])
    else:
        for i in range(8):
            axes[1,1].scatter(x_hat[i,0].item(),x_hat[i,1].item(),c='g',s=100,marker=f'${i}$')
    
    


    


def compute_mean_std_of_latents(train_dataloader,latent_model):
    #compute variance of encode
    mu =0
    for train_sam in train_dataloader:
        x,  y = train_sam['image'],train_sam['label']
        x = x.to(latent_model.device)
        z = latent_model.encode_with_log_var(x)
        z = z.view(z.size(0), -1)
        mu += z.mean()
    mu /= len(train_dataloader)
    var = 0
    for train_sam in train_dataloader:
        x,  y =  train_sam['image'],train_sam['label']
        x = x.to(latent_model.device)
        z = latent_model.encode_with_log_var(x)
        z = z.view(z.size(0), -1)
        var += ((z - mu)**2).mean()
    var /= len(train_dataloader)
    print(f"Mean of the latent space: {mu}")
    print(f"Variance of the latent space: {var}")
    print(f"Correction scale {1/torch.sqrt(var)}")
    return 1/torch.sqrt(var), mu

def plot_tsne(val_dataloader,latent_model):
    samples = []
    samples_y = []
    for val_sample in val_dataloader:
        x,  y = val_sample
        
        samples_y.append(y)

        x = x.to(latent_model.device)
        z = latent_model.encode(x)
        z = z.view(z.size(0), -1)
        samples.append(z)
    
    
    
    z = torch.cat(samples)
    samples_y = torch.cat(samples_y)

    #convert to numpy
    z = z.detach().cpu().numpy()
    samples_y = samples_y.detach().cpu().numpy()
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    z_tsne = tsne.fit_transform(z)
    plt.figure()
    for i in range(10):
        plt.scatter(z_tsne[samples_y == i, 0], z_tsne[samples_y == i, 1], label=f'{i}',s=10)
    plt.legend()
    plt.title("TSNE of the latent space")
    plt.savefig("tsne_latent_space.png")
    plt.close()

if __name__ == "__main__":  
    main()

