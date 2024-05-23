import tqdm
import torch
from torch import nn as nn
import numpy as np
import  pytorch_lightning as pl
from EMA import EMA
from utils import set_seed, plot_density_from_samples
from torch.utils.data import DataLoader
from ddpm import GaussianMixtureDataset
import matplotlib.pyplot as plt
from ddpm import DiffMLP, VAE, LabelCLIP,TimestepEmbedder,TimeInputMLP

class ScoreBasedModelling(pl.LightningModule):

    def __init__(self, d_latent,image_encoder,label_encoder, diffusion_model,image_decoder):
        super().__init__()
        self.model = diffusion_model
        self.image_encoder = image_encoder #convert image to latent space
        self.label_encoder = label_encoder #convert label to latent space
        self.image_decoder = image_decoder
        self.non_class_conditional_token = 8

        self.loss = nn.MSELoss(reduction='none')
        self.N=500
        self.tspan = (1e-2,5.)
        self.d_latent = d_latent
        t_list = torch.linspace(self.tspan[0], self.tspan[1], self.N)
        self.register_buffer('t_list', t_list)
        
        #freeze the image encoder and label encoder and image decoder
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
        self.label_encoder.eval()
        self.label_encoder.requires_grad_(False)
        self.image_decoder.eval()
        self.image_decoder.requires_grad_(False)
        self.save_hyperparameters(ignore=['image_encoder','label_encoder','diffusion_model','image_decoder'])

    
    def forward_solve(self, u0, t, type='exact'):
        raise NotImplementedError
    def score_p_x_t_given_x0(self, xt, x0, t_list):
        raise NotImplementedError
    def reverse_solve(self, y, eta_t, tau):
        raise NotImplementedError
    
    
    @torch.no_grad()
    def generate(self,N=1000,y=None):
        self.model.eval()
        x_t = torch.randn((N, self.d_latent), requires_grad=False,device = self.device)
        t = torch.ones(N,device = self.device,dtype=torch.long)*(self.N-1)
        y = self.label_encoder(torch.ones(N,device=self.device,dtype=torch.long)*(y if y is not None else self.non_class_conditional_token))
        
        #X = torch.zeros((N, 2, self.N), requires_grad=False)
        
        for i in tqdm.tqdm(range(self.N-1)): 
            eta_t = self.model(x_t,t,y)
            x_t = self.reverse_solve(x_t, eta_t,i)
            t = t - 1
            #X[:,:,i+1] = x_t
        
        x_t = self.image_decoder(x_t)
        return x_t
    
    def training_step(self, batch, batch_idx):
        x0,y = batch
        x0 = self.image_encoder(x0).chunk(2,dim=-1)[0]
        #prob 0-1 for all y and make them self.non_class_conditional_token 20%
        
        y = torch.where(torch.rand(y.shape,device=self.device) < 0.2, torch.ones_like(y,device=self.device)*self.non_class_conditional_token, y)
        y = self.label_encoder(y)

        t_index = torch.randint(0, self.N, (x0.shape[0],),device=self.device)
        xt = self.forward_solve(x0, t_index.reshape(-1,1), type='exact')
        score_x_t = self.score_p_x_t_given_x0(xt, x0, t_index.reshape(-1,1))
        t_list = self.t_list[t_index]
        l_t =(1 - torch.exp(-2*t_list)) #torch.ones_like(t_list) # self.one_minus_alpha_t_bar[t_index] #self.one_minus_alpha_t_bar[t_index] #self.one_minus_alpha_t_bar[t_index]  #
        score_pred = self.model(xt, t_index,y)
        l = ((self.loss(score_pred, score_x_t).sum(dim=-1)*l_t)/l_t.sum()).sum()
        self.log_dict({'train_loss': l})
        return {'loss': l}
    
    def configure_optimizers(self):
        e_opt = torch.optim.AdamW( list(self.model.parameters()), lr=1e-2)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False
        }
        return [e_opt],[e_scheduler]



class FwdOrnsteinUhlenbeckProcess(ScoreBasedModelling):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_solve(self, u0, t, type='exact'):
        t = self.t_list[t]
        w = torch.randn(u0.shape)
        if type == 'exact':
            return u0 * np.exp(-t) + np.sqrt(1 - np.exp(-2*t)) * w
        if type == 'sde':
            out = u0
            tau  = t / self.N
            for _ in range(self.N):
                out = out - tau * out + np.sqrt(2 * tau) * w
        return out 

    def score_p_x_t_given_x0(self, xt, x0, t_index):
        t_list = self.t_list[t_index]
        return (x0 * torch.exp(-1*t_list) - xt)/ (1 - torch.exp(-2*t_list))
    
    def reverse_solve(self, y, eta_t, i):
        beta = 0.1
        tau = self.tspan[1] / self.N
        w = torch.normal(0,1,(y.shape),device=self.device)
        const_temp1 = torch.Tensor([1+beta], device = self.device)
        const_temp2 = torch.Tensor([tau * beta],device = self.device)
        return y + tau*(y + const_temp1*eta_t) + (2*const_temp2)**(0.5)*w

class DDPMProcess(ScoreBasedModelling):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        betas =  self.constant_beta_schedule()
        alpha_t = 1 -  betas
        alpha_t_bar = torch.cumprod(alpha_t, 0)


        sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar)
        one_minus_alpha_t_bar = 1. - alpha_t_bar

        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alpha_t_bar', sqrt_alpha_t_bar)
        self.register_buffer('one_minus_alpha_t_bar', one_minus_alpha_t_bar)
    
    def constant_beta_schedule(self):
        return torch.ones((self.N,))* 0.02 
    
    def linear_beta_schedule(self):
        beta_start = 0.01
        beta_end = 0.03
        return torch.linspace(beta_start, beta_end, self.N)
    
    def sigmoid_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, self.N)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        timesteps = self.N
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def score_p_x_t_given_x0(self, xt, x0, t_list):
        alpha_bar = self.sqrt_alpha_t_bar[t_list]
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_list]
        return (alpha_bar*x0 - xt)/(one_minus_alpha_t_bar)

    def forward_solve(self, u0, t_list, type='exact'):
        w = torch.randn(u0.shape,device=self.device)
        alpha_bar = self.sqrt_alpha_t_bar[t_list]
        
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_list]
        return alpha_bar*u0 + torch.sqrt(one_minus_alpha_t_bar)*w
    
    def reverse_solve(self, y, eta_t, i):
        w = torch.normal(0,1,(y.shape),device=self.device)
        beta_i = self.betas[i]
        const_beta_1 = 1. /torch.Tensor([(1-beta_i)**(0.5)])
        return const_beta_1*(y+beta_i*eta_t) + (beta_i)**(0.5)*w
    



if __name__ == "__main__":

    

    ########## Hyperparameters and settings ##########
    set_seed(42)
    num_epochs = 200
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    diff_logger = pl.loggers.TensorBoardLogger('lightning_logs', name='DDPM')
    vae_logger = pl.loggers.TensorBoardLogger('lightning_logs', name='VAE')
    clip_logger = pl.loggers.TensorBoardLogger('lightning_logs', name='CLIP')
    plots_dir= 'plots'
    d_latent = 16
    d_input=2
    num_classes= 8
    num_layers = 2
    mlp_ratio = 2.0
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
        vae = VAE(2,d_latent,beta=beta)
        trainer = pl.Trainer(max_epochs=num_epochs, accelerator=accelerator, log_every_n_steps=5, logger=vae_logger,gradient_clip_val=0.5,callbacks=[lr_monitor,EMA(decay=0.9999,validate_original_weights=True,every_n_steps=10)])# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
        trainer.fit(vae, train_dataloader, val_dataloader)
    else:
        vae = VAE.load_from_checkpoint(latent_encoding_checkpoint,d_in=d_input,d_latent=d_latent)

    ######## Extract the encoder and decoder ########
    
    image_encoder = vae.encoder
    image_decoder = vae.decoder

    ################## Train Label encoder ########

    if label_encoding_checkpoint is None:
        label_clip = LabelCLIP(vae.encoder, d_latent, num_classes)
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




