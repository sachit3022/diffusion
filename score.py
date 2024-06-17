import tqdm
import torch
from torch import nn as nn
import numpy as np
import  pytorch_lightning as pl
from utils import plot_samples,inverse_transform
import matplotlib.pyplot as plt

class ScoreBasedModelling(pl.LightningModule):

    def __init__(self, d_latent,diffusion_model,image_encoder,label_encoder,image_decoder,lr=3e-4,inverse_transform=None):
        super().__init__()

        self.model = diffusion_model
        self.image_encoder = image_encoder 
        self.label_encoder = label_encoder
        self.image_decoder = image_decoder

        self.classifier_strength=2.0
        self.non_y_class_conditional_token = 10
        self.non_s_class_conditional_token = 10
        self.N=1000
        self.tspan = (1.0e-4,0.02)
        self.register_buffer('t_list', torch.linspace(self.tspan[0], self.tspan[1], self.N))
        self.lr = lr
        self.scale = 1
        self.log_freq= self.trainer.max_epochs//10
        self.inverse_transform = inverse_transform if inverse_transform is not None else nn.Identity()
        
        self.d_latent = (d_latent,) if isinstance(d_latent, int) else d_latent
        self.plot_style = 'density' if isinstance(d_latent, int) else 'image'
        self.loss = nn.MSELoss(reduction='none')

        
        #freeze the image encoder and label encoder and image decoder
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
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
    def generate(self,N=1000,x_0=None,y=None,s=None):

        self.model.eval()
        curr_t = self.N-1
        
        if x_0 is not None:
            curr_t = self.N//10
            x_0 = self.image_encoder(x_0)#.chunk(2,dim=1)[0]*self.scale
            x_t = self.forward_solve(x_0, torch.ones((x_0.size(0),1),device=self.device,dtype=torch.long)*curr_t, type='exact')[0]
        else:
            x_t = torch.randn((N, *self.d_latent), requires_grad=False,device = self.device)
        
        y = torch.ones(x_t.size(0),device=self.device,dtype=torch.long)*self.non_y_class_conditional_token if y is None else y
        s = torch.ones(x_t.size(0),device=self.device,dtype=torch.long)*self.non_y_class_conditional_token if s is None else s
        
        x_t = self._generate(x_t, y, s,curr_t)
        return x_t

        
    def _generate(self,x_t, y, s,start_t):

        y_c = self.label_encoder(y,s)
        y_phi = self.label_encoder(torch.ones_like(y,device=self.device,dtype=torch.long)*self.non_y_class_conditional_token,torch.ones_like(s,device=self.device,dtype=torch.long)*self.non_s_class_conditional_token)
        t = (torch.ones((x_t.size(0),),device=self.device,dtype=torch.long)*start_t)
        for i in range(start_t,-1,-1):
            eta_t = (1+self.classifier_strength)*self.model(x_t,t,y_c) - self.classifier_strength*self.model(x_t,t,y_phi)
            x_t = self.reverse_solve(x_t, eta_t,i)
            t = t-1
        
        x_t = self.image_decoder(x_t/self.scale)
        x_t = self.inverse_transform(x_t)
        return x_t
    
    @torch.no_grad()
    def interpolate(self,x0,x1,y=None,s=None):
        self.model.eval()

        inpterpolation_time = self.N//10
        
        alpha = 0.5 

        x_00 = self.image_encoder(x0)#.chunk(2,dim=1)[0]*self.scale
        x_11 = self.image_encoder(x1)
        t = (torch.ones((x0.size(0),),device=self.device,dtype=torch.long)*inpterpolation_time).reshape(-1,1)
        x_t0 = self.forward_solve(x_00, t, type='exact')[0]
        x_t1 = self.forward_solve(x_11, t, type='exact')[0]
        
        x_t = alpha*x_t0 + (1-alpha)*x_t1 
        y = torch.ones(x_t.size(0),device=self.device,dtype=torch.long)*self.non_y_class_conditional_token if y is None else y
        s = torch.ones(x_t.size(0),device=self.device,dtype=torch.long)*self.non_y_class_conditional_token if s is None else s
        
        x_t = self._generate(x_t, y, s,inpterpolation_time)
        return x_t


    def training_step(self, batch, batch_idx):
        x0,y,s = batch['X'],batch['label'],batch['sensitive']
        x0 = self.image_encoder(x0)#.chunk(2,dim=1)[0]*self.scale
        
        
        #randomly make 20% of the labels 10
        y = torch.where(torch.rand(y.shape,device=self.device) < 0.2, torch.ones_like(y,device=self.device)*self.non_y_class_conditional_token, y)
        s = torch.where(torch.rand(s.shape,device=self.device) < 0.2, torch.ones_like(s,device=self.device)*self.non_s_class_conditional_token, s)
        
        y = self.label_encoder(y,s)

        t_index = torch.randint(0, self.N, (x0.shape[0],),device=self.device)
        xt,noise = self.forward_solve(x0, t_index.reshape(-1,1), type='exact')

        #score_x_t = self.score_p_x_t_given_x0(xt, x0, t_index.reshape(-1,1)) # if we want to use score matching
    
        
        l_t = torch.ones_like(t_index) #  other options: 1/self.betas[t_index] # self.one_minus_alpha_t_bar[t_index]
        noise_pred = self.model(xt, t_index,y) #noise_pred = noise_pred.clip(-1,1)


        l = ((self.loss(noise_pred.reshape(noise_pred.size(0),-1), noise.reshape(noise.size(0),-1)).mean(dim=-1)*l_t)/l_t.sum()).sum()
        
        self.log_dict({'train_loss': l}, prog_bar=True,on_epoch=True)

        return {'loss': l}
    
    def configure_optimizers(self):
        e_opt = torch.optim.AdamW( list(self.model.parameters()), lr=self.lr)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False
        }
        return [e_opt],[e_scheduler]

    def on_train_epoch_end(self):
        if (self.current_epoch+1) % (self.log_freq) == 0:
            x_gen = self.generate(N=100)
            plot_samples(x_gen.detach().cpu(),how=self.plot_style,title='generated_samples',save=True,show=False,filepath=f'inverse_generated_samples_{self.current_epoch}.png',ax=None)

class DDPMProcess(ScoreBasedModelling):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        betas =  self.linear_beta_schedule()
        alpha_t = 1 -  betas
        alpha_t_bar = torch.cumprod(alpha_t, 0)


        sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar)
        one_minus_alpha_t_bar = 1. - alpha_t_bar
        

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_t', alpha_t)
        self.register_buffer('sqrt_alpha_t_bar', sqrt_alpha_t_bar)
        self.register_buffer('one_minus_alpha_t_bar', one_minus_alpha_t_bar)
    
    def constant_beta_schedule(self):
        return torch.ones((self.N,))* 0.02 
    
    def linear_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
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
        xt = xt.reshape(xt.size(0),-1)
        x0 = x0.reshape(x0.size(0),-1)
        alpha_bar = self.sqrt_alpha_t_bar[t_list]
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_list]
        return ((1/alpha_bar)*(xt - one_minus_alpha_t_bar**(0.5)*x0)).reshape(xt.size(0),*self.d_latent)

    def forward_solve(self, u0, t_list, type='exact'):
        
        u0 = u0.reshape(u0.size(0),-1)

        w = torch.randn(u0.shape,device=self.device)
        alpha_bar = self.sqrt_alpha_t_bar[t_list]
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_list]
        return  (alpha_bar * u0 + one_minus_alpha_t_bar**(0.5) * w).reshape(u0.size(0),*self.d_latent), w.reshape(u0.size(0),*self.d_latent)
    
    def noise_add(self, x_t, t_index, w):
        x_t = x_t.reshape(x_t.size(0),-1)
        w  = w.reshape(w.size(0),-1)
        alpha_bar = self.sqrt_alpha_t_bar[t_index]
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[t_index]**(0.5)
        return ((x_t - one_minus_alpha_t_bar*w)/(alpha_bar)).reshape(x_t.size(0),*self.d_latent)
    
    def score_reverse_solve(self, y, eta_t, i):
        w = torch.normal(0,1,(y.shape),device=self.device)
        beta_i = self.betas[i]
        const_beta_1 = 1. /torch.Tensor([(1-beta_i)**(0.5)])
        return const_beta_1*(y+beta_i*eta_t) + (beta_i)**(0.5)*w
    
    def reverse_solve(self, x_t, eta_t, i):
        x_t = x_t.reshape(x_t.size(0),-1)
        eta_t = eta_t.reshape(eta_t.size(0),-1)
        one_minus_alpha_t_bar = self.one_minus_alpha_t_bar[i]
        if i != 0:
            w = torch.normal(0,1,(x_t.shape),device=self.device)
        else:
            w = torch.normal(0,1,(x_t.shape),device=self.device)*0

        return ((1/self.alpha_t[i])**(0.5)*(x_t - self.betas[i]*eta_t/(one_minus_alpha_t_bar**(0.5))) + (self.betas[i])**(0.5)*w).reshape(x_t.size(0),*self.d_latent)


