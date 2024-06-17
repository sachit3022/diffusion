import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import pairwise
from models.utils import weights_init,TimestepEmbedder
import torch
from diffusers import UNet2DModel

import math

def initialize_weights(m):
    for name, param in m.named_parameters():
        if isinstance(m, nn.Conv2d):
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        elif isinstance(m, nn.Linear):
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        elif isinstance(m, nn.BatchNorm2d):
            if 'weight' in name:
                torch.nn.init.ones_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        #tansposed convolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.conv = torch.nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2,stride=2)

    def forward(self, x):
        #x = self.conv(x)
        return self.up(x) 

    
    
class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x = self.maxpool(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels,out_channels,dropout=0.1,cond=False):
        """"
        Replicating the Resnet Block. 
        Motivation to replicate is that this is same as regression task and resnet units have proved powerful in regression tasks.
        We keep it as it is.
        """
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        initialize_weights(self)

    def forward(self, x,y=None):
        return self.conv_layer(x)




class UnetEncoder(nn.Module):
    def __init__(self, image_shape, channels = [4,8,16], dropout=0.1):
        super().__init__()
        C,H,W = image_shape #HXWXC image
        assert  H == W, "The input image should be square"
    
        input_conv = ResnetBlock(3,channels[0],dropout,cond=False)
        modules = [input_conv]
        for ch in pairwise(channels):
            in_channels,out_channels = ch
            modules.append(Downsample())
            modules.append(ResnetBlock(in_channels,out_channels,dropout,cond=False))
        output_conv = ResnetBlock(channels[-1],channels[-1]*2,dropout,cond=False)

        modules.append(output_conv)
        self.model = nn.Sequential(*modules)
            
    def forward(self, x):
        return self.model(x)
        
class UnetDecoder(nn.Module):
    def __init__(self, image_shape, channels = [16,8,3], dropout=0.1):
        super().__init__()
        C,H,W = image_shape #HXWXC image
        assert  H == W, "The input image should be square"
        #7X7X12 -> 14X14X6 -> 28X28X3
        input_conv = ResnetBlock(channels[0],channels[0],dropout)
        modules = [input_conv]
        for ch in pairwise(channels):
            in_channels,out_channels = ch
            modules.append(Upsample())
            modules.append(ResnetBlock(in_channels,out_channels,dropout))
        output_conv = ResnetBlock(channels[-1],channels[-1],dropout,cond=False)
        modules.append(output_conv)
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)



class TransformerBlock(nn.Module):
    def __init__(self,channels,num_heads,mlp_mult) -> None:
        super().__init__()

        # self.self_attention = nn.MultiheadAttention(channels,num_heads,batch_first=True)
        # self.norm = nn.LayerNorm(channels)
        # self.mlp1= nn.Sequential(
        #     nn.Linear(channels,mlp_mult*channels),
        #     nn.ReLU(),
        #     nn.Linear(mlp_mult*channels,channels)
        # )
        self.cross_attention = nn.MultiheadAttention(channels,num_heads,batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp2= nn.Sequential(
            nn.Linear(channels,mlp_mult*channels),
            nn.ReLU(),
            nn.Linear(mlp_mult*channels,channels)
        )
        initialize_weights(self)
    
    def forward(self,x,cond):
        """
        x : B X C X H X W
        cond : B X C X H X W
        """
        #make x to be B X (H X W) X C
        B,C,H,W = x.size()
        x = x.permute(0,2,3,1)
        x = x.view(x.size(0),-1,x.size(-1))
        # x = x + self.self_attention(x,x,x)[0]
        # x = self.norm(x)
        # x = self.mlp1(x)
        x = x + self.cross_attention(x,cond,cond)[0]
        x = self.norm2(x)
        x = x + self.mlp2(x)
        x = x.view(B,C,H,W)
        return x

class ChannelMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )
        self.net.apply(weights_init)
    def forward(self, x):
        return self.net(x)
   
class ConditionalUnet(nn.Module):
    def __init__(self,d_latent,dropout=0.1):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=28, # example size of input images
            in_channels=3, # number of input channels (e.g., 3 for RGB images)
            out_channels=3, # number of output channels
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            dropout=0.1,
            attention_head_dim = 8,
            class_embed_type= "identity",
            norm_num_groups=16,
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
        )
    def forward(self, x,t,y=None):
        #x = F.pad(x, (1, 0, 1, 0), mode='constant', value=0)
        x = self.model(x,t,y).sample
        #x = x[:,:,1:,1:]
        return x
        


"""

class ConditionalUnet(nn.Module):
    def __init__(self,d_latent,dropout=0.1):
        super().__init__()
        D = d_latent[0]
        C = d_latent[1]* d_latent[2]
        self.time_embedding = TimestepEmbedder(D)
        #7x7x16 (pad)(I) -> 8x8x64(x1) -> 4x4x128(x2) -> 2x2x256(x3) -> 4x4x128(x4) -> 8x8x64(x5) ->(crop) 7x7x16(O)
        #-> ResnetBlock + Attention Block
        #-> Down/up




        self.i_block = ResnetBlock(3,64,dropout)

        self.x1_block = ResnetBlock(64,64,dropout,cond=True)
        self.x1_cross = TransformerBlock(64,8,2)
        self.x1_down = Downsample()
        
        self.x2_block = ResnetBlock(64,128,dropout,cond=True)
        self.x2_cross = TransformerBlock(128,8,2)
        self.x2_down = Downsample()
        
        self.x3_block = ResnetBlock(128,256,dropout,cond=True)
        self.x3_cross = TransformerBlock(256,8,2)
        
        self.x4_block = ResnetBlock(256,128,dropout,cond=True)
        self.x4_cross = TransformerBlock(128,8,2)
        
        self.x4_up = Upsample()
        
        self.x5_block = ResnetBlock(128*2,64,dropout,cond=True)
        self.x5_cross = TransformerBlock(64,8,2)

        self.x5_up = Upsample()

        self.x6_block = ResnetBlock(64*2,64,dropout,cond=True)
        self.x6_cross = TransformerBlock(64,8,2)
        
        self.o_block = ResnetBlock(64,16,dropout,cond=True)

        mul = 1
        self.time1 = ChannelMLP(mul*D,64)
        self.time2 = ChannelMLP(mul*D,128)
        self.time3 = ChannelMLP(mul*D,256)
        self.time4 = ChannelMLP(mul*D,128)
        self.time5 = ChannelMLP(mul*D,64)
        self.time6 = ChannelMLP(mul*D,64)


        
    def forward(self, x,t):

        time_embeds = self.time_embedding(t)
        #y = y.permute(0,2,3,1).view(y.size(0),-1,y.size(1))
        time_embeds = time_embeds.unsqueeze(1).repeat(1,16,1)
        #cond_inp = torch.cat([time_embeds,y], dim=-1) # shape: BxLx2XD
        cond_inp = time_embeds
        
        x = F.pad(x, (1, 0, 1, 0), mode='constant', value=0)
        
        x_i = self.i_block(x) # 64x 8x8

        x_1 = self.x1_block(x_i)# 64x 8x8
        cond_inp_1 = self.time1(cond_inp)
        x_1 = self.x1_cross(x_1,cond_inp_1)
        
        x_1d = self.x1_down(x_1)# 128 x 4x4
        #print("x_1d",x_1d.size())
        x_2 = self.x2_block(x_1d)# 128 x 4x4
        cond_inp_2 = self.time2(cond_inp)
        x_2 = self.x2_cross(x_2,cond_inp_2)

        x_2d = self.x2_down(x_2)# 256 x 2x2
        #print("x_2d",x_2d.size())
        x_3 = self.x3_block(x_2d)# 256 x 2x2
        cond_inp_3 = self.time3(cond_inp)
        x_3 = self.x3_cross(x_3,cond_inp_3)

        x_4 = self.x4_block(x_3)
        cond_inp_4 = self.time4(cond_inp)
        x_4 = self.x4_cross(x_4,cond_inp_4)
        
        
        x_4u = self.x4_up(x_4) # 128 x 4x4
        
        #print("x_4u",x_4u.size())
        x_4_c = torch.cat([x_4u,x_2],dim=1) # 64x 8x8
        x_5 = self.x5_block(x_4_c)# 64x 8x8
        cond_inp_5 = self.time5(cond_inp)
        x_5 = self.x5_cross(x_5,cond_inp_5)

        x_5u = self.x5_up(x_5) # 64x 8x8
        #print("x_5u",x_5u.size())

        x_5_c = torch.cat([x_5u,x_1],dim=1) # 64x 8x8
        x_6 = self.x6_block(x_5_c)
        cond_inp_6 = self.time6(cond_inp)
        x_6 = self.x6_cross(x_6,cond_inp_6)


        x_o = self.o_block(x_6)
        #crop( last 2 diamens by 1 and 1)64x 8x8
        x_o = x_o[:,:,1:,1:]

        return x_o
""" 