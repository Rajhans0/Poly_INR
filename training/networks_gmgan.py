
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils.ops import bias_act
from torch_utils.ops import filtered_lrelu
from torch_utils import persistence
from torch_utils import misc
import numpy as np
import pickle
import dnnlib
import legacy

SLOPE = 0.2
#dummy layer norm
class LayerNorm(nn.Module):
    def __init__(self, df):
        super(LayerNorm, self).__init__()
    def forward(self, x):       
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_planes, planes)

    def forward(self, x):
        out = F.leaky_relu((self.fc1(x)), negative_slope=SLOPE)
        out  = out.clamp(min=-256, max= 256)
        return out
class BasicBlockGM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(BasicBlockGM, self).__init__()
        self.z_dim = planes
        self.fb0 =   BasicBlock(self.z_dim, self.z_dim)
        self.fc0_1 = nn.Linear(self.z_dim, 2*self.z_dim)
        self.fc0_2 = nn.Linear(self.z_dim, self.z_dim)
        self.fc0_3 = nn.Linear(self.z_dim, 2*2)
        self.fc0_4 = nn.Linear(self.z_dim, 2)
        self.fc0_5 = FullyConnectedLayer(in_planes, self.z_dim, activation='lrelu', lr_multiplier=1.0)

    def forward(self, x, xy1, grid, gx):
        xy1 = self.fc0_5(xy1.squeeze(1))
        xy_b =  self.fc0_2(xy1).view(-1, 1,self.z_dim)
        xy_ = self.fc0_1(xy1).view(-1, 2, self.z_dim)
        xy_b1 =  self.fc0_4(xy1).view(-1, 1,2)
        xy_1 = self.fc0_3(xy1).view(-1,2 , 2)
        grid = torch.matmul( grid, xy_1) + xy_b1
        g1 = torch.matmul(grid, xy_) + xy_b        
        x =  x*F.leaky_relu(g1, negative_slope=SLOPE)
        x = self.fb0(x)
        return x, grid, g1



@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = .1,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
        rand_embedding = False,     # Use random weights for class embedding
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # additions
        embed_path = 'in_embeddings/tf_efficientnet_lite0.pkl'
        with open(embed_path, 'rb') as f:
            self.embed = pickle.Unpickler(f).load()['embed']
        print(f'loaded imagenet embeddings from {embed_path}: {self.embed}')
        if rand_embedding:
            self.embed.__init__(num_embeddings=self.embed.num_embeddings, embedding_dim=self.embed.embedding_dim)
            print(f'initialized embeddings with random weights')

        # Construct layers.
        self.embed_proj = FullyConnectedLayer(self.embed.embedding_dim, self.z_dim, activation='lrelu') if self.c_dim > 0 else None

        features = [self.z_dim + (self.z_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers

        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if self.c_dim > 0:
            self.register_buffer('w_avg', torch.zeros([self.c_dim, w_dim]))
        else:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1.0, truncation_cutoff=None, update_emas=False):
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed_proj(self.embed(c.argmax(1)))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            # Track class-wise center
            if self.c_dim > 0:
                for i, label in enumerate(c.argmax(1)):
                    self.w_avg[label].copy_(x[i].detach().lerp(self.w_avg[label], self.w_avg_beta))
            else:
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        if truncation_psi != 1:
            if self.c_dim > 0:
                for i, label in enumerate(c.argmax(1)):
                    x[i, :truncation_cutoff] = self.w_avg[label].lerp(x[i, :truncation_cutoff], truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        #print(self.w_avg.shape, self.num_ws, self.w_dim)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------
class GMganSynthesis(nn.Module):
    def __init__(self, grid, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.grid = grid
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.z_dim1 = 1024
        self.nc = nc

        #  lelvel-0
        self.fc0_1 = nn.Linear(self.z_dim, 2*self.z_dim1)
        self.fc0_2 = nn.Linear(self.z_dim, self.z_dim1)
        self.fc0_0 = nn.Linear(self.z_dim1, self.z_dim1)
        self.fc0_3 = nn.Linear(self.z_dim, 2*2)
        self.fc0_4 = nn.Linear(self.z_dim, 2)
        # #  lelvel-1
        self.fb1 =  BasicBlockGM(self.z_dim, self.z_dim1)        

        # #  lelvel-2
        self.fb2 =  BasicBlockGM(self.z_dim, self.z_dim1) 

        # #  lelvel-3
        self.fb3 =  BasicBlockGM(self.z_dim, self.z_dim1) 

        # #  lelvel-4
        self.fb4 =  BasicBlockGM(self.z_dim, self.z_dim1)
        self.fb5 =  BasicBlockGM(self.z_dim, self.z_dim1)
        self.fb6 =  BasicBlockGM(self.z_dim, self.z_dim1)
        self.fb7 =  BasicBlockGM(self.z_dim, self.z_dim1)
        self.fb8 =  BasicBlockGM(self.z_dim, self.z_dim1)
        self.fb9 =  BasicBlockGM(self.z_dim, self.z_dim1)  
        #convert to image, one can replace this 1x1 conv layer with a linear layer
        self.to_rgb =  nn.Conv2d(self.z_dim1, nc, 1, 1, 0, bias=True)

 

    def forward(self, input, c=None,  **kwargs):
   
        grid = self.grid
        f = input.split(1, dim=1)
        xy1 = f[0]
        xy_b =  (self.fc0_2(xy1)).view(-1, 1,self.z_dim1)
        xy_ = (self.fc0_1(xy1)).view(-1, 2, self.z_dim1)
        xy_b1 =  self.fc0_4(xy1).view(-1, 1,2)
        xy_1 = self.fc0_3(xy1).view(-1,2 , 2)
        grid = torch.matmul( grid, xy_1) + xy_b1
        g1 = torch.matmul(grid, xy_) + xy_b
        x = F.leaky_relu((self.fc0_0(g1)), negative_slope=SLOPE)
        x = x.clamp(min=-256, max=256)

        x, grid, g1 = self.fb1(x,f[1], grid, x)
        x, grid, g1 = self.fb2(x,f[2], grid, g1)
        x, grid, g1 = self.fb3(x,f[3], grid, g1)
        x, grid, g1 = self.fb4(x,f[4], grid, g1)
        x, grid, g1 = self.fb5(x,f[5], grid, g1)
        x, grid, g1 = self.fb6(x,f[6], grid, g1)
        x, grid, g1 = self.fb7(x,f[7], grid, g1)
        x, grid, g1 = self.fb8(x,f[8], grid, g1)
        x, grid, g1 = self.fb9(x,f[9], grid, g1)    
   
        x = x.transpose(1,2).view(-1, self.z_dim1, self.img_resolution, self.img_resolution)
        x = self.to_rgb(x)
     
        return x



class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={},
        **kwargs,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.hw= 32
        self.img_channels = img_channels

        h = (self.hw-1)
        a = (torch.Tensor(range(self.hw)))/(h)
        g = torch.meshgrid(a, a)
        g = torch.cat((g[0].view(1, self.hw,self.hw, 1), g[1].view(1, self.hw,self.hw, 1),),dim=3)
        self.gridt = nn.Parameter(g.view(1, self.hw*self.hw, 2), requires_grad=False)

        #print(mapping_kwargs.num_ws)
        # Mapping and Synthesis Networks
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=512, num_ws=10, **mapping_kwargs)
        #DummyMapping()  # to fit the StyleGAN API
        Synthesis =  GMganSynthesis
        #print(cond, Synthesis)
        self.synthesis = Synthesis(self.gridt, z_dim=512, nc=img_channels, img_resolution=img_resolution, **synthesis_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        w = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(w, c)
        return img

class SuperresGenerator(nn.Module):
    def __init__(
        self,
        path_stem,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={},
        **kwargs,
    ):
        super().__init__()

        with dnnlib.util.open_url(path_stem) as f:
            G_stem = legacy.load_network_pkl(f)['G_ema']
        self.mapping = G_stem.mapping
        self.synthesis = G_stem.synthesis

        self.z_dim = G_stem.z_dim
        self.c_dim = G_stem.c_dim
        self.w_dim = G_stem.w_dim
        self.hw= img_resolution
        self.img_resolution = img_resolution
        self.img_channels = G_stem.img_channels

        h = (self.hw-1)
        a = (torch.Tensor(range(self.hw)))/(h)
        g = torch.meshgrid(a, a)
        g = torch.cat((g[0].view(1, self.hw,self.hw, 1), g[1].view(1, self.hw,self.hw, 1),),dim=3)
        self.gridt = nn.Parameter(g.view(1, self.hw*self.hw, 2), requires_grad=False)
        self.synthesis.grid = self.gridt
        self.synthesis.img_resolution = img_resolution


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        w = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(w, c)
        return img
###test###
# 