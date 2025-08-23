# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional
from einops import repeat
import math
import numpy as np

from Michelangelo.michelangelo.models.modules import checkpoint
from Michelangelo.michelangelo.models.modules.embedder import FourierEmbedder

from Michelangelo.michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from Michelangelo.michelangelo.models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer
)

from .tsal_base import ShapeAsLatentModule


class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 fourier_embedder_ID: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        # self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        ######
        voxel_reso = 4
        x_y = np.linspace(-8, 8, voxel_reso)   # 50
        z_res = np.linspace(-8, 8, voxel_reso) # 16
        xv, yv, zv = np.meshgrid(x_y, x_y, x_y, indexing='ij')
        voxel_centers = torch.tensor(np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T, device=device, dtype=dtype).reshape([-1,3])
        # dummy_tensor = torch.randn((voxel_centers.shape[0]+1, width), device=device, dtype=dtype)
        # dummy_tensor[:,:192] = voxel_centers.reshape([-1])

        dummy_tensor2 = torch.randn((num_latents, width), device=device, dtype=dtype)*0.02
        dummy_tensor2[:,:192] = voxel_centers.reshape([-1])*0.01                            
        self.query = nn.Parameter(dummy_tensor2)
        
        ######


        
        self.point_feats = point_feats
        self.fourier_embedder = fourier_embedder
        self.fourier_embedder_ID = fourier_embedder_ID

        #### PE_gama_xyz
        # self.input_proj = nn.Linear(self.fourier_embedder.out_dim+point_feats+self.fourier_embedder_ID.out_dim,width,device=device,dtype=dtype)
        
        ### no PE
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim+point_feats,width,device=device,dtype=dtype)

        ### PE_xyz
        # self.input_proj = nn.Linear(self.fourier_embedder.out_dim+point_feats+3,width,device=device,dtype=dtype)


        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 18]    0:3 -> voxel centers, 3 -> voxel ID, 4:7-> xyz, 7:18-> features
            feats (torch.FloatTensor or None): [B, N, 18]    0:3 -> voxel centers, 3 -> voxel ID, 4:7-> xyz, 7:18-> features

        Returns:

        """
        bs = pc.shape[0]
        voxel_ID = pc[:,:,3]
        voxel_coords = pc[:,:,:3]
        feats = feats[:,:,7:]

        ### voxel_coords embedding
        voxel_coords_emb = self.fourier_embedder_ID(voxel_coords) 

        ### voxel_ID embedding
        # voxel_ID_emb = self.fourier_embedder_ID(voxel_ID.unsqueeze(-1)) 
        
        data = self.fourier_embedder(pc[:,:,4:7])
       
        if feats is not None:
            # data = torch.cat([data, voxel_coords_emb, feats], dim=-1) # voxel_coords embedding   gamma(xyz)
            
            # data = torch.cat([data, voxel_ID_emb, feats], dim=-1) # voxel_ID embedding
    
            data = torch.cat([data, feats], dim=-1).to(dtype=torch.float32)  # 10_xyz_pe no pe
        data = self.input_proj(data) # data: [100, 16384, 1->51] [100, 16384, 11]
        query = repeat(self.query, "m c -> b m c", b=bs) # learnable queries

        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class CrossAttentionDecoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)

class GS_decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56):
        super(GS_decoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skip
        self.output_ch = output_ch
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])
        for i in range(D-1):
            self.pts_linears.append(nn.Linear(W, W))
            # self.pts_linears.append(nn.BatchNorm1d(W))
            self.pts_linears.append(nn.LayerNorm(W))
            # self.pts_linears.append(nn.InstanceNorm1d(W))
            self.pts_linears.append(nn.ReLU())
        
        self.output_linear = nn.Linear(in_features=W, out_features=output_ch)
        # self.bn_layer_output = nn.BatchNorm1d(output_ch)
        
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            # x = F.relu(x)
            # x = self.act(x)
        # x = self.bn_layer_output(self.output_linear(x))
  
        x = self.output_linear(x)
        return x

# class GS_decoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56):
#         super(GS_decoder, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch

#         # self.conv = nn.Conv3d(1, 14, 6, stride=1, padding=0)
#         # self.output_linear_0 = nn.Linear(in_features=64000, out_features=40000)
#         self.output_linear_1 = nn.Linear(in_features=14, out_features=14)
        
#         # self.bn_layer_output = nn.BatchNorm1d(output_ch)
        
#     def forward(self, x):
#         # x = self.conv(x.reshape([x.shape[0],1,40,40,40])).reshape([x.shape[0],14,-1]).transpose(2,1)
#         # x = self.output_linear_1(x)
#         x=x

       
#         return x


class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = True,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()
        # ##### learnable output query
        # self.learnable_output_queries = nn.Parameter(torch.randn(125,40000))
        
        self.use_checkpoint = use_checkpoint

        self.num_latents = num_latents
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi, input_dim=3)
        
        self.fourier_embedder_ID = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi, input_dim=3)
        # self.fourier_embedder_ID = FourierEmbedder(num_freqs=6, include_pi=include_pi, input_dim=1)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            fourier_embedder_ID=self.fourier_embedder_ID,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.embed_dim = embed_dim
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )
        # original: (3,1024,width*256,[4],14*16384)  now (3,1024,width*256,[4],14*40000)
        # self.GS_decoder = GS_decoder(3,1024,width*512,[4],16384*14)  # original
        
        self.GS_decoder = GS_decoder(3,1024,width*512,[4],40000*14)    # filtered

        
        self.kl_emb_proj_mean = nn.Linear((num_latents-1)*embed_dim, 64*64*4, dtype=dtype) 
        self.kl_emb_proj_var = nn.Linear((num_latents-1)*embed_dim, 64*64*4, dtype=dtype)
        
        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """
     
        latents, center_pos = self.encoder(pc, feats)

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()

        return latents, center_pos, posterior


    
    def decode(self, latents: torch.FloatTensor, volume_queries=None):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        #########
        # learn_volume_queries = (volume_queries.transpose(2,1)@self.learnable_output_queries).transpose(2,1)
        # logits = self.query_geometry(volume_queries, latents)
        # return self.GS_decoder(logits.reshape([logits.shape[0], volume_queries.shape[1],14]))
        #########
    
        return self.GS_decoder(latents.reshape(latents.shape[0],-1))

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """

        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)
        
        return logits, center_pos, posterior


class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = True,  ###### change here
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__(
            device=device,
            dtype=dtype,
            num_latents=1 + num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.width = width

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, c]
            sample_posterior (bool):

        Returns:
            shape_embed (torch.FloatTensor)
            kl_embed (torch.FloatTensor):
            posterior (DiagonalGaussianDistribution or None):
        """
        shape_embed, latents = self.encode_latents(pc, feats)
        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)
        kl_embed = kl_embed.reshape([kl_embed.shape[0],-1])
        mu, log_var = self.kl_emb_proj_mean(kl_embed), self.kl_emb_proj_var(kl_embed)
        std = torch.exp(0.5 * log_var).to(log_var.device)
        eps = torch.randn_like(std).to(std.device)
        z = mu + std * eps.to(std.device) 
        return shape_embed, mu, log_var, z, posterior
        # return shape_embed, posterior.mean.reshape([kl_embed.shape[0],-1]), posterior.logvar.reshape([kl_embed.shape[0],-1]), kl_embed.reshape([kl_embed.shape[0],-1]), posterior

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):

        x, _ = self.encoder(pc, feats)   # [B, 257, 384] [B, num_latents, width]
        shape_embed = x[:, 0]
        latents = x[:, 1:]

        return shape_embed, latents

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents

        return kl_embed, posterior

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            shape_embed (torch.FloatTensor): [B, projection_dim]
            logits (torch.FloatTensor): [B, M]
            posterior (DiagonalGaussianDistribution or None).

        """
        shape_embed, kl_embed, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(kl_embed)
        logits = self.query_geometry(volume_queries, latents)

        return shape_embed, logits, posterior
