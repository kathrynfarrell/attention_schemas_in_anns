"""
Adapted from the open source code for "Emerging Properties in Self-Supervised Vision Transformers"
by Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

Inspired by the architecture described in "Attention Schema in Neural Agents"
by Dianbo Liu, Samuele Bolotta, He Zhu, Yoshua Bengio, and Guillaume Dumas
"""
import math
from functools import partial
from IPython import embed

import torch
from torch._C import _create_function_from_trace_with_dict
import torch.nn as nn
# from torchrl.data.replay_buffers.storages import tree_unflatten

from utils import trunc_normal_

import gumbel_max_pytorch

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def recompute(self, x, new_scores, v):
        B, N, C = x.shape
        x = (new_scores @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn, v

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VitAttentionSchema(nn.Module):
    def __init__(self, img_size=[256], patch_size=16, in_chans=3, num_classes=2, embed_dim=384, depth=6,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, policy_dim=1, **kwargs):

        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.evaluating_outputs = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # h2_dim the dimension of h2
        h2_dim = 200
        act_supp_dim = num_heads*(num_patches+1)

        # RNN returns (sequence length, hidden_size) tensor
        self.rnn = nn.RNN(embed_dim, h2_dim)
        mlp_hidden_dim = int(h2_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim)
        self.predictor = Mlp(in_features=h2_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim)

        self.activator = Mlp(in_features=h2_dim, hidden_features=mlp_hidden_dim, out_features=act_supp_dim)
        self.suppressor = Mlp(in_features=h2_dim, hidden_features=mlp_hidden_dim, out_features=act_supp_dim)


        self.norm1 = norm_layer(embed_dim)
        self.attention = Attention(embed_dim, num_heads=num_heads)
        self.norm2 = norm_layer(embed_dim)
        self.policy = nn.Linear(embed_dim, policy_dim)

    def forward(self, x):
        if not self.evaluating_outputs:
          x = self.prepare_tokens(x)
        h1, attn_score, v = self.attention(self.norm1(x))
        h1 = self.mlp(self.norm2(h1))
        h2, hidden_state = self.rnn(h1)
        B, heads, N, _ = attn_score.shape
        active = self.activator(h2).reshape(B, heads, N, N)
        suppressed = self.suppressor(h2).reshape(B, heads, N, N)
        active_suppressed = torch.stack((active, suppressed), dim=4)
        attn_mask = gumbel_max_pytorch.gumbel_softmax(active_suppressed, hard=True)
        new_scores = torch.mul(attn_score, attn_mask[:,:,:,:,0])
        h1m = self.attention.recompute(x, new_scores, v)
        pred_attn = self.predictor(h2)
        policy = self.policy(h1m[:, 0, :]) # use cls token for prediction label
        return pred_attn, h1m, policy

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x):
      x = self.prepare_tokens(x)
      h1, attn_score, v = self.attention(self.norm1(x))
      h1 = self.mlp(self.norm2(h1))
      h2, hidden_state = self.rnn(h1)
      B, heads, N, _ = attn_score.shape
      active = self.activator(h2).reshape(B, heads, N, N)
      suppressed = self.suppressor(h2).reshape(B, heads, N, N)
      active_suppressed = torch.stack((active, suppressed), dim=4)
      attn_mask = gumbel_max_pytorch.gumbel_softmax(active_suppressed, hard=True)
      new_scores = torch.mul(attn_score, attn_mask[:,:,:,:,0])
      # pred_attn = self.predictor(h2)
      return new_scores


class VitControl(nn.Module):
    def __init__(self, img_size=[256], patch_size=16, in_chans=3, num_classes=2, embed_dim=384, depth=6,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, policy_dim=1, **kwargs):

        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.evaluating_outputs = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.attention = Attention(embed_dim, num_heads=num_heads)
        self.norm2 = norm_layer(embed_dim)
        self.policy = nn.Linear(embed_dim, policy_dim)

    def forward(self, x):
        if not self.evaluating_outputs:
          x = self.prepare_tokens(x)
        h1, _, _ = self.attention(self.norm1(x))
        h1 = self.mlp(self.norm2(h1))
        policy = self.policy(h1[:, 0, :]) # use cls token for prediction label
        return h1, policy

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x):
      x = self.prepare_tokens(x)
      _, attn_score, _ = self.attention(self.norm1(x))
      return attn_score

# MULTI AGENT NETWORKS
# Vit Agents
class MultiAgentVit(nn.Module):
  def __init__(self, n_agents, n_agent_outputs, schema,
      img_size=[256], in_chans=3, critic=False):
    super().__init__()
    self.n_agents = n_agents
    self.critic = critic
    self.schema = schema

    if schema:
      self.networks = nn.ModuleList(
          [VitAttentionSchema(
            img_size=img_size,
            in_chans=in_chans,
            policy_dim=n_agent_outputs)

          for i in range(n_agents)]
          )
    else:
      self.networks = nn.ModuleList(
          [VitControl(
            img_size=img_size,
            in_chans=in_chans,
            policy_dim=n_agent_outputs)

          for i in range(n_agents)]
          )

  # the way we shape outputs/inputs depends on whether the common
  # module is working for the actor or the critic.

  def forward(self, *inputs):
    if len(inputs) == 1:
      inputs = inputs[0]
      if len(inputs.shape) > 4:
        inputs = inputs.squeeze()
    responses = []
    pred_attns = []
    h1ms = []
    # in: [n_envs, field + player channels, w, h]
    for i, net in enumerate(self.networks):

      if self.schema:
        pred_attn, h1m, response = net(inputs)
        pred_attns.append(pred_attn[:, 0, :])
        h1ms.append(h1m[:, 0, :])
      else:
        _, response = net(inputs)
      n_envs, output_dim = response.shape
      field_dim = int(math.sqrt(output_dim/2))
      response = response.reshape(n_envs,
          field_dim, field_dim*2)

      responses.append(response)

    if self.schema:
      # out: [n_envs, n_agents, field_dim, field_dim*2]
      # print("returning h1m "+str(torch.stack(h1ms, dim=1).shape)+", pred_attns "+str(torch.stack(pred_attns, dim=1).shape)+" responses " + str(torch.stack(responses, dim=1).shape))
      return torch.stack(h1ms, dim=1), torch.stack(pred_attns, dim=1), torch.stack(responses, dim=1)
    else:
      # control does not return predicted and actual attns
      return torch.stack(responses, dim=1)

# Mlp Agents
class MultiAgentMlp(nn.Module):
  def __init__(self, n_agents, n_agent_outputs, field_dim, n_envs=1, action_logits=False, from_observation=False):
    super().__init__()
    self.n_envs = n_envs
    self.n_agents = n_agents
    self.field_dim = field_dim
    self.action_logits = action_logits
    self.from_observation = from_observation
    if not from_observation:
      in_features = (2*field_dim**2)
    else:
      in_features = (field_dim**2)
    self.networks = nn.ModuleList(
        [Mlp(
          in_features=in_features,
          out_features=n_agent_outputs)

          for i in range(n_agents)]
          )

  def forward(self, *inputs):
    if len(inputs) == 1:
      inputs = inputs[0]
      # print("mlp sees "+str(inputs.shape))
      inputs = torch.flatten(inputs, start_dim=-2)
      # print("running mlp on "+str(inputs.shape))
    responses = []
    # in: [..., n_agents, flattened field info]
    for i, net in enumerate(self.networks):
      response = net(inputs[..., i, :])
      if self.action_logits:
        response = response.view(*response.shape[:-1], self.field_dim, self.field_dim,2)
      responses.append(response)
    # out: [n_envs, n_agents, 1, 1, 1] if not making action logits
    if not self.action_logits:
      output = torch.stack(responses, dim=-2).unsqueeze(-1).unsqueeze(-1)
      if len(output.shape) > 4:
        output = output.squeeze(-5)
    # [n_envs, n_agents, fd, fd, 2] for action logits
    else:
      output = torch.stack(responses, dim=1)
    # print("mlp returns "+str(output.shape))
    return output
