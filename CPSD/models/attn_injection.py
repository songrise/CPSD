# -*- coding : utf-8 -*-
# @FileName  : attn_injection.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Mar 20, 2024
# @Github    : https://github.com/songrise
# @Description: implement attention dump and attention injection for CPSD

from __future__ import annotations

from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops
from diffusers.models import unet_2d_condition, attention, transformer_2d, resnet
from diffusers.models.unets import unet_2d_blocks

# from diffusers.models.unet_2d import CrossAttnUpBlock2D
from typing import Optional, List

T = torch.Tensor
import os


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = (True,)
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.0
    shared_score_shift: float = 0.0
    only_self_level: float = 0.0


def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        # self.processor = attention_processor.AttnProcessor2_0()
        self.processor = attention_processor.AttnProcessor()  # for torch 1.11.0

    def __call__(
        self,
        attn: attention_processor.Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        return self.processor(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )


class DumpAttentionProcessor(DefaultAttentionProcessor):
    def __init__(
        self,
        output_dir: str,
    ):
        super().__init__()
        self.output_dir = output_dir
        # the processor are inherited from the DefaultAttentionProcessor

    def __str__(self):
        return f"DumpAttn(output_dir={self.output_dir})"

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # dump here
        # if self.save_self_attention:
        query_save = query.clone().detach()
        key_save = key.clone().detach()
        value_save = value.clone().detach()
        torch.save(query_save, f"{self.output_dir}_query.pt")
        torch.save(key_save, f"{self.output_dir}_key.pt")
        torch.save(value_save, f"{self.output_dir}_value.pt")
        del query_save, key_save, value_save

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InjectAttentionProcessor(DefaultAttentionProcessor):
    def __init__(
        self,
        output_dir: str,
        inject_query: bool = False,
        inject_key: bool = False,
        inject_value: bool = False,
        injection_dir: str = "",
    ):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.inject_query = inject_query
        self.inject_key = inject_key
        self.inject_value = inject_value
        self.injection_dir = injection_dir
        with open(f"{self.injection_dir}_query.pt", "r") as f:
            self.query = torch.load(f)
        with open(f"{self.injection_dir}_key.pt", "r") as f:
            self.key = torch.load(os.path.join(f, "key.pt"))
        with open(f"{self.injection_dir}_value.pt", "r") as f:
            self.value = torch.load(os.path.join(f, "value.pt"))
        #!HARDCODED Mar 20: need init a processor for the self-attn in each of transformer block in upsampling path

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # inject here
        #! need debug
        batch_size = query.shape[0]
        if self.inject_query:
            query = self.query[:batch_size]
        if self.inject_key:
            key = self.key[:batch_size]
        if self.inject_value:
            value = self.value[:batch_size]

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PnPAttentionProcessor(DefaultAttentionProcessor):
    def __init__(
        self,
        inject_query: bool = True,
        inject_key: bool = True,
        inject_value: bool = True,
    ):
        super().__init__()

        self.inject_query = inject_query
        self.inject_key = inject_key
        self.inject_value = inject_value
        self.share_enabled = True
        #!HARDCODED Mar 20: need init a processor for the self-attn in each of transformer block in upsampling path

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        # inject here, swap the q k
        batch_size = query.shape[0] // 2  # 2 since CFG is used
        if self.share_enabled and batch_size > 1:  # when == 1, no need to inject,
            ref_q_uncond, ref_q_cond = query[0, ...].unsqueeze(0), query[
                1, ...
            ].unsqueeze(0)
            ref_k_uncond, ref_k_cond = key[0, ...].unsqueeze(0), key[1, ...].unsqueeze(
                0
            )
            # temp, also share the value
            ref_v_uncond, ref_v_cond = value[0, ...].unsqueeze(0), value[
                1, ...
            ].unsqueeze(0)
            if self.inject_query:
                query = torch.cat([ref_q_uncond, ref_q_cond] * batch_size, dim=0)
            if self.inject_key:
                key = torch.cat([ref_k_uncond, ref_k_cond] * batch_size, dim=0)
            if self.inject_value:
                value = torch.cat([ref_v_uncond, ref_v_cond] * batch_size, dim=0)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # inject here, swap the attention map

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SharedAttentionProcessor(DefaultAttentionProcessor):

    def __init__(self, style_aligned_args: StyleAlignedArgs):
        super().__init__()
        self.share_attention = style_aligned_args.share_attention
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.full_attention_share = style_aligned_args.full_attention_share
        self.shared_score_scale = style_aligned_args.shared_score_scale
        self.shared_score_shift = style_aligned_args.shared_score_shift

    def shifted_scaled_dot_product_attention(
        self, attn: attention_processor.Attention, query: T, key: T, value: T
    ) -> T:
        logits = torch.einsum("bhqd,bhkd->bhqk", query, key) * attn.scale
        logits[:, :, :, query.shape[2] :] += self.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum("bhqk,bhkd->bhqd", probs, value)

    def shared_call(
        self,
        attn: attention_processor.Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if self.step >= self.start_inject:
        if self.adain_queries:
            query = adain(query)
        if self.adain_keys:
            key = adain(key)
        if self.adain_values:
            value = adain(value)
        if self.share_attention:
            key = concat_first(key, -2, scale=self.shared_score_scale)
            value = concat_first(value, -2)
            if self.shared_score_shift != 0:
                hidden_states = self.shifted_scaled_dot_product_attention(
                    attn,
                    query,
                    key,
                    value,
                )
            else:
                hidden_states = nnf.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
        else:
            hidden_states = nnf.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        # hidden_states = adain(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call__(
        self,
        attn: attention_processor.Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        if self.full_attention_share:
            b, n, d = hidden_states.shape
            hidden_states = einops.rearrange(
                hidden_states, "(k b) n d -> k (b n) d", k=2
            )
            hidden_states = super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = einops.rearrange(
                hidden_states, "k (b n) d -> (k b) n d", n=n
            )
        else:
            hidden_states = self.shared_call(
                attn, hidden_states, hidden_states, attention_mask, **kwargs
            )

        return hidden_states


class SharedResBlockWrapper(nn.Module):
    def __init__(self, block: resnet.ResnetBlock2D):
        super().__init__()
        self.block = block
        self.output_scale_factor = self.block.output_scale_factor
        self.share_enabled = True

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ):
        if self.share_enabled:
            feat = self.block(input_tensor, temb, scale)
            batch_size = feat.shape[0] // 2
            if batch_size == 1:
                return feat

            feat_uncond, feat_cond = feat[0, ...].unsqueeze(0), feat[1, ...].unsqueeze(
                0
            )
            input_tensor = self.block.conv_shortcut(input_tensor)
            input_uncond, input_cond = input_tensor[0, ...].unsqueeze(0), input_tensor[
                1, ...
            ].unsqueeze(0)
            # since feat = (input + h) / scle
            feat_uncond, feat_cond = (
                feat_uncond * self.output_scale_factor,
                feat_cond * self.output_scale_factor,
            )
            h_uncond, h_cond = feat_uncond - input_uncond, feat_cond - input_cond
            # only share the h, the residual is not shared
            h_shared = torch.cat([h_uncond, h_cond] * batch_size, dim=0)
            output_shared = (input_tensor + h_shared) / self.output_scale_factor
            return output_shared
        else:
            return self.block(input_tensor, temb, scale)


def _get_switch_vec(total_num_layers, level):
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > 0.5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers)
    vec = vec % (total_num_layers // num_switch)
    vec = vec == 0
    if to_flip:
        vec = ~vec
    return vec


def register_attention_processors(
    unet: unet_2d_condition.UNet2DConditionModel,
    mode: str,  # dump or inject
    base_dir: str,
    share_resblock: bool = True,
    share_attn: bool = True,
    share_attn_layers: Optional[int] = None,
    share_resnet_layers: Optional[int] = None,
    share_query: bool = True,
    share_key: bool = True,
    share_value: bool = True,
):
    if mode == "dump":
        if os.path.exists(base_dir):
            print(f"Dump path ``{base_dir}'' already exists, will be overwritten")
        os.makedirs(base_dir, exist_ok=True)
    elif mode == "inject":
        assert os.path.exists(
            base_dir
        ), f"Specified injection path ``{base_dir}'' does not exist"

    up_blocks: List[unet_2d_blocks.CrossAttnUpBlock2D] = unet.up_blocks[
        1:
    ]  # skip the first block, which is UpBlock2D
    layer_idx_attn = 0
    layer_idx_resnet = 0
    for block in up_blocks:
        # each block should have 3 transformer layer
        #  transformer_layer : transformer_2d.Transformer2DModel
        if share_resblock:
            if share_resnet_layers is not None:
                resnet_wrappers = []
                resnets = block.resnets
                for resnet_block in resnets:
                    if layer_idx_resnet not in share_resnet_layers:
                        resnet_wrappers.append(resnet_block)
                    else:
                        resnet_wrappers.append(SharedResBlockWrapper(resnet_block))
                        print(f"Share resnet feature set for layer {layer_idx_resnet}")
                    layer_idx_resnet += 1
                block.resnets = nn.ModuleList(resnet_wrappers)
        if share_attn:
            for transformer_layer in block.attentions:
                transformer_block: attention.BasicTransformerBlock = (
                    transformer_layer.transformer_blocks[0]
                )
                self_attn: attention_processor.Attention = transformer_block.attn1
                cross_attn: attention_processor.Attention = transformer_block.attn2

                if mode == "dump":
                    dump_processor = DumpAttentionProcessor(
                        os.path.join(
                            base_dir, f"block_{block_idx}_layer_{selfattn_id}_sa"
                        )
                    )
                    self_attn.set_processor(dump_processor)
                    print(
                        f"Dump processor set for self-attention in layer {layer_idx_attn}"
                    )
                elif mode == "inject":
                    inject_prefix = os.path.join(
                        base_dir, f"block_{block_idx}_layer_{selfattn_id}_sa"
                    )
                    inject_processor = InjectAttentionProcessor(
                        inject_prefix,
                        inject_query=True,
                        inject_key=True,
                        inject_value=True,
                        injection_dir=inject_prefix,
                    )
                    self_attn.set_processor(inject_processor)
                    print(
                        f"Inject processor set for self-attention in layer {layer_idx_attn}"
                    )
                elif mode == "pnp":
                    if (
                        share_attn_layers is not None
                        and layer_idx_attn in share_attn_layers
                    ):
                        pnp_inject_processor = PnPAttentionProcessor(
                            inject_query=share_query,
                            inject_key=share_key,
                            inject_value=share_value or layer_idx_attn < 1,
                        )
                        self_attn.set_processor(pnp_inject_processor)
                        print(
                            f"Pnp inject processor set for self-attention in layer {layer_idx_attn}"
                        )
                elif mode == "style_aligned":
                    sa_args = StyleAlignedArgs(shared_score_shift=0.6)
                    shared_processor = SharedAttentionProcessor(sa_args)
                    self_attn.set_processor(shared_processor)
                    print(
                        f"Shared processor set for self-attention in block layer {layer_idx_attn}"
                    )
                layer_idx_attn += 1


def unset_attention_processors(
    unet: unet_2d_condition.UNet2DConditionModel,
    unset_share_attn: bool = False,
    unset_share_resblock: bool = False,
):
    up_blocks: List[unet_2d_blocks.CrossAttnUpBlock2D] = unet.up_blocks[
        1:
    ]  # skip the first block, which is UpBlock2D
    block_idx = 1
    layer_idx = 0
    for block in up_blocks:
        if unset_share_resblock:
            resnet_origs = []
            resnets = block.resnets
            for resnet_block in resnets:
                if isinstance(resnet_block, SharedResBlockWrapper):
                    resnet_origs.append(resnet_block.block)
                else:
                    resnet_origs.append(resnet_block)
            block.resnets = nn.ModuleList(resnet_origs)
        if unset_share_attn:
            for transformer_layer in block.attentions:
                layer_idx += 1
                transformer_block: attention.BasicTransformerBlock = (
                    transformer_layer.transformer_blocks[0]
                )
                self_attn: attention_processor.Attention = transformer_block.attn1
                cross_attn: attention_processor.Attention = transformer_block.attn2
                self_attn.set_processor(DefaultAttentionProcessor())
        block_idx += 1
        layer_idx = 0


def register_shared_norm(
    pipeline: StableDiffusionXLPipeline,
    share_group_norm: bool = True,
    share_layer_norm: bool = True,
):
    def register_norm_forward(
        norm_layer: nn.GroupNorm | nn.LayerNorm,
    ) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, "orig_forward"):
            setattr(norm_layer, "orig_forward", norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_
        return norm_layer

    def get_norm_layers(
        pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]
    ):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_["layer"].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_["group"].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {"group": [], "layer": []}
    get_norm_layers(pipeline.unet, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [
        register_norm_forward(layer) for layer in norm_layers["layer"]
    ]


class Handler:

    def register(
        self,
        style_aligned_args: StyleAlignedArgs,
    ):
        self.norm_layers = register_shared_norm(
            self.pipeline,
            style_aligned_args.share_group_norm,
            style_aligned_args.share_layer_norm,
        )
        init_attention_processors(self.pipeline, style_aligned_args)

    def remove(self):
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        init_attention_processors(self.pipeline, None)

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        self.norm_layers = []


if __name__ == "__main__":
    from transformers import CLIPTextModel, CLIPTokenizer, logging
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        PNDMScheduler,
        DDIMScheduler,
    )

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    model_key = "stabilityai/stable-diffusion-2-1-base"
    device = torch.device("cuda")
    unet = UNet2DConditionModel.from_pretrained(
        model_key, subfolder="unet", use_safetensors=True
    ).to(device)

    register_attention_processors(unet, "inject", "/root/autodl-tmp/CPSD/attn/debug")
