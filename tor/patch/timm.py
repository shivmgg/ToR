# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tor.merge import bipartite_soft_matching, merge_source, merge_wavg
from tor.utils import parse_r, parse_keep_rate, complement_idx
import math

class ToRBlock(Block):
    """
    Modifications:
     - Apply tor between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, tokens=None, token_fusion=False, get_idx=False) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        keep_rate = self._tor_info["keep_rate"].pop(0)
        token_fusion = self._tor_info["token_fusion"]
        attn_size = self._tor_info["size"] if self._tor_info["prop_attn"] else None
        B, N, C = x.shape
        x_attn, metric, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), attn_size, keep_rate)
        x = x + self._drop_path1(x_attn)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if token_fusion == True:
                ##if token-fusion is enabled. 
                remaining_tokens = math.ceil((1.0) * (N - left_tokens - 1))
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                compl = compl[:, :remaining_tokens]
                non_topk_x = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
                # non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                # extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]                
                # x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)

                # remaining_tokens_index = idx.unsqueeze(-1).expand(-1, -1, metric.shape[-1])
                # remaining_tokens_metric = torch.gather(metric, dim=1, index=remaining_tokens_index)  # [B, left_tokens, C // num_heads]                
                pruned_tokens_metric = torch.gather(metric, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, metric.shape[-1]))  # [B, N-1-left_tokens, C // num_heads]                   
                # fused_token_metric = torch.mean(pruned_tokens_metric, dim=1, keepdim=True)  # [B, 1, C]
                # metric = torch.cat([metric[:, 0:1], remaining_tokens_metric, fused_token_metric], dim=1)

            else:
                #if token-fusion is disabled. 
                x = torch.cat([x[:, 0:1], x_others], dim=1)
                remaining_tokens_index = idx.unsqueeze(-1).expand(-1, -1, metric.shape[-1])
                remaining_tokens_metric = torch.gather(metric, dim=1, index=remaining_tokens_index)  # [B, left_tokens, C // num_heads]
                metric = torch.cat([metric[:, 0:1], remaining_tokens_metric], dim=1)

            r = self._tor_info["r"].pop(0)
            if r > 0:
                # Apply tor here
                merge, _ = bipartite_soft_matching(
                    pruned_tokens_metric,
                    r,
                    class_token = False,
                    distill_token = False,
                )
                #print("merging:", merge.shape, x.shape)
                if self._tor_info["trace_source"]:
                    self._tor_info["source"] = merge_source(
                        merge, non_topk_x, self._tor_info["source"]
                    )
                non_topk_x, self._tor_info["size"] = merge_wavg(merge, non_topk_x, None)
                x = torch.cat([x[:, 0:1], x_others, non_topk_x], dim=1)
                self._tor_info["size"] = torch.ones_like(x[..., 0, None])

            # print("after merging:", x.shape)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx
        
        return x


class ToRAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None, keep_rate=None, tokens=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        if keep_rate is None:
            keep_rate = 1.0 
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(size)
        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        if keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, k.mean(1), None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
            return x, k.mean(1), index, idx, cls_attn, left_tokens
        
        # Return k as well here
        return x, k.mean(1), None, None, None, left_tokens


def make_tor_class(transformer_class):
    class ToRVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tor_info["r"] = parse_r(len(self.blocks), self.r)
            self._tor_info["drop_loc"] = self.drop_loc
            self._tor_info["token_fusion"] = self.token_fusion
            self._tor_info["keep_rate"] = parse_keep_rate(len(self.blocks), self.keep_rate, self.drop_loc) 
            self._tor_info["size"] = None
            self._tor_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToRVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies tor to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tor_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToRVisionTransformer = make_tor_class(model.__class__)

    model.__class__ = ToRVisionTransformer
    model.r = 0
    model.keep_rate = 1.0
    model.drop_loc = []
    model.token_fusion = True
    model._tor_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "keep_rate": model.keep_rate,
        "drop_loc": model.drop_loc,
        "token_fusion": model.token_fusion,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tor_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToRBlock
            module._tor_info = model._tor_info
        elif isinstance(module, Attention):
            module.__class__ = ToRAttention
