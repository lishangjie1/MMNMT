# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from fairseq.utils import print_r0
from typing import Callable, Dict, Tuple, Optional

import math
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F
from .share_mem import share_mem
from torch.nn.parameter import Parameter
import torch.nn.init as init
gumbel_map: Dict[torch.device, Callable] = {}
import logging
logger = logging.getLogger(__name__)
import torch.nn.functional as F
# logging
SAMPLE_FRACTION = 0.2


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype)
    output.scatter_(
        len(output.shape) - 1, indices, 1
    )
    return output


def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)


def top2gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    second_expert_policy='sampling',
    normalize_gate_prob_before_dropping=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    capacity_factor=1.0,
    batch_prioritized_routing=False,
    has_tutel=False,
    eom_dropout_module=None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    if has_tutel:
        from tutel import moe as tutel_moe
        fused_cumsum_sub_one=tutel_moe.fast_cumsum_sub_one
    else:
        fused_cumsum_sub_one=lambda mask: torch.cumsum(mask, dim=0) - 1
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    
    # language perception mask 
    # if lp_logits is not None:
    #     if num_updates is not None and moe_lang_perception_warmup is not None and num_updates < moe_lang_perception_warmup:
    #         mask_ratio = moe_lang_perception_ratio * num_updates / moe_lang_perception_warmup
    #     else:
    #         mask_ratio = moe_lang_perception_ratio
    #     lp_gates = F.softmax(lp_logits, dim=1)
    #     lp_mask = torch.ones_like(lp_gates)
    #     mask_num = int(lp_gates.shape[1] * mask_ratio)
    #     if mask_num > 0:
    #         lp_mask[(torch.arange(len(lp_gates)).unsqueeze(1), lp_gates.topk(mask_num, largest=False).indices)] = 0.0
    #     # recover lp_mask if gate value > threshold
    #     lp_mask[lp_gates > moe_lang_perception_outlier_threshold] = 1.0
    #     logits = logits.masked_fill(~lp_mask.bool(), float("-inf"))
    #     gates = F.softmax(logits, dim=1)
    #     # Straight through
    #     lp_mask = lp_mask - lp_gates.detach() + lp_gates
    #     gates = lp_mask * gates

    # else:
    gates = F.softmax(logits, dim=1)
        
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if eval_mode:
        if moe_eval_capacity_token_fraction > 0.0:
            capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
        else:
            # default moe_eval_capacity_token_fraction
            moe_eval_capacity_token_fraction = 0.75
            capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = int(2 * math.ceil(num_tokens / num_experts) * capacity_factor)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_experts)
    if second_expert_policy == 'sampling':
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    else:
        logits_w_noise = logits
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1, keepdim=True)
    mask2 = one_hot(indices2_s, num_experts)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)

    if normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s

    if second_expert_policy == 'random':
        sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
        mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

    # Compute locations in capacity buffer
    if input_mask is not None and input_mask.any():
        nonpadding = ~ input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)
        mask2 = mask2 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    if batch_prioritized_routing:
        # if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = fused_cumsum_sub_one(sorted_mask1) * sorted_mask1
        importance_sorted_locations1 =  sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]

        sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = fused_cumsum_sub_one(sorted_mask2) * sorted_mask2
        importance_sorted_locations2 =  sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = importance_sorted_locations1, importance_sorted_locations2
    else:
        locations1 = fused_cumsum_sub_one(mask1)
        locations2 = fused_cumsum_sub_one(mask2)
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts
    # balance load loss for language perception
    # if lp_logits is not None:
    #     me_lp = torch.mean(lp_gates, dim=0)
    #     indices1_lp = torch.argmax(lp_gates, dim=1, keepdim=True)
    #     mask_lp = one_hot(indices1_lp, num_experts)
    #     ce_lp = torch.mean(mask_lp.to(lp_gates.dtype), dim=0)
    #     l_aux_lp = torch.mean(me_lp * ce_lp)
    #     l_aux_lp = l_aux_lp * num_experts * num_experts
    #     l_aux += l_aux_lp

    # for logging purposes
    metadata["overflow_expert1"] = 100 * torch.sum(mask1 * torch.ge(locations1, capacity)) / torch.sum(mask1)
    metadata["overflow_expert2"] = 100 * torch.sum(mask2 * torch.ge(locations2, capacity)) / torch.sum(mask2)

    # from fairseq import utils
    # utils.print_r0("overflow_expert1: " + str(metadata["overflow_expert1"].detach().cpu().item()) + f", capacity: {capacity}, mask shape: {mask1.shape}, mask_sum: {torch.sum(mask1).detach().cpu().item()}, num_tokens: {num_tokens}, moe_eval_capacity_token_fraction: {moe_eval_capacity_token_fraction}, locations1: {locations1.max()}")
    # utils.print_r0("overflow_expert2: " + str(metadata["overflow_expert2"].detach().cpu().item()) + f", capacity: {capacity}, mask shape: {mask2.shape}, mask_sum: {torch.sum(mask2).detach().cpu().item()}, num_tokens: {num_tokens}, moe_eval_capacity_token_fraction: {moe_eval_capacity_token_fraction}, locations2: {locations2.max()}")


    # Remove locations outside capacity from mask
    mask1_, mask2_ = mask1, mask2
    mask1 = mask1 * torch.lt(locations1, capacity)
    mask2 = mask2 * torch.lt(locations2, capacity)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    expert2_hist = 100 * torch.histc((indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    expert2_hist = torch.sort(expert2_hist, dim=0, descending=True).values +  torch.finfo(torch.float32).tiny

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()


    metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()

    if not normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        gates1_s = (gates * mask1).sum(dim=1)
        gates2_s = (gates * mask2).sum(dim=1)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

    if has_tutel:
        locations1_s = torch.sum(locations1 * mask1_, dim=1)
        locations2_s = torch.sum(locations2 * mask2_, dim=1)
        return l_aux, metadata, capacity, num_experts, [indices1_s, indices2_s], [locations1_s, locations2_s], [gates1_s, gates2_s]

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    # EOM for gates1_s and gates2_s
    if eom_dropout_module:
        gates1_s = eom_dropout_module(gates1_s)
        gates2_s = eom_dropout_module(gates2_s)
    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    locations2_sc = one_hot(locations2_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1), locations1_sc.to(gates1.dtype).unsqueeze(1)
    )
    combine2_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates2.unsqueeze(-1), locations2_sc.to(gates2.dtype).unsqueeze(1)
    )
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    is_untouted=~(dispatch_mask.any(-1).any(-1))
    if input_mask is not None:
        metadata["unrouted_token_rate"]=(is_untouted.sum()-input_mask.sum())/(is_untouted.size(0)-input_mask.sum())
    else:
        metadata["unrouted_token_rate"]=is_untouted.sum()/is_untouted.size(0)

    if use_fp32:
        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine_weights, dispatch_mask, metadata

class MixGate(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.W_lang = torch.nn.Linear(dim, 1)
        self.W_tok = torch.nn.Linear(dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, h_lang, h_tok):
        return self.sigmoid(self.W_lang(h_lang) + self.W_tok(h_tok))

# class MultiGate(torch.nn.Module):
#     def __init__(self, gate_num: int, in_features: int, out_features: int, bias: bool = True,
#                  device=None, dtype=None) -> None:
        
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.gate_num = gate_num
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight_q = Parameter(torch.empty((gate_num, in_features, in_features), **factory_kwargs))
#         self.weight_v = Parameter(torch.empty((gate_num, out_features, in_features), **factory_kwargs))
        
#         if bias:
#             self.bias = Parameter(torch.empty(gate_num, out_features, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         for i in range(self.gate_num):
#             init.kaiming_uniform_(self.weight_q[i], a=math.sqrt(5))
#             init.kaiming_uniform_(self.weight_v[i], a=math.sqrt(5))
#             if self.bias is not None:
#                 fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_v[i])
#                 bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#                 init.uniform_(self.bias[i], -bound, bound)

#     def forward(self, input: Tensor) -> Tensor:
#         return F.linear(input, self.weight, self.bias)

#     def extra_repr(self) -> str:
#         return 'gate_num={}, in_features={}, out_features={}, bias={}'.format(
#             self.gate_num, self.in_features, self.out_features, self.bias is not None
#         )


class MixGate(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.W_lang = torch.nn.Linear(dim, 1)
        self.W_tok = torch.nn.Linear(dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, h_lang, h_tok):
        return self.sigmoid(self.W_lang(h_lang) + self.W_tok(h_tok))

class MultiGate(torch.nn.Module):
    def __init__(self, gate_num: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.gate_num = gate_num
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((gate_num, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(gate_num, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for i in range(self.gate_num):
            init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'gate_num={}, in_features={}, out_features={}, bias={}'.format(
            self.gate_num, self.in_features, self.out_features, self.bias is not None
        )
        
class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        is_encoder,
        use_fp32=False,
        second_expert_policy='sampling',
        normalize_gate_prob_before_dropping=False,
        moe_eval_capacity_token_fraction=0.25,
        batch_prioritized_routing=False,
        capacity_factor=1.0,
        moe_expert_output_masking=0.0,
        use_moe_lang_perception=False,
        lang_list=None,
    ) -> None:
        super().__init__()
        self.is_encoder = is_encoder
        if use_moe_lang_perception and not self.is_encoder:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False) # share gate
            assert lang_list is not None
            self.lang_list = lang_list # multi-gate following languages or language branch
            self.lang2id = dict([(lang, idx) for idx, lang in enumerate(lang_list)])
            self.id2lang = dict([(idx, lang) for idx, lang in enumerate(lang_list)])
            self.lang_id2branch_id = {}
            # group = ["af,da,de,en,fy,is,li,nb,nl,nn,no,sv,yi", 
            #         "br,cy,ga,gd", "be,bg,bs,cs,hr,mk,pl,ru,sk,sl,sr,uk,sh", 
            #         "lt,lv", 
            #         "as,bn,fa,gu,hi,ku,mr,ne,or,pa,ps,si,tg,ur", 
            #         "el", 
            #         "ca,es,fr,gl,it,oc,pt,ro,wa",
            #         "sq",
            #         "am,ar,ha,he,mt",
            #         "az,kk,ky,tk,tr,tt,ug,uz",
            #         "eo",
            #         "et,fi,hu,se",
            #         "eu",
            #         "id,mg,ms",
            #         "ig,rw,xh,zu",
            #         "ja",
            #         "ka",
            #         "km,vi",
            #         "kn,ml,ta,te",
            #         "ko",
            #         "my,zh",
            #         "th"]
            group = ['br,cy,ga,gd,et,fi,hu,se,id,mg,ms,az,kk,ky,tk,tr,tt,ug,uz,as,bn,fa,gu,hi,ku,mr,ne,or,pa,ps,si,tg,ur,kn,ml,ta,te', 
            'ka,be,bg,bs,cs,hr,mk,pl,ru,sk,sl,sr,uk,sh,am,ar,ha,he,mt', 
            'eo,af,da,de,en,fy,is,li,nb,nl,nn,no,sv,yi,ca,es,fr,gl,it,oc,pt,ro,wa', 
            'eu,sq,ig,rw,xh,zu,my,zh,ja,ko,th,km,vi,lt,lv,el']
            self.group_num = len(group)
            for branch_id, branch in enumerate(group):
                for lang in branch.split(','):
                    lang_id = self.lang2id[lang]
                    self.lang_id2branch_id[lang_id] = branch_id
            
            self.branch_wg = MultiGate(self.group_num, model_dim, num_experts, bias=False)
            
            
            # build hierarchy language family
            # 印欧语系 af,as,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,es,fa,fr,fy,ga,gd,gl,gu,hi,hr,is,it,ku,li,lt,lv,mk,mr,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,si,sk,sl,sq,sr,sv,tg,uk,ur,wa,yi
                # 日耳曼语族 af,da,de,en,fy,is,li,nb,nl,nn,no,sv,yi
                # 凯尔特语族 br,cy,ga,gd
                # 斯拉夫语族 be,bg,bs,cs,hr,mk,pl,ru,sk,sl,sr,uk,sh
                # 波罗的语族 lt,lv
                # 印度-伊朗语族 as,bn,fa,gu,hi,ku,mr,ne,or,pa,ps,si,tg,ur
                # 希腊语族 el
                # 罗曼语族 ca,es,fr,gl,it,oc,pt,ro,wa
                # 阿尔巴尼亚语族 sq
            # 亚非语系 am,ar,ha,he,mt
            # 突厥语族 az,kk,ky,tk,tr,tt,ug,uz
            # 人工语言 eo
            # 乌拉尔语系 et,fi,hu,se
            # 孤立语言 eu
            # 南岛语系 id,mg,ms
            # 尼日尔-刚果语系 ig,rw,xh,zu
            # 日本—琉球语系 ja
            # 南高加索语系 ka
            # 南亚语系 km,vi
            # 达罗毗荼语系 kn,ml,ta,te
            # 朝鲜语系 ko
            # 汉藏语系 my,zh
            # 壮侗语系 th

        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        # if use_moe_lang_perception:
        #     self.wg = torch.nn.Linear(2*model_dim, num_experts, bias=False)
        # else:
        #     self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.use_fp32 = use_fp32
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing = batch_prioritized_routing
        self.capacity_factor=capacity_factor
        self.eom_dropout_module = None
        if moe_expert_output_masking > 0.0:
            self.moe_expert_output_masking = moe_expert_output_masking
            from fairseq.modules import FairseqDropout
            self.eom_dropout_module = FairseqDropout(self.moe_expert_output_masking, module_name=self.__class__.__name__)
        self.use_moe_lang_perception = use_moe_lang_perception
    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def forward(self, input: torch.Tensor=None, mask: Optional[torch.Tensor] = None, has_tutel=False, logits:torch.Tensor=None, lang_embeddings:torch.Tensor=None, sentence_embeddings:torch.Tensor=None, batch_lang_ids=None) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        if logits is None:
            # if self.use_moe_lang_perception:
            #     assert lang_embeddings is not None
            #     input = torch.cat([input, lang_embeddings], dim=1)
            #     logits = self.wg(input)
            # else:
            #     logits = self.wg(input)
            if self.use_moe_lang_perception and not self.is_encoder:
                # assert batch_lang_ids is not None
                # import time
                # share_logits = self.wg(input)
                
                # batch_lang_ids_l = batch_lang_ids.cpu().detach().numpy().tolist()
                # branch_ids = [self.lang_id2branch_id[lang_id] for lang_id in batch_lang_ids_l]
                # unique_branch_ids = set(branch_ids)

                # branch_ids = torch.LongTensor(branch_ids).to(share_logits)

                # branch_logits = torch.zeros_like(share_logits)
                # for branch_id in unique_branch_ids:
                #     select_mask = (branch_ids == branch_id)
                #     input_selected = input[select_mask]

                #     weight_q = self.branch_wg.weight_q[branch_id] # 1024*1024
                #     weight_v = self.branch_wg.weight_v[branch_id] # 32 * 1024

                #     branch_input_s = input_selected.matmul(weight_q.T)
                #     branch_logits_s = branch_input_s.matmul(weight_v.T)

                #     # fill branch_logits
                #     branch_logits[select_mask] = branch_logits_s
                # logits = 0.5 * share_logits + 0.5 * branch_logits

                assert batch_lang_ids is not None
                batch_branch_ids = torch.zeros_like(batch_lang_ids)
                for lang_id in self.lang_id2branch_id.keys():
                    batch_branch_ids[ batch_lang_ids == lang_id ] = self.lang_id2branch_id[lang_id]
                weight = self.branch_wg.weight.index_select(dim=0, index=batch_branch_ids)
                
                #input = torch.cat([input, lang_embeddings], dim=1)
                branch_logits = weight.matmul(input.unsqueeze(-1)).squeeze(-1)
                share_logits = self.wg(input)
                logits = 0.5 * share_logits + 0.5 * branch_logits


                # assert batch_lang_ids is not None
                # import time
                # share_logits = self.wg(input)
                
                # batch_lang_ids_l = batch_lang_ids.cpu().detach().numpy().tolist()
                # unique_langs = [ self.id2lang[lang_id] for lang_id in set(batch_lang_ids_l) ]
                # branch_ids = [self.lang_id2branch_id[lang_id] for lang_id in batch_lang_ids_l]
                # unique_branch_ids = set(branch_ids)

                # branch_ids = torch.LongTensor(branch_ids).to(share_logits)

                # branch_logits = torch.zeros_like(share_logits)

                # #branch_routing = []
                # all_logits = []
                # for branch_id in range(self.group_num):
                #     weight = self.branch_wg.weight[branch_id] # 32 * 1024
                #     branch_logits_i = input.matmul(weight.T)

                #     all_logits.append(branch_logits_i)


                # loss = 0
                # for branch_id in range(self.group_num):
                #     select_mask = (branch_ids == branch_id)
                #     if len(select_mask) > 0:
                #         branch_logits_i = all_logits[branch_id]
                #         # fill branch_logits
                #         branch_logits[select_mask] = branch_logits_i[select_mask]

                    
                #         # calculate distance between different gate
                #         positive_logits = branch_logits_i[select_mask]
                #         positive_log_softmax = F.log_softmax(positive_logits.float(), dim=1)
                #         for i in range(self.group_num):
                #             if i == branch_id:
                #                 continue
                #             negtive_logits = all_logits[i][select_mask]
                #             negtive_log_softmax = F.log_softmax(negtive_logits.float(), dim=1)

                #             kl_loss = F.kl_div(negtive_log_softmax, positive_log_softmax, log_target=True, reduction='sum')

                #             loss += kl_loss
                
                # loss = loss / (self.group_num - 1) / len(branch_ids)

                # print(f"kl loss: {loss}")

                # logits = 0.5 * share_logits + 0.5 * branch_logits
            else:
                logits = self.wg(input)

        return top2gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            second_expert_policy=self.second_expert_policy,
            normalize_gate_prob_before_dropping=self.normalize_gate_prob_before_dropping,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            capacity_factor=self.capacity_factor,
            batch_prioritized_routing=self.batch_prioritized_routing,
            has_tutel=has_tutel,
            eom_dropout_module=self.eom_dropout_module
            )
