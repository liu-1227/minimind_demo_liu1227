#定义了一个用于 Hugging Face Transformers 库的模型配置类
from transformers import PretrainedConfig

#通过继承pretrainconfig，定义自己的模型参数，并传到huggingface
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"# 模型类型标识符


    def __init__(
        self,

        ######特殊token id#######
        bos_token_id: int = 1,#句子开始标记
        eos_token_id: int = 2,#句子结束标记

        ######基础transformer参数#######
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 32768,
        dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = None,#FFN中间层维度（4*4）
        rms_norm_eps: float = 1e-05,#归一化层的数值稳定常数

        ######GQA（分组查询注意力）#######
        num_key_value_heads: int = 2,

        ######RoPE（旋转位置编码）#######
        vocab_size: int = 6400,
        rope_theta: int = 1000000,#基础频率
        inference_rope_scaling: bool = False,#长度扩展
        flash_attention: bool = True,#动态扩展

        ############ MoE（混合专家） ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,#堵在均衡损失权重
        seq_aux: bool = True,#序列级辅助损失
        norm_topk_prob: bool = True,#归一化 top-k 概率
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn

#继承nn.Module类
class RMSNorm(nn.Module):
#__init__初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
#_norm
    def norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim = True)+self.eps)
#forward
    def forward(self, x):
        return self.weight * self.norm(x.float()).type_as(x)

import math
from typing import Optional
def preconpute_freqs_cis(dim:int,
                         end:int(32*1024),
                         rope_base,#RoPE基础频率
                         rope_scaling:Optional[dict]=None):#缩放配置（YaRN）
    #计算RoPE频率（θ_i = base^{-2i/d}）
    freqs = 1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))
    #初始化注意力缩放因子
    attn_factor = 1.0

    #YaRN频率修改
    #rope_scaling不是空，即用户传入了YaRN配置
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow = (
            rope_scaling["original_max_position_embeddings"],#原始训练长度L
            rope_scaling["factor"],#扩展倍数
            rope_scaling["beta_fast"],#高频边界波长
            rope_scaling["beta_slow"],#低频边界波长
        )
        #推断长度大于训练长度，使用缩放
        if end > orig_max:
            #波长λ到维度i的映射，
            #波长公式：λ_i = 2π · base^{2(i)/d} ---> i = (L · log(b/(2π))) / (2·log(base))
            inv_dim = lambda b : ((dim * math.log(orig_max / (b * 2 * math.pi)) )
                                  /(2 * math.log(rope_base)))

            #划分高低维度
            #low：不需要缩放的高频； high：需要缩放的低频
            low = max(math.floor(inv_dim(beta_fast)),0)
            high =min(math.ceil(inv_dim(beta_slow)),dim//2-1)

            #计算缩放因子
            #low前，ramp为0；high后，ramp为1；low和high之间，先行过度
            # 公式: ramp(i) = clip((i - low)/(high - low), 0, 1)
            ramp = torch.clamp(
                #每个维度的i对齐到low，归一化到[0,1]
                (torch.arange(dim//2,device = freqs.device).float()-low)/max(high-low,0.001),
                            0,
                            1,#clamp（ ,0,1）限制到0-1
            )

            #ramp=0(高频)：系数为1，原频率不变
            #ramp=1（低频）：系数为 1/factor
            #ramp在0-1：平滑过渡
            #缩放公式：新频率 = 原频率 × (1 - ramp + ramp/factor)
            freqs = freqs*(1-ramp+ramp/factor)

        #根据end，生成位置索引t
        t = torch.arange(end,device = freqs.device).float()

        #计算外积，将t和频率部分相乘，得到每个位置的旋转角度
        freqs = torch.outer(t,freqs).float()

        # 扩展维度，每一对维度都有两相同的cos/sin
        #[end,d/2] --> [end,d]
        freqs_cos = (
            torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        )
        freqs_sin = (
            torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        )

        return freqs_cos,freqs_sin

#编写RoPE
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #旋转辅助函数，[a,b] --> [-b,a]
    def rotate_half(x):
        # 将向量分成两半并旋转
        # [x0, x1, x2, x3] → [-x1, x0, -x3, x2]
        return torch.cat(
            (-x[x.shape[-1]//2 :],x[...: x.shape[-1]//2]),
            dim = -1
        )
    #将旋转位置编码应用到qk上，
    #q' = q·cosφ + rotate_half(q)·sinφ
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) +
                (rotate_half(q)*sin.unsqueeze(unsqueeze_dim)))
    #k' = k·cosφ + rotate_half(k)·sinφ
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) +
                (rotate_half(k)*sin.unsqueeze(unsqueeze_dim)))
    return q_embed,k_embed










