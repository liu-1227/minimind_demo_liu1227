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