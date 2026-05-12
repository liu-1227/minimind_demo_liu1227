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

############################ RMSNorm ###############################
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

############################ RoPE&YaRN ###############################
import math
from typing import Optional,Tuple
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


############################ GQA ###############################
#多次获取kv
def repeat_kv(x:torch.Tensor,n_rep:int) -> torch.Tensor:
    bs,slen,num_key_value_heads,heads_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:, :,None, :]#增加维度None
        .expand(bs,slen,num_key_value_heads,n_rep,heads_dim)#扩展张量n_rep
        .reshape(bs,slen,num_key_value_heads*n_rep,heads_dim)#重塑形状，合并维度
    )

from torch.nn import functional as F
class Attention(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()
        #kv头数量(MHA)
        self.num_key_value_heads = args.num_attention_heads \
            if args.num_key_value_heads is None \
            else args.num_key_value_heads

        #assert条件若为假，则会报错AssertionError
        #分组前提：注意力头数可以被kv头数整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        "num_key_value_heads must be divisible by num_attention_heads"

        #分组计算
        self.n_local_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads

        #每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        #线性投影层
        self.q_proj = nn.Linear(args.hidden_size,args.num_attention_heads * self.head_dim,bias = False)
        self.k_proj = nn.Linear(args.hidden_size,self.num_key_value_heads * self.head_dim,bias = False)
        self.v_proj = nn.Linear(args.hidden_size,self.num_key_value_heads * self.head_dim,bias = False)

        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size,bias = False)#将多头输出映射到原始维度

        #Dropout层
        self.atten_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        #是否支持flsah
        self.flash = hasattr(torch.nn.functional,"scaled_dot_product_attention") and args.flash_attention

    #forward方法
    def forward(self,
        x:torch.Tensor,
        position_embdding:Tuple[torch.Tensor,torch.Tensor],
        past_key_value : Optional[Tuple[torch.Tensor,torch.Tensor]] = None,
        use_cache = False,
        attention_mask : Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        #线性投影并重塑为多头格式
        #投影，计算qkv
        bsz, seq_len, _ = x.shape # [batch, seq_len, hidden_size]
        xq = self.q_proj(x) #hidden_size=n_local_heads * head_dim
        xk = self.k_proj(x) #hidden_size=num_kv_heads * head_dim
        xv = self.v_proj(x)

        #把输入拆封成多个头，view将最后一个维度拆成两个
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        #q,k --> RoPE
        cos,sin = position_embdding
            #为qk添加位置信息
        xq,xk = apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])

        #k,v --> repeat（注意kv cache）
        #kv缓存处理，用于自回归生成
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim=1)
            xv = torch.cat([past_key_value[1],xv],dim=1)
        past_kv = (xk,xv) if use_cache else None#保存用于下一步
        #重复kv，并调整维度顺序
        #维度转置：[bsz, seq_len, heads, dim] -> [bsz, heads, seq_len, dim]
        xq = xq.transpose(1, 2)  # [bsz, n_local_heads, seq_len, head_dim]
        #重复kv头：
        # [bsz, num_kv_heads, seq_len, dim] -> [bsz, n_local_heads, seq_len, dim]（通过重复）
        xk = repeat_kv(xk,self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv,self.n_rep).transpose(1, 2)


        #进行attention计算，
        #内置实现
        if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask ==1)):
            #注意力扩展掩码
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,seq_len,-1).bool()
            )
            output = F.scaled_dot_product_attention(
                xq,xk,xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True#因果掩码，不能看到未来
            )
        #自己定义，手动实现
        else:
            #注意力分数：（q*k^T）/sqart(d)
            scores = (xq@xk.transpose(-2,-1)/math.sqrt(self.head_dim))
            # [bsz, heads, seq_len, seq_len]

            #添加因果掩码（下三角矩阵）
            scores = scores + torch.triu(
                torch.full((seq_len,seq_len),float('-inf'),device=scores.device),
                diagonal = 1
            ).unsqueeze(0).unsqueeze(0)

        #拼接头，输出投影并返回
            #添加自定义注意力源码
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * 1e9
                scores = scores + extended_attention_mask
            #softmax+dropout
            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores@xv

        #调整维度：[bsz, heads, seq_len, dim] -> [bsz, seq_len, heads*dim]
        output = output.transpose(1,2).reshape(bsz,seq_len,-1)
        # 最终投影：heads*dim -> hidden_size
        output = self.resid_dropout(self.o_proj(output))
        return output,past_kv











