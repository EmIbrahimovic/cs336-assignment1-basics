from collections import OrderedDict

import torch
from jaxtyping import *
from torch import nn, Tensor
import numpy as np
from einops import einsum, rearrange, repeat


class Linear(nn.Module):

    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features, out_features, device=None, dtype=None, *args, **kwargs):
        """
        Construct a linear transformation module.

        Parameters:
                in_features: int final dimension of the input
                out_features: int final dimension of the output
                device: torch.device | None = None Device to store the parameters on
                dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__(*args, **kwargs)

        self.in_features = in_features
        self.out_features = out_features

        sigma = np.sqrt(2/(in_features + out_features))
        self.weight = nn.Parameter(torch.empty([out_features, in_features], dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, 0, sigma, -3*sigma, 3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):

    weight: Tensor
    dmodel: int
    num_embeddings: int

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, *args, **kwargs):
        """
        Construct an embedding module.

        Parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__(*args, **kwargs)

        self.num_embeddings = num_embeddings
        self.dmodel = embedding_dim
        self.weight = nn.Parameter(torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
    """
    d_model: int
    eps: float
    weight: Tensor

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, *args, **kwargs):
        """
        Construct the RMSNorm module.

        Parameters:
                d_model: int Hidden dimension of the model
                eps: float = 1e-5 Epsilon value for numerical stability
                device: torch.device | None = None Device to store the parameters on
                dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones([d_model], dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(1/self.d_model * x.pow(2).sum(-1, keepdim=True) + self.eps)
        rms_norm = x / rms * self.weight

        return rms_norm.to(in_dtype)


class SiLU(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        Construct the SiLU module
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        Apply the SiLU activation function.
        """
        return torch.sigmoid(x) * x

class SwiGLU(nn.Module):

    d_model: int
    d_ff: int
    silu: SiLU
    w1: Linear
    w2: Linear
    w3: Linear

    def __init__(self, d_model, d_ff=None, device=None, dtype=None, *args, **kwargs):
        """
        Construct a linear transformation module.

        Parameters:
                in_features: int final dimension of the input
                out_features: int final dimension of the output
                device: torch.device | None = None Device to store the parameters on
                dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__(*args, **kwargs)

        if not d_ff:
            d_ff = np.ceil((8/3 * d_model) / 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.silu = SiLU()


    def forward(self, x):
        """
        Do the forward pass, applying the SiLU activation function and gating.
        """
        U = self.silu(self.w1(x))
        G = self.w3(x)
        D = self.w2(U * G)

        return D



class RoPE(nn.Module):
    """
    Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Assuming d_k even?

        Parameters
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        assert d_k % 2 == 0, "Model dimension is odd (not divisible by 2), I'm confused"
        self.d_k = d_k

        seq_positions = repeat(torch.arange(max_seq_len, device=device), "seq_len -> seq_len d", d=d_k//2)
        pair_indices = repeat(torch.arange(d_k//2, device=device), "d -> seq_len d", seq_len=max_seq_len)
        thetas = seq_positions / (theta ** (2 * pair_indices / d_k)) # thetas = seq_positions * theta ** (-2 * pair_indices / d_k)
        sin = torch.sin(thetas)
        cos = torch.cos(thetas)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

        The token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        """
        in_dtype = x.dtype
        x = x.float()

        # [batches x seq_len]
        sin = torch.repeat_interleave(self.sin, 2, dim=-1)[token_positions] # max_seq_len, d_k/2 --> sl, dk
        cos = torch.repeat_interleave(self.cos, 2, dim=-1)[token_positions] # max_seq_len, d_k/2 --> sl, dk

        funky_x = rearrange([-x[..., 1::2], x[..., 0::2]], "two ... seq_len d -> ... seq_len (d two)")
        answer = x * cos + funky_x * sin

        return answer.to(in_dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Applies softmax to the input tensor along the specified direction, returning a tensor of the same size.
    """
    x_diff = x - torch.max(x, dim=dim, keepdim=True).values
    exp_sum = torch.exp(x_diff).sum(dim=dim, keepdim=True)

    return torch.exp(x_diff) / exp_sum

def dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v).
    Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
    seq_len). The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    """
    d_k = K.shape[-1]
    attention = einsum(Q, K, "... queries d_k, ... seq_len d_k -> ... queries seq_len") * (d_k ** -0.5)
    masked_attention = torch.where(mask,
                                   attention,
                                   torch.tensor(float("-inf"), device=attention.device, dtype=attention.dtype)) # ... queries seqlen to see how much each token should attend to each other token
    return einsum(V, softmax(masked_attention, dim=-1), "... seq_len d_v, ... queries seq_len -> ... queries d_v") # essentially a vector matrix multiplication.


class CausalMultiHeadedSelfAttention(nn.Module):

    d_model: int
    num_heads: int
    d_k: int
    d_v: int
    q_proj: Linear
    l_proj: Linear
    v_proj: Linear
    output_proj: Linear
    rope_layer: RoPE = None

    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, rope_theta=None, device=None, dtype=None, *args, **kwargs):
        super().__init__()

        if dtype:
            self.dtype = dtype

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)

        if rope_theta is not None:
            self.rope_layer = RoPE(rope_theta, self.d_k, max_seq_len, device)


    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions=None):
        """
        Performs a forward pass of _causal_ multi-headed self attention with the given input.
        Returns a tensor of dimension "... seq_len d_v" // d_model
        """
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], device=x.device, dtype=bool))

        Q = rearrange(self.q_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)

        # Apply rope to the Q and K layers. RoPE returns vectors of the same size. And applies the transformation to every batch-like dimension
        if self.rope_layer is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2])

            Q = self.rope_layer(Q, token_positions)
            K = self.rope_layer(K, token_positions)

        attn_result = rearrange(dot_product_attention(Q, K, V, mask), "... h seq_len d_v -> ... seq_len (h d_v)")
        answer = self.output_proj(attn_result)

        return answer

class TransformerBlock(nn.Module):

    ln1: RMSNorm
    attn: CausalMultiHeadedSelfAttention
    ln2: RMSNorm
    ffn: SwiGLU

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, rope_theta=None, device=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device)
        self.attn = CausalMultiHeadedSelfAttention(d_model,
                                                   num_heads,
                                                   max_seq_len=max_seq_len,
                                                   rope_theta=rope_theta,
                                                   device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]):
        x = x + self.attn(self.ln1(x))
        ffn = self.ffn(self.ln2(x))
        return x + ffn


class TransformerLM(nn.Module):

    token_embeddings: Embedding
    layers: nn.Sequential
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, rope_theta, device=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.Sequential(
            OrderedDict([
                (f"{i}", TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device))
                for i in range(num_layers)
            ])
        )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(self, x: str):
        x = self.token_embeddings(x)
        x = self.ln_final(self.layers(x))
        x = self.lm_head(x)
        return x

