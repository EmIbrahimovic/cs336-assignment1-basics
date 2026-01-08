import torch
from sympy.physics.control import gain_margin
from torch import nn, Tensor
import numpy as np
from einops import einsum, rearrange

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

    lookup_table: Tensor
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
        self.lookup_table = nn.Parameter(torch.empty([num_embeddings, embedding_dim]))
        torch.nn.init.trunc_normal_(self.lookup_table, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.lookup_table[token_ids]


class RMSNorm(nn.Module):
    """
    Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
    """
    d_model: int
    eps: float
    gain: Tensor

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
        self.gain = nn.Parameter(torch.ones([d_model], dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(1/self.d_model * x.pow(2).sum(-1, keepdim=True) + self.eps)
        rms_norm = x / rms * self.gain

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
    weight1: Tensor
    weight2: Tensor
    weight3: Tensor

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
        self.weight1 = nn.Parameter(torch.empty([d_ff, d_model], dtype=dtype, device=device))
        self.weight2 = nn.Parameter(torch.empty([d_model, d_ff], dtype=dtype, device=device))
        self.weight3 = nn.Parameter(torch.empty([d_ff, d_model], dtype=dtype, device=device))

        sigma = np.sqrt(2/(d_model + d_ff))
        nn.init.trunc_normal_(self.weight1, 0, sigma, -3*sigma, 3*sigma)
        nn.init.trunc_normal_(self.weight2, 0, sigma, -3*sigma, 3*sigma)
        nn.init.trunc_normal_(self.weight3, 0, sigma, -3*sigma, 3*sigma)

        self.silu = SiLU()


    def forward(self, x):
        """
        Do the forward pass, applying the SiLU activation function and gating.
        """
        after_silu = self.silu(einsum(x, self.weight1, "... dmodel, dff dmodel -> ... dff"))
        to_be_gated = einsum(x, self.weight3, "... dmodel, dff dmodel -> ... dff")
        return einsum(after_silu * to_be_gated, self.weight2, "... dff, dmodel dff -> ... dmodel")



class RoPE(nn.Module):
    """
    Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
tensor. The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct
the RoPE module and create buffers if needed.
theta: float Θ value for the RoPE
d_k: int dimension of query and key vectors
max_seq_len: int Maximum sequence length that will be inputted
device: torch.device | None = None Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
Note that you should tolerate x with an arbitrary number of batch dimensions. You should
assume that the token positions are a tensor of shape (..., seq_len) specifying the token
positions of x along the sequence dimension.
You should use the token positions to slice your (possibly precomputed) cos and sin tensors
along the sequence dimension.
    """
