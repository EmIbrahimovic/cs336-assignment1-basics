# Nullchar questions

The first few Unicode characters are special characters: they don't have a textual representation. 

That's why when printed it doesn't exist. But it's still useful to know what character it is.

## ..

'hello my name is emira and this ml thing is sick'
->
list of numbers. 
= list of bytes. 

large vocab size is a problem: you'll need more bytes to represent the same sentence
sparseness is a problem: if you know it's sparse then you can just reduce the vocab size

# UTF-8 questions

UTF-16, UTF-32 produce longer lists than UTF-8 = they gotta represent each character, not matter how small with
lotsa bytes. 

the most common letters we come across are small in Unicode so UTF-8 is gonna lead to less wasted space.
If we had a different primary alphabet then UTF-16, or 32 would make more sense bc they'll produce shorter lists

take any character that's longer than utf-8 lmao ćčšđ

so it seems like we have special characters in a byte stream signifying that what follows is a multi-byte character

# Resource accounting

```
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
```

Embedding layer: `[vocab_size, d_model]` (trainable params)

Transformer block: 
```
PreNorm [d_model]
Attention
    RoPE
    Wk [d_model, num_heads * d_k]
    Wq [d_model, num_heads * d_k]
    Wv [d_model, num_heads * d_v]
    Wo [num_heads * d_v, d_model]
SwiGLU
    wup = (d_model, d_ff)
    wdown = (d_ff, d_model)
    wgate = (d_model, d_ff)
PostNorm [d_model]
```

Norm:  `[d_model]`

Final linear layer: `[d_model, vocab_size]`


Trainable parameters:
```
vocab_size * d_model
= 50,257 × 1,600 = 80,411,200

num_layers * (16* d_model*d_model + 2*d_model)
    16*d_model^2 = 40960000
    2*d_model=3200

    Inner sum = 40,963,200
    Multiply by num_layers: 48 × 40,963,200 = 1,966,233,600

d_model
= 1,600

d_model * vocab_size
= 1,600 × 50,257 = 80,411,200

=
2,127,057,600
```


Bytes assuming single precision:
```
2,127,057,600 parameters × 4 bytes = 8,508,230,400 bytes ≈ 7.93 GB
```

## MatMuls and FLOPs

Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length token s.


Embedding layer: `[vocab_size, d_model]` (trainable params)
`X -> embedding[X]`

Transformer block: 
```
PreNorm [d_model] X -> scale and softmax
Attention
    RoPE
    Wk [d_model, num_heads * d_k] X Wk -> H (context_length, d_model) (d_model, d_k)
    Wq [d_model, num_heads * d_k] X Wq -> H (context_length, d_model) (d_model, d_k)
    Wv [d_model, num_heads * d_v] X Wv -> H (context_length, d_model) (d_model, d_v)
    A = Hx KQ = (context_length d_k) (d_k context_length)
    Wo [num_heads * d_v, d_model] 
    Hx A V = (con_len, con_len) (con_len d_v)
    
    H (AV) Wo = (con_len, H d_v) (H d_v, d_model)
SwiGLU
    wup = (d_model, d_ff)
    X Wup -> (con_len, d_model) (d_model, d_ff)
    wgate = (d_model, d_ff)
    X Wgate -> (con_len, d_model) (d_model, d_ff); then pointwise
    wdown = (d_ff, d_model)
    G Wdown -> (con_len, d_ff) (d_ff, d_model)
PostNorm [d_model] X -> scale and softmax
```

Norm:  `[d_model]` X -> scale and softmax

Final linear layer: `[d_model, vocab_size]`
`X W -> (con_len, d_model) (d_model, vocab_size)`

Output = `(con_len, vocab_size)` (not softmaxed, these are logits)
Or if there's batches `(B, con_len, vocab_size)`
For each sequence in the batch, for each position it returns a probability distribution. 

FLOPS
```
num_layers * (
3 * num_heads * con_len * d_model * d_k+ # kqv projections
num_heads * con_len * d_k * con_len +  # qt k
con_len * con_len * d_v + # av
con_len * num_heads * d_k * d_model # output
+
2*con_len * d_model*d_ff + con_len * d_ff * d_model # ffn
)
+
con_len*d_model*vocab_size
```

Simplified
```
num_layers * (
4 * con_len * d_model**2 + 2 * con_len**2 * d_model + # Projections and attention
12 * con_len * d_model**2 # FFN 
)
+
con_len*d_model*vocab_size # Linear for logits
```

```
48*
  (16*1024*1600^2 + 2*1024^2*1600)
+
1024*1600*50257
=
48 * 45,298,483,200 + 82,341,068,800
=
2,256,668,262,400
```

```
within the transformer block:
  projections are taking up: 13,841,203,200 
  ffn is taking up 12*con_len * d_model**2 = 31,457,280,000
```

One transformer layer is on the same order of magnitude of flops as our final Linear.
Within the transformer layer the FFN's are eating 60 bil FLOPs.

## Scaling GPT-2

GPT-2 small (12 layers, 768 d_model, 12 heads), 
GPT-2 medium (24 layers, 1024 d_model, 16 heads), 
GPT-2 large (36 layers, 1280 d_model, 20 heads). 

As the model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?

see `calculations.py`

The proportion that goes into the final linear layer decreases from 23% in small to 3% in XL (which makes sense as the
transformer block block increases).
As the d_model and num_heads grows the proportion of the transformer block FLOPS that go into the FFN also 
grows: 64%, 66%, 68%, 69%. The attention projection decreases: 7->5->4->3%

### Scaling context length

Attention gains in the % FLOPS, while the FFN decreases. 69->32%. Attention (projections and the attention matrix) take
up the rest. 

The final linear layer also decreases from 3 to 1.7% of total parameters.

# Optimizer

## Learning rates

In 100 iters, with lr = 1 on the toy example: 25.52965545654297, quickly drops to 15ish and gets down to 12.160255432128906

In 10 iters with For lr=10.0, in 10 iterations:
1. 28.936603546142578, 
2. 3.8889389038085938

For lr=100.0, in 10 iterations:
1. 22.737520217895508
2. 1.8321589557736552e-23

For lr=1000.0, in 10 iterations:
1. 22.729143142700195
2. 8205.2197265625
3. 1417170.75
4. 157645088.0
5. 12769251328.0
6. 805886033920.0
7. 41371557691392.0
8. 1779981281656832.0
9. 6.560631264116736e+16
10. 2.1066917666095104e+18

Yeah, having a more aggressive learning rate 10 or 100 helped us reach smaller losses in 10 interations, but going overboard
with 1000 led to hella divergence in loss.

## Resource accounting

Let us compute how much memory and compute running AdamW requires. Assume we are using
float32 for every tensor.

1. How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.

For simplicity, when calculating memory usage of activations, consider only the following components:
- Transformer block 
  - RMSNorm(s)
  - Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax,
  weighted sum of values, output projection. 
  - Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
- final RMSNorm
- output embedding
- cross-entropy on logits

**Deliverable:** An algebraic expression for each of parameters, activations, gradients, and opti-
mizer state, as well as the total.

```
Number of parameters: 	2*d_model*vocab_size + d_model + num_layers*(16*d_model**2 + 2*d_model)
Number of activations: 	batch_size*(2*con_len*d_model + con_len*vocab_size + num_layers*(2*con_len**2*num_heads + 20*con_len*d_model))
Number of gradients: 	2*d_model*vocab_size + d_model + num_layers*(16*d_model**2 + 2*d_model)
Number of states: 	4*d_model*vocab_size + 2*d_model + 2*num_layers*(16*d_model**2 + 2*d_model)
Peak memory during opt: batch_size*(2*con_len*d_model + con_len*vocab_size + num_layers*(2*con_len**2*num_heads + 20*con_len*d_model)) + 8*d_model*vocab_size + 4*d_model + 4*num_layers*(16*d_model**2 + 2*d_model)
```

Knowing which terms dominate tells us that scaling something in those terms will lead to linear memory cons. increase or 
quadratic, as is the case with `con_len`. Or technically with `d_model`, but that should not dominate.

2. Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

**Deliverable:** An expression that looks like a · batch_size + b for numerical values a, b, and a
number representing the maximum batch size.

```
Parameters: 4 × 2,127,057,600 = 8,508,230,400 bytes ≈ 7.93 GB
Gradients: 4 × 2,127,057,600 = 8,508,230,400 bytes ≈ 7.93 GB
AdamW states: 8 × 2,127,057,600 = 17,016,460,800 bytes ≈ 15.86 GB
Subtotal: 31.72 GB

Activations per batch: 2,572,960,768 × 4 = 10,291,843,072 bytes ≈ 9.59 GB
```

```
Total = 31.72 GB + 9.59 × B GB ≤ 80 GB
9.59 × B ≤ 48.28
B ≤ 5.03
```

```
Peak memory during opt: 4,144,186,368*batch_size + 8,508,230,400
80GB is 80*1e9 bytes
1 16bit float is 2 bytes
max_batch_size= 123014725//16188228 =7
```

3. How many FLOPs does running one step of AdamW take?

**Deliverable:** An algebraic expression, with a brief justification.

About 20x the number of parameters, meaning: 
`20 x [2*d_model*vocab_size + d_model + num_layers*(16*d_model**2 + 2*d_model)]`

4. Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022].
An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
assume that the backward pass has twice the FLOPs of the forward pass.

**Deliverable:** The number of days training would take, with a brief justification.

Assuming a computation-bound regime.
FLOPS = 3x flops for a forward pass.
2,256,668,262,400 * 3 * 1024 = 6,932,484,902,092,800 6932484902092800
for 400k passes
2,772,993,960,837,120,000,000 = 2.773 * 10^21

Time = Total FLOPs / Effective FLOPs_per_second
     = 2.772 × 10²¹ / 9.75 × 10¹²
     = 2.843 × 10⁸ seconds

2.843 × 10⁸ seconds / (24 × 3600) = 3,290 days ≈ 9.0 years

Also batch size of 1024 won't fit onto the device: gradient accumulation is necessary to get larger effective batch sizes.
Distributed training to the rescue to shorten this 9 year count.
