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

num_layers * (d_model*num_heads*(2*d_k + d_v) + 2*d_model + 3*d_model*d_ff)
    d_model*num_heads*(2*d_k + d_v) = 1,600 × 25 × (2×64 + 64) = 1,600 × 25 × 192 = 7,680,000
    2*d_model = 2 × 1,600 = 3,200
    3*d_model*d_ff = 3 × 1,600 × 6,400 = 30,720,000

    Inner sum = 7,680,000 + 3,200 + 30,720,000 = 38,403,200
    Multiply by num_layers: 48 × 38,403,200 = 1,843,353,600

d_model
= 1,600

d_model * vocab_size
= 1,600 × 50,257 = 80,411,200

=
2,004,177,600
```


Bytes assuming single precision:
```
2,004,177,600 parameters × 4 bytes = 8,016,710,400 bytes ≈ 8.02 GB
```

## MatMuls and FLOPs

Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.


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

FLOPS
```
num_layers * (
num_heads * con_len * d_model * d_k+
num_heads * con_len * d_k * con_len + 
con_len * num_heads * d_k * d_model
+
2*con_len * d_model*d_ff + con_len * d_ff * d_model
)
+
con_len*d_model*vocab_size
```
```
num_heads * con_len * d_model * d_k
    = 25 × 1,024 × 1,600 × 64 = 2,621,440,000
num_heads * con_len * d_k * con_len
    = 25 × 1,024 × 64 × 1,024 = 1,677,721,600
con_len * num_heads * d_k * d_model
    = 1,024 × 25 × 64 × 1,600 = 2,621,440,000
2 * con_len * d_model * d_ff
    = 2 × 1,024 × 1,600 × 6,400 = 20,971,520,000
con_len * d_ff * d_model
    = 1,024 × 6,400 × 1,600 = 10,485,760,000
    Sum of inner terms:
    2,621,440,000 + 1,677,721,600 + 2,621,440,000 + 20,971,520,000 + 10,485,760,000 = 38,377,881,600

Multiply by num_layers:
    48 × 38,377,881,600 = 1,842,138,316,800
Final term: con_len * d_model * vocab_size
    = 1,024 × 1,600 × 50,257 = 82,341,068,800

Total:
1,842,138,316,800 + 82,341,068,800 = 1,924,479,385,600
The expression evaluates to 1,924,479,385,600 (approximately 1.92 trillion operations/FLOPs for a forward pass).
```

One transformer layer is on the same order of magnitude of flops as our final Linear.
Within the transformer layer the FFN's are eating 60 bil FLOPs.

## Scaling GPT-2



