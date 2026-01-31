import sympy as sp

batch_size = sp.symbols('batch_size')
vocab_size = sp.symbols('vocab_size') # 50,257
con_len = sp.symbols('con_len') # 1,024
num_layers = sp.symbols('num_layers') # 48
d_model = sp.symbols('d_model') # 1,600
num_heads = sp.symbols('num_heads') # 25
d_ff = sp.symbols('d_ff') # 6,400
d_ff.subs([('d_ff', 4*d_model)])
d_k = d_model / num_heads

# ===========================================
#               FLOPS
# ===========================================

# WARNING: Not necessarily correct.
# Values for a single batch.
rms_norm_a = d_model
kqv_projections_a = 3 * num_heads * con_len * d_model * d_k
attention_a = num_heads * con_len * d_k * con_len
attention_value_a = num_heads * con_len * con_len * d_k
softmax_a = attention_value_a
output_projection_a = con_len * num_heads * d_k * d_model
feed_forward_a = 2 * con_len * d_model * d_ff + con_len * d_ff * d_model

transformer_block_f = kqv_projections_a + attention_a + attention_value_a + output_projection_a + feed_forward_a

final_linear_f = con_len * d_model * vocab_size

total_f = num_layers * transformer_block_f + final_linear_f

# ===========================================
#               ACTIVATIONS ACC
# ===========================================
activations_single_batch = (3*con_len * d_model + con_len*vocab_size +
               num_layers * (20*con_len*d_model + 2*num_heads * con_len**2)) # 16, 1
num_activations = activations_single_batch * batch_size
num_parameters = (2* d_model * vocab_size +
                  num_layers * (2 * d_model + 16 * d_model**2) +
                  d_model)

num_gradients = num_parameters
num_states = 2*num_parameters

total_opt_mem = num_activations + num_parameters + num_gradients + num_states

# ===========================================
#               GPT-2 Hyperparameters
# ===========================================
gpt2_params = {
    "XL": [
        (vocab_size, 50257),
        (con_len, 1024),
        (num_layers, 48),
        (d_model, 1600),
        (num_heads, 25),
        (d_ff, 6400)
    ],
    "large": [
        (vocab_size, 50257),
        (con_len, 1024),
        (num_layers, 36),
        (d_model, 1280),
        (num_heads, 20),
        (d_ff, 4*1280) # question
    ],
    "medium": [
        (vocab_size, 50257),
        (con_len, 1024),
        (num_layers, 24),
        (d_model, 1024),
        (num_heads, 16),
        (d_ff, 4*1024)
    ],
    "small": [
        (vocab_size, 50257),
        (con_len, 1024),
        (num_layers, 12),
        (d_model, 768),
        (num_heads, 12),
        (d_ff, 4*768)
    ]
}

gpt2_xl_params = {
    "small con_len": [
        (vocab_size, 50257),
        (con_len, 1024),
        (num_layers, 48),
        (d_model, 1600),
        (num_heads, 25),
        (d_ff, 6400)
    ],
    "large con_len": [
        (vocab_size, 50257),
        (con_len, 16394),
        (num_layers, 48),
        (d_model, 1600),
        (num_heads, 25),
        (d_ff, 6400)
    ],
}


# ===========================================
#               PRINTING
# ===========================================

def print_flops():
    precision = 4
    for version, version_params in gpt2_xl_params.items():
        batch_size.subs(1)
        total_flops = total_f.subs(version_params)
        transformer_block_flops = transformer_block_f.subs(version_params)
        version_params = dict(version_params)

        print()
        print(f"=== GPT-2 {version}")
        print(f"Total FLOPs: {int(total_flops):,}")
        print(f"\t > Transformer block")
        print(f"\t \t KQV projections \t\t\t\t{int(kqv_projections_a.subs(version_params)):,} \t {(kqv_projections_a / transformer_block_flops).evalf(precision, version_params)}")
        print(f"\t \t Attention projection \t\t\t{int(attention_a.subs(version_params)):,} \t {(attention_a / transformer_block_flops).evalf(precision, version_params)}")
        print(f"\t \t Attention-value projection \t{int(attention_value_a.subs(version_params)):,} \t {(attention_value_a / transformer_block_flops).evalf(precision, version_params)}")
        print(f"\t \t Output projection \t\t\t\t{int(output_projection_a.subs(version_params)):,} \t {(output_projection_a / transformer_block_flops).evalf(precision, version_params)}")
        print(f"\t \t Feed-forward net \t\t\t\t{int(feed_forward_a.subs(version_params)):,} \t {(feed_forward_a / transformer_block_flops).evalf(precision, version_params)}")
        print(f"\t\t == Total single transformer block \t{int(transformer_block_f.subs(version_params)):,}")
        print(f"\t\t == Total {num_layers.subs(version_params)} blocks \t\t{int((num_layers * transformer_block_f).subs(version_params)):,} \t {((num_layers * transformer_block_f) / total_flops).evalf(precision, version_params)}")
        print(f"\t > Final linear \t{int(final_linear_f.subs(version_params)):,} \t {(final_linear_f / total_flops).evalf(precision, version_params)}")

def print_optimizer_memory():
    # Memory usage:
    # 1. Parameters
    # 2. Gradient
    # 3. Optimizer states
    # 4. Activations

    print(f"Number of parameters: \t{num_parameters}")
    print(f"Number of activations: \t{num_activations}")
    print(f"Number of gradients: \t{num_gradients}")
    print(f"Number of states: \t{num_states}")
    print(f"Peak memory during opt: \t{total_opt_mem}")


def print_gpt_2_mem():
    params = gpt2_params['small']
    print(f"Number of parameters: \t{(num_parameters.subs(params))}")
    print(f"Number of activations: \t{(num_activations.subs(params))}")
    print(f"Number of gradients: \t{(num_gradients.subs(params))}")
    print(f"Number of states: \t{(num_states.subs(params))}")
    print(f"Peak memory during opt: \t{(total_opt_mem.subs(params))}")

    print(f"80GB is 80*1e9 bytes")
    print(f"1 16bit float is 2 bytes")
    max_batch_size = sp.solvers.solve(total_opt_mem.subs(params) * 2 - 80*10**9, batch_size)
    print(f"{max_batch_size=}")

print_gpt_2_mem()

# print_flops()
