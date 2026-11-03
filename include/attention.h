#ifndef ATTENTION_H
#define ATTENTION_H

#include "layers.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;

    Linear *q_proj;
    Linear *k_proj;
    Linear *v_proj;
    Linear *out_proj;
} Attention;

typedef struct {
    Tensor *q;        
    Tensor *k;        
    Tensor *v;        

    Tensor *probs;
    Tensor *context;  
} AttentionCache;

Attention *attention_create(
    size_t embed_dim,
    size_t num_heads,
    uint64_t seed
);

void attention_free(Attention *attn);
void attention_zero_grad(Attention *attn);
void attention_cache_free(AttentionCache *cache);

Tensor *attention_forward(
    const Attention *attn,
    const Tensor *x,
    AttentionCache **out_cache
);

Tensor *attention_backward(
    Attention *attn,
    const Tensor *x,
    const Tensor *dout,
    const AttentionCache *cache
);

#endif