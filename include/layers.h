
#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    Tensor *weight; 
    Tensor *dweight;
    size_t vocab_size;
    size_t dim;
} Embedding;

Embedding *embedding_create(
    size_t vocab_size,
    size_t dim,
    uint64_t seed
);

void embedding_free(Embedding *layer);
void embedding_zero_grad(Embedding *layer);

Tensor *embedding_forward(
    const Embedding *layer,
    const uint32_t *ids,
    size_t batch_size,
    size_t seq_len
);

bool embedding_backward(
    Embedding *layer,
    const uint32_t *ids,
    size_t batch_size,
    size_t seq_len,
    const Tensor *dout
);

typedef struct {
    Tensor *weight;   // [in_dim, out_dim]
    Tensor *bias;     // [out_dim]
    Tensor *dweight;
    Tensor *dbias;
    size_t in_dim;
    size_t out_dim;
} Linear;

Linear *linear_create(
    size_t in_dim,
    size_t out_dim,
    uint64_t seed
);

void linear_free(Linear *layer);
void linear_zero_grad(Linear *layer);

Tensor *linear_forward(
    const Linear *layer,
    const Tensor *x
);

Tensor *linear_backward(
    Linear *layer,
    const Tensor *x,
    const Tensor *dout
);

Tensor *gelu_backward(
    const Tensor *x,
    const Tensor *dout
);

typedef struct {
    Tensor *gamma;    // [dim]
    Tensor *beta;     // [dim]
    Tensor *dgamma;
    Tensor *dbeta;
    size_t dim;
    float eps;
} LayerNorm;

LayerNorm *layernorm_create(
    size_t dim,
    float eps
);

void layernorm_free(LayerNorm *layer);
void layernorm_zero_grad(LayerNorm *layer);

Tensor *layernorm_forward(
    const LayerNorm *layer,
    const Tensor *x
);

Tensor *layernorm_backward(
    LayerNorm *layer,
    const Tensor *x,
    const Tensor *dout
);

Tensor *residual_forward(
    const Tensor *a,
    const Tensor *b
);

bool residual_backward(
    const Tensor *dout,
    Tensor *grad_a,
    Tensor *grad_b
);

#endif
