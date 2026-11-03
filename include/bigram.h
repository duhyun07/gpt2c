#ifndef BIGRAM_H
#define BIGRAM_H

#include "tensor.h"
#include "dataset.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    Tensor *weights;
    Tensor *grads; 
    size_t vocab_size;
} Bigram;

Bigram *bigram_create(size_t vocab_size, uint64_t seed);

void bigram_free(Bigram *model);

bool bigram_loss(
    const Bigram *model,
    const TokenBatch *batch,
    float *out_loss
);

bool bigram_loss_backward(
    Bigram *model,
    const TokenBatch *batch,
    float *out_loss
);

bool bigram_sgd(
    Bigram *model,
    float learning_rate
);

bool bigram_generate(
    const Bigram *model,
    uint32_t start_id,
    size_t num_new_tokens,
    float temperature,
    uint64_t *rng_state,
    uint32_t *out_ids
);

#endif