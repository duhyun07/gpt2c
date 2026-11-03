#ifndef DATASET_H
#define DATASET_H

#include "tokenizer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    DATASET_TRAIN = 0,
    DATASET_VAL = 1
} DatasetSplit;

typedef struct {
    Tokenizer tokenizer;

    uint32_t *tokens;

    size_t num_tokens;
    size_t train_tokens;
} Dataset;

typedef struct {
    uint32_t *x;
    uint32_t *y;

    size_t batch_size;
    size_t context_length;
} TokenBatch;


Dataset *dataset_load(const char *path);

void dataset_free(Dataset *dataset);

TokenBatch *token_batch_create(
    size_t batch_size,
    size_t context_length
);

void token_batch_free(TokenBatch *batch);

bool dataset_sample_batch(
    const Dataset *dataset,
    DatasetSplit split,
    TokenBatch *batch,
    uint64_t *rng_state
);

#endif