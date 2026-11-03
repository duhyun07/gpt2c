#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static unsigned char *read_file(
    const char *path,
    size_t *out_len
)
{
    if (path == NULL || out_len == NULL) {
        return NULL;
    }

    FILE *fp = fopen(path, "rb");

    if (fp == NULL) {
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }

    long file_size = ftell(fp);

    if (file_size <= 0 ||
        (uintmax_t)file_size > SIZE_MAX) {
        fclose(fp);
        return NULL;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }

    size_t len = (size_t)file_size;

    unsigned char *buffer = malloc(len);

    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, len, fp);

    if (bytes_read != len) {
        free(buffer);
        fclose(fp);
        return NULL;
    }

    if (fclose(fp) != 0) {
        free(buffer);
        return NULL;
    }

    *out_len = len;

    return buffer;
}


Dataset *dataset_load(const char *path)
{
    size_t text_len = 0;

    unsigned char *text = read_file(path, &text_len);

    if (text == NULL) {
        return NULL;
    }

    size_t train_len =
        (text_len / 10) * 9 +
        ((text_len % 10) * 9) / 10;

    if (train_len == 0 || train_len >= text_len) {
        free(text);
        return NULL;
    }

    if (text_len > SIZE_MAX / sizeof(uint32_t)) {
        free(text);
        return NULL;
    }

    Dataset *dataset = calloc(1, sizeof(Dataset));

    if (dataset == NULL) {
        free(text);
        return NULL;
    }

    if (!tokenizer_build(
            &dataset->tokenizer,
            text,
            train_len
        )) {
        goto error;
    }

    dataset->tokens = malloc(
        text_len * sizeof(uint32_t)
    );

    if (dataset->tokens == NULL) {
        goto error;
    }

    if (!tokenizer_encode(
            &dataset->tokenizer,
            text,
            text_len,
            dataset->tokens
        )) {
        goto error;
    }

    dataset->num_tokens = text_len;
    dataset->train_tokens = train_len;

    free(text);

    return dataset;

error:
    free(text);
    dataset_free(dataset);

    return NULL;
}


void dataset_free(Dataset *dataset)
{
    if (dataset == NULL) {
        return;
    }

    free(dataset->tokens);
    free(dataset);
}

TokenBatch *token_batch_create(
    size_t batch_size,
    size_t context_length
)
{
    if (batch_size == 0 || context_length == 0) {
        return NULL;
    }

    // Check multiplication overflow
    if (batch_size > SIZE_MAX / context_length) {
        return NULL;
    }

    size_t numel = batch_size * context_length;

    if (numel > SIZE_MAX / sizeof(uint32_t)) {
        return NULL;
    }

    TokenBatch *batch = calloc(1, sizeof(TokenBatch));

    if (batch == NULL) {
        return NULL;
    }

    batch->batch_size = batch_size;
    batch->context_length = context_length;

    batch->x = malloc(numel * sizeof(uint32_t));
    batch->y = malloc(numel * sizeof(uint32_t));

    if (batch->x == NULL || batch->y == NULL) {
        token_batch_free(batch);
        return NULL;
    }

    return batch;
}


void token_batch_free(TokenBatch *batch)
{
    if (batch == NULL) {
        return;
    }

    free(batch->x);
    free(batch->y);
    free(batch);
}

static uint64_t rng_next(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));

    z = (z ^ (z >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27)) *
        UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}

bool dataset_sample_batch(
    const Dataset *dataset,
    DatasetSplit split,
    TokenBatch *batch,
    uint64_t *rng_state
)
{
    if (dataset == NULL ||
        dataset->tokens == NULL ||
        batch == NULL ||
        batch->x == NULL ||
        batch->y == NULL ||
        rng_state == NULL) {
        return false;
    }

    if (batch->batch_size == 0 ||
        batch->context_length == 0 ||
        dataset->train_tokens > dataset->num_tokens) {
        return false;
    }

    if (split != DATASET_TRAIN && split != DATASET_VAL) {
        return false;
    }

    size_t split_start;
    size_t split_len;

    if (split == DATASET_TRAIN) {
        split_start = 0;
        split_len = dataset->train_tokens;
    } else {
        split_start = dataset->train_tokens;
        split_len =
            dataset->num_tokens - dataset->train_tokens;
    }

    size_t T = batch->context_length;
    size_t B = batch->batch_size;

    if (split_len <= T) {
        return false;
    }

    size_t num_starts = split_len - T;

    for (size_t b = 0; b < B; b++) {
        size_t offset =
            (size_t)(rng_next(rng_state) % num_starts);

        size_t start = split_start + offset;

        for (size_t t = 0; t < T; t++) {
            size_t index = b * T + t;

            batch->x[index] =
                dataset->tokens[start + t];

            batch->y[index] =
                dataset->tokens[start + t + 1];
        }
    }

    return true;
}