#include "bigram.h"

#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t rng_next(uint64_t *state)
{
    uint64_t z =
        (*state += UINT64_C(0x9E3779B97F4A7C15));

    z = (z ^ (z >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27)) *
        UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}

static double rng_uniform(uint64_t *state)
{
    return (double)(rng_next(state) >> 11) *
           (1.0 / 9007199254740992.0);
}


Bigram *bigram_create(size_t vocab_size, uint64_t seed)
{
    if (vocab_size == 0 || vocab_size > UINT32_MAX) {
        return NULL;
    }

    Bigram *model = calloc(1, sizeof(Bigram));

    if (model == NULL) {
        return NULL;
    }

    size_t shape[] = {vocab_size, vocab_size};

    model->weights = tensor_create(2, shape);
    model->grads = tensor_zeros(2, shape);
    model->vocab_size = vocab_size;

    if (model->weights == NULL || model->grads == NULL) {
        bigram_free(model);
        return NULL;
    }

    uint64_t state = seed;

    for (size_t i = 0; i < model->weights->numel; i++) {
        double r = rng_uniform(&state);

        model->weights->data[i] =
            0.02f * (float)(2.0 * r - 1.0);
    }

    return model;
}


void bigram_free(Bigram *model)
{
    if (model == NULL) {
        return;
    }

    tensor_free(model->weights);
    tensor_free(model->grads);

    free(model);
}

static bool bigram_validate_batch(
    const Bigram *model,
    const TokenBatch *batch
)
{
    if (model == NULL ||
        model->weights == NULL ||
        model->weights->data == NULL ||
        model->weights->shape == NULL ||
        batch == NULL ||
        batch->x == NULL ||
        batch->y == NULL) {
        return false;
    }

    if (batch->batch_size == 0 ||
        batch->context_length == 0 ||
        batch->batch_size > SIZE_MAX / batch->context_length) {
        return false;
    }

    size_t V = model->vocab_size;
    size_t N = batch->batch_size * batch->context_length;

    if (model->weights->ndim != 2 ||
        model->weights->shape[0] != V ||
        model->weights->shape[1] != V) {
        return false;
    }

    for (size_t i = 0; i < N; i++) {
        if (batch->x[i] >= V || batch->y[i] >= V) {
            return false;
        }
    }

    for (size_t i = 0; i < model->weights->numel; i++) {
        if (!isfinite(model->weights->data[i])) {
            return false;
        }
    }

    return true;
}

static bool bigram_compute_loss(
    const Bigram *model,
    const TokenBatch *batch,
    Tensor *grads,
    float *out_loss
)
{
    if (out_loss == NULL ||
        !bigram_validate_batch(model, batch)) {
        return false;
    }

    size_t V = model->vocab_size;
    size_t N = batch->batch_size * batch->context_length;

    if (grads != NULL) {
        if (grads->data == NULL ||
            grads->numel != model->weights->numel) {
            return false;
        }

        tensor_fill(grads, 0.0f);
    }

    double total_loss = 0.0;

    for (size_t n = 0; n < N; n++) {
        uint32_t x = batch->x[n];
        uint32_t y = batch->y[n];

        const float *row =
            &model->weights->data[(size_t)x * V];

        double max_logit = row[0];

        for (size_t j = 1; j < V; j++) {
            if (row[j] > max_logit) {
                max_logit = row[j];
            }
        }

        double sum_exp = 0.0;

        for (size_t j = 0; j < V; j++) {
            sum_exp += exp((double)row[j] - max_logit);
        }

        double sample_loss =
            max_logit + log(sum_exp) - (double)row[y];

        total_loss += sample_loss;

        if (grads != NULL) {
            float *grad_row =
                &grads->data[(size_t)x * V];

            for (size_t j = 0; j < V; j++) {
                double probability =
                    exp((double)row[j] - max_logit) / sum_exp;

                double target = (j == y) ? 1.0 : 0.0;

                grad_row[j] += (float)(
                    (probability - target) / (double)N
                );
            }
        }
    }

    *out_loss = (float)(total_loss / (double)N);

    return isfinite(*out_loss);
}


bool bigram_loss(
    const Bigram *model,
    const TokenBatch *batch,
    float *out_loss
)
{
    return bigram_compute_loss(
        model, batch, NULL, out_loss
    );
}


bool bigram_loss_backward(
    Bigram *model,
    const TokenBatch *batch,
    float *out_loss
)
{
    if (model == NULL || model->grads == NULL) {
        return false;
    }

    return bigram_compute_loss(
        model, batch, model->grads, out_loss
    );
}

bool bigram_sgd(
    Bigram *model,
    float learning_rate
)
{
    if (model == NULL ||
        model->weights == NULL ||
        model->grads == NULL ||
        model->weights->data == NULL ||
        model->grads->data == NULL ||
        model->weights->numel != model->grads->numel ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0f) {
        return false;
    }

    for (size_t i = 0; i < model->weights->numel; i++) {
        model->weights->data[i] -=
            learning_rate * model->grads->data[i];
    }

    return true;
}

bool bigram_generate(
    const Bigram *model,
    uint32_t start_id,
    size_t num_new_tokens,
    float temperature,
    uint64_t *rng_state,
    uint32_t *out_ids
)
{
    if (model == NULL ||
        model->weights == NULL ||
        model->weights->data == NULL ||
        rng_state == NULL ||
        start_id >= model->vocab_size ||
        !isfinite(temperature) ||
        temperature <= 0.0f ||
        (num_new_tokens > 0 && out_ids == NULL)) {
        return false;
    }

    if (num_new_tokens == 0) {
        return true;
    }

    size_t V = model->vocab_size;

    if (V > SIZE_MAX / sizeof(double)) {
        return false;
    }

    double *probs = malloc(V * sizeof(double));

    if (probs == NULL) {
        return false;
    }

    uint32_t current = start_id;

    for (size_t t = 0; t < num_new_tokens; t++) {
        const float *row =
            &model->weights->data[(size_t)current * V];

        double max_logit = -INFINITY;

        for (size_t j = 0; j < V; j++) {
            if (!isfinite(row[j])) {
                free(probs);
                return false;
            }

            double score = (double)row[j] / temperature;

            if (score > max_logit) {
                max_logit = score;
            }
        }

        double sum = 0.0;

        for (size_t j = 0; j < V; j++) {
            double score = (double)row[j] / temperature;

            probs[j] = exp(score - max_logit);
            sum += probs[j];
        }
        double r = rng_uniform(rng_state) * sum;
        double cumulative = 0.0;

        uint32_t next = (uint32_t)(V - 1);

        for (size_t j = 0; j < V; j++) {
            cumulative += probs[j];

            if (r < cumulative) {
                next = (uint32_t)j;
                break;
            }
        }

        out_ids[t] = next;
        current = next;
    }

    free(probs);

    return true;
}