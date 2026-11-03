
#include "layers.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
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

static float random_weight(uint64_t *state)
{
    double u = (double)(rng_next(state) >> 11) *
               (1.0 / 9007199254740992.0);

    return 0.02f * (float)(2.0 * u - 1.0);
}

static bool same_shape(
    const Tensor *a,
    const Tensor *b
)
{
    if (!a || !b ||
        !a->data || !b->data ||
        !a->shape || !b->shape ||
        a->ndim == 0 ||
        a->ndim != b->ndim ||
        a->numel != b->numel) {
        return false;
    }

    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    return true;
}

static bool last_dim_is(const Tensor *x, size_t dim)
{
    return x != NULL &&
           x->data != NULL &&
           x->shape != NULL &&
           x->ndim >= 1 &&
           dim > 0 &&
           x->shape[x->ndim - 1] == dim &&
           x->numel % dim == 0;
}

Embedding *embedding_create(
    size_t vocab_size,
    size_t dim,
    uint64_t seed
)
{
    if (vocab_size == 0 ||
        dim == 0 ||
        vocab_size > UINT32_MAX) {
        return NULL;
    }

    Embedding *layer = calloc(1, sizeof(*layer));

    if (layer == NULL) {
        return NULL;
    }

    size_t shape[] = {vocab_size, dim};

    layer->weight = tensor_create(2, shape);
    layer->dweight = tensor_zeros(2, shape);

    if (!layer->weight || !layer->dweight) {
        embedding_free(layer);
        return NULL;
    }

    layer->vocab_size = vocab_size;
    layer->dim = dim;

    for (size_t i = 0; i < layer->weight->numel; i++) {
        layer->weight->data[i] = random_weight(&seed);
    }

    return layer;
}

void embedding_free(Embedding *layer)
{
    if (layer == NULL) {
        return;
    }

    tensor_free(layer->weight);
    tensor_free(layer->dweight);
    free(layer);
}

void embedding_zero_grad(Embedding *layer)
{
    if (layer != NULL) {
        tensor_fill(layer->dweight, 0.0f);
    }
}

Tensor *embedding_forward(
    const Embedding *layer,
    const uint32_t *ids,
    size_t batch_size,
    size_t seq_len
)
{
    if (layer == NULL ||
        layer->weight == NULL ||
        layer->weight->data == NULL ||
        ids == NULL ||
        batch_size == 0 ||
        seq_len == 0 ||
        batch_size > SIZE_MAX / seq_len) {
        return NULL;
    }

    size_t shape[] = {
        batch_size,
        seq_len,
        layer->dim
    };

    Tensor *out = tensor_create(3, shape);

    if (out == NULL) {
        return NULL;
    }

    size_t N = batch_size * seq_len;

    for (size_t i = 0; i < N; i++) {
        if (ids[i] >= layer->vocab_size) {
            tensor_free(out);
            return NULL;
        }

        memcpy(
            out->data + i * layer->dim,
            layer->weight->data +
                (size_t)ids[i] * layer->dim,
            layer->dim * sizeof(float)
        );
    }

    return out;
}

bool embedding_backward(
    Embedding *layer,
    const uint32_t *ids,
    size_t batch_size,
    size_t seq_len,
    const Tensor *dout
)
{
    if (layer == NULL ||
        layer->dweight == NULL ||
        layer->dweight->data == NULL ||
        ids == NULL ||
        batch_size == 0 ||
        seq_len == 0 ||
        batch_size > SIZE_MAX / seq_len ||
        dout == NULL ||
        dout->data == NULL ||
        dout->shape == NULL ||
        dout->ndim != 3 ||
        dout->shape[0] != batch_size ||
        dout->shape[1] != seq_len ||
        dout->shape[2] != layer->dim) {
        return false;
    }

    size_t N = batch_size * seq_len;

    for (size_t i = 0; i < N; i++) {
        if (ids[i] >= layer->vocab_size) {
            return false;
        }
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < layer->dim; j++) {
            size_t index =
                (size_t)ids[i] * layer->dim + j;

            layer->dweight->data[index] +=
                dout->data[i * layer->dim + j];
        }
    }

    return true;
}
