#include "attention.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

static bool valid_input(
    const Attention *a,
    const Tensor *x
)
{
    if (!a ||
        !a->q_proj || !a->k_proj ||
        !a->v_proj || !a->out_proj ||
        !x || !x->data || !x->shape ||
        x->ndim != 3 ||
        a->embed_dim == 0 ||
        a->num_heads == 0 ||
        a->embed_dim % a->num_heads != 0 ||
        x->shape[0] == 0 ||
        x->shape[1] == 0 ||
        x->shape[2] != a->embed_dim) {
        return false;
    }

    size_t B = x->shape[0];
    size_t T = x->shape[1];
    size_t C = x->shape[2];

    if (B > SIZE_MAX / T ||
        B * T > SIZE_MAX / C) {
        return false;
    }

    return x->numel == B * T * C;
}

static bool shape_btc(
    const Tensor *x,
    size_t B,
    size_t T,
    size_t C
)
{
    return x && x->data && x->shape &&
           x->ndim == 3 &&
           x->shape[0] == B &&
           x->shape[1] == T &&
           x->shape[2] == C &&
           x->numel == B * T * C;
}

Attention *attention_create(
    size_t embed_dim,
    size_t num_heads,
    uint64_t seed
)
{
    if (embed_dim == 0 ||
        num_heads == 0 ||
        embed_dim % num_heads != 0) {
        return NULL;
    }

    Attention *a = calloc(1, sizeof(*a));

    if (!a) {
        return NULL;
    }

    a->embed_dim = embed_dim;
    a->num_heads = num_heads;
    a->head_dim = embed_dim / num_heads;

    a->q_proj = linear_create(
        embed_dim, embed_dim, seed
    );

    a->k_proj = linear_create(
        embed_dim, embed_dim, seed + 1
    );

    a->v_proj = linear_create(
        embed_dim, embed_dim, seed + 2
    );

    a->out_proj = linear_create(
        embed_dim, embed_dim, seed + 3
    );

    if (!a->q_proj || !a->k_proj ||
        !a->v_proj || !a->out_proj) {
        attention_free(a);
        return NULL;
    }

    return a;
}

void attention_free(Attention *a)
{
    if (!a) {
        return;
    }

    linear_free(a->q_proj);
    linear_free(a->k_proj);
    linear_free(a->v_proj);
    linear_free(a->out_proj);

    free(a);
}

void attention_zero_grad(Attention *a)
{
    if (!a) {
        return;
    }

    linear_zero_grad(a->q_proj);
    linear_zero_grad(a->k_proj);
    linear_zero_grad(a->v_proj);
    linear_zero_grad(a->out_proj);
}

void attention_cache_free(AttentionCache *cache)
{
    if (!cache) {
        return;
    }

    tensor_free(cache->q);
    tensor_free(cache->k);
    tensor_free(cache->v);
    tensor_free(cache->probs);
    tensor_free(cache->context);

    free(cache);
}


static double attention_score(
    const Attention *a,
    const AttentionCache *cache,
    size_t b,
    size_t h,
    size_t i,
    size_t j,
    size_t T
)
{
    size_t C = a->embed_dim;
    size_t D = a->head_dim;

    size_t qi = (b * T + i) * C + h * D;
    size_t kj = (b * T + j) * C + h * D;

    double score = 0.0;

    for (size_t d = 0; d < D; d++) {
        score +=
            (double)cache->q->data[qi + d] *
            cache->k->data[kj + d];
    }

    return score / sqrt((double)D);
}


Tensor *attention_forward(
    const Attention *a,
    const Tensor *x,
    AttentionCache **out_cache
)
{
    if (!out_cache) {
        return NULL;
    }

    *out_cache = NULL;

    if (!valid_input(a, x)) {
        return NULL;
    }

    size_t B = x->shape[0];
    size_t T = x->shape[1];
    size_t C = a->embed_dim;
    size_t H = a->num_heads;
    size_t D = a->head_dim;

    AttentionCache *cache = calloc(
        1, sizeof(*cache)
    );

    if (!cache) {
        return NULL;
    }

    cache->q = linear_forward(a->q_proj, x);
    cache->k = linear_forward(a->k_proj, x);
    cache->v = linear_forward(a->v_proj, x);

    size_t prob_shape[] = {B, H, T, T};

    cache->probs = tensor_zeros(4, prob_shape);

    cache->context = tensor_zeros(3, x->shape);

    if (!cache->q || !cache->k || !cache->v ||
        !cache->probs || !cache->context) {
        goto fail;
    }

    for (size_t b = 0; b < B; b++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t i = 0; i < T; i++) {

                size_t pbase =
                    ((b * H + h) * T + i) * T;

                double max_score = -INFINITY;

                for (size_t j = 0; j <= i; j++) {
                    double s = attention_score(
                        a, cache, b, h, i, j, T
                    );

                    if (!isfinite(s)) {
                        goto fail;
                    }

                    if (s > max_score) {
                        max_score = s;
                    }
                }

                double denom = 0.0;

                for (size_t j = 0; j <= i; j++) {
                    double s = attention_score(
                        a, cache, b, h, i, j, T
                    );

                    double e = exp(s - max_score);

                    cache->probs->data[pbase + j] =
                        (float)e;

                    denom += e;
                }

                for (size_t j = 0; j <= i; j++) {
                    cache->probs->data[pbase + j] =
                        (float)(
                            (double)cache->probs->data[
                                pbase + j
                            ] / denom
                        );
                }

                size_t out_base =
                    (b * T + i) * C + h * D;

                for (size_t d = 0; d < D; d++) {
                    double z = 0.0;

                    for (size_t j = 0; j <= i; j++) {
                        size_t v_idx =
                            (b * T + j) * C + h * D + d;

                        z +=
                            (double)cache->probs->data[
                                pbase + j
                            ] * cache->v->data[v_idx];
                    }

                    cache->context->data[
                        out_base + d
                    ] = (float)z;
                }
            }
        }
    }

    Tensor *out = linear_forward(
        a->out_proj, cache->context
    );

    if (!out) {
        goto fail;
    }

    *out_cache = cache;

    return out;

fail:
    attention_cache_free(cache);
    return NULL;
}

Tensor *attention_backward(
    Attention *a,
    const Tensor *x,
    const Tensor *dout,
    const AttentionCache *cache
)
{
    if (!valid_input(a, x) || !cache) {
        return NULL;
    }

    size_t B = x->shape[0];
    size_t T = x->shape[1];
    size_t C = a->embed_dim;
    size_t H = a->num_heads;
    size_t D = a->head_dim;

    if (!shape_btc(dout, B, T, C) ||
        !shape_btc(cache->q, B, T, C) ||
        !shape_btc(cache->k, B, T, C) ||
        !shape_btc(cache->v, B, T, C) ||
        !shape_btc(cache->context, B, T, C) ||
        !cache->probs ||
        !cache->probs->data ||
        !cache->probs->shape ||
        cache->probs->ndim != 4 ||
        cache->probs->shape[0] != B ||
        cache->probs->shape[1] != H ||
        cache->probs->shape[2] != T ||
        cache->probs->shape[3] != T ||
        T > SIZE_MAX / sizeof(double)) {
        return NULL;
    }

    Tensor *dcontext = NULL;
    Tensor *dq = NULL;
    Tensor *dk = NULL;
    Tensor *dv = NULL;

    Tensor *dx = NULL;
    Tensor *dx_k = NULL;
    Tensor *dx_v = NULL;

    double *dp = NULL;

    dcontext = linear_backward(
        a->out_proj,
        cache->context,
        dout
    );

    dq = tensor_zeros(3, x->shape);
    dk = tensor_zeros(3, x->shape);
    dv = tensor_zeros(3, x->shape);

    dp = malloc(T * sizeof(double));

    if (!dcontext || !dq || !dk || !dv || !dp) {
        goto fail;
    }

    double scale = 1.0 / sqrt((double)D);

    for (size_t b = 0; b < B; b++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t i = 0; i < T; i++) {

                size_t pbase =
                    ((b * H + h) * T + i) * T;

                size_t qi =
                    (b * T + i) * C + h * D;

                double weighted_sum = 0.0;

                for (size_t j = 0; j <= i; j++) {
                    size_t kj =
                        (b * T + j) * C + h * D;

                    double p =
                        cache->probs->data[pbase + j];

                    double dprob = 0.0;

                    for (size_t d = 0; d < D; d++) {
                        double go =
                            dcontext->data[qi + d];

                        dprob +=
                            go * cache->v->data[kj + d];

                        dv->data[kj + d] +=
                            (float)(p * go);
                    }

                    dp[j] = dprob;
                    weighted_sum += p * dprob;
                }

                for (size_t j = 0; j <= i; j++) {
                    size_t kj =
                        (b * T + j) * C + h * D;

                    double p =
                        cache->probs->data[pbase + j];

                    double ds =
                        p * (dp[j] - weighted_sum)
                        * scale;

                    for (size_t d = 0; d < D; d++) {
                        dq->data[qi + d] +=
                            (float)(
                                ds * cache->k->data[
                                    kj + d
                                ]
                            );

                        dk->data[kj + d] +=
                            (float)(
                                ds * cache->q->data[
                                    qi + d
                                ]
                            );
                    }
                }
            }
        }
    }

    dx = linear_backward(a->q_proj, x, dq);

    dx_k = linear_backward(a->k_proj, x, dk);

    dx_v = linear_backward(a->v_proj, x, dv);

    if (!dx || !dx_k || !dx_v) {
        goto fail;
    }

    for (size_t n = 0; n < x->numel; n++) {
        dx->data[n] +=
            dx_k->data[n] + dx_v->data[n];
    }

    free(dp);
    tensor_free(dcontext);
    tensor_free(dq);
    tensor_free(dk);
    tensor_free(dv);
    tensor_free(dx_k);
    tensor_free(dx_v);

    return dx;

fail:
    free(dp);
    tensor_free(dcontext);
    tensor_free(dq);
    tensor_free(dk);
    tensor_free(dv);
    tensor_free(dx);
    tensor_free(dx_k);
    tensor_free(dx_v);

    return NULL;
}