#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    float *data;
    size_t *shape;
    size_t ndim;
    size_t numel;
} Tensor;

Tensor *tensor_create(size_t ndim, const size_t *shape);
Tensor *tensor_zeros(size_t ndim, const size_t *shape);
Tensor *tensor_clone(const Tensor *src);
void tensor_free(Tensor *tensor);

void tensor_fill(Tensor *tensor, float value);

bool tensor_flat_index(
    const Tensor *tensor,
    const size_t *indices,
    size_t *out_index
);

bool tensor_get(
    const Tensor *tensor,
    const size_t *indices,
    float *out_value
);

bool tensor_set(
    Tensor *tensor,
    const size_t *indices,
    float value
);

void tensor_print(const Tensor *tensor);

Tensor *tensor_add(const Tensor *a, const Tensor *b);
Tensor *tensor_mul(const Tensor *a, const Tensor *b);
Tensor *tensor_exp(const Tensor *src);
Tensor *tensor_log(const Tensor *src);
Tensor *tensor_gelu(const Tensor *src);

Tensor *tensor_matmul(const Tensor *a, const Tensor *b);
Tensor *tensor_transpose(const Tensor *src);

bool tensor_sum(const Tensor *src, float *out_sum);
bool tensor_mean(const Tensor *src, float *out_mean);

Tensor *tensor_softmax(const Tensor *src);

#endif // TENSOR_H
