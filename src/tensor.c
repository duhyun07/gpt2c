#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_create(size_t ndim, const size_t *shape)
{
   if (ndim == 0 || shape == NULL) {
      return NULL;
   }

   if (ndim > __SIZE_MAX__ / sizeof(size_t)) {
      return NULL;
   }

   size_t numel = 1;

   for (size_t i = 0; i < ndim; i ++) {
      if (shape[i] == 0) {
         return NULL;
      }

      if (numel > __SIZE_MAX__ / shape[i]) {
         return NULL;
      }

      numel *= shape[i];
   }

   if (numel > __SIZE_MAX__ / sizeof(float)) {
      return NULL;
   }

   Tensor *tensor = malloc(sizeof(Tensor));

   if (tensor == NULL) {
      return NULL;
   }

   tensor->data = NULL;
   tensor->shape = NULL;
   tensor->ndim = ndim;
   tensor->numel = numel;

   tensor->shape = malloc(ndim * sizeof(size_t));

   if (tensor->shape == NULL) {
      tensor_free(tensor);
      return NULL;
   }

   memcpy(tensor->shape, shape, ndim * sizeof(size_t));

   tensor->data = malloc(numel * sizeof(float));

   if (tensor->data == NULL) {
      tensor_free(tensor);
      return NULL;
   }

   return tensor;
}

Tensor *tensor_zeros(size_t ndim, const size_t *shape)
{
   Tensor *tensor = tensor_create(ndim, shape);

   if (tensor == NULL) {
      return NULL;
   }

   tensor_fill(tensor, 0.0f);

   return tensor;
}

Tensor *tensor_clone(const Tensor *src)
{
   if (src == NULL || src->shape == NULL || src->data == NULL) {
      return NULL;
   }

   Tensor *clone = tensor_create(src->ndim, src->shape);
   
    if (clone == NULL) {
      return NULL;
    }

    if (clone->numel != src->numel) {
      tensor_free(clone);
      return NULL;
    }

    memcpy(
      clone->data,
      src->data,
      src->numel * sizeof(float)
    );

    return clone;
}

void tensor_free(Tensor *tensor)
{
   if (tensor == NULL) {
      return;
   }

   free(tensor->data);
   free(tensor->shape);
   free(tensor);
}

void tensor_fill(Tensor *tensor, float value)
{
   if (tensor == NULL || tensor->data == NULL) {
      return;
   }

   for (size_t i = 0; i < tensor->numel; i++) {
      tensor->data[i] = value;
   }
}

bool tensor_flat_index(
    const Tensor *tensor,
    const size_t *indices,
    size_t *out_index
)
{
   if (tensor == NULL || indices == NULL || out_index == NULL) {
      return false;
   }

   if (tensor->shape == NULL | tensor->ndim == 0) {
      return false;
   }

   size_t flat_idx = 0;

   for (size_t i = 0; i < tensor->ndim; i++) {
      if (tensor->shape[i] == 0 || indices[i] >= tensor->shape[i]) {
         return false;
      }

      if (flat_idx > (__SIZE_MAX__ - indices[i]) / tensor->shape[i]) {
         return false;
      }

      flat_idx = flat_idx * tensor->shape[i] + indices[i];
   }

   if (flat_idx >= tensor->numel) {
      return false;
   }

   *out_index = flat_idx;

   return true;
}

bool tensor_get(
    const Tensor *tensor,
    const size_t *indices,
    float *out_value
)
{
   if (tensor == NULL || tensor->data == NULL || out_value == NULL) {
      return false;
   }

   size_t flat_idx;

   if (!tensor_flat_index(tensor, indices, &flat_idx)) {
      return false;
   }

   *out_value = tensor->data[flat_idx];

   return true;
}

bool tensor_set(
    Tensor *tensor,
    const size_t *indices,
    float value
)
{
   if (tensor == NULL || tensor->data == NULL) {
      return false;
   }

   size_t flat_idx;

   if (!tensor_flat_index(tensor, indices, &flat_idx)) {
      return false;
   }

   tensor->data[flat_idx] = value;

   return true;
}


static void tensor_print_recursive(
    const Tensor *tensor,
    size_t dim,
    size_t offset
)
{
    printf("[");

    if (dim == tensor->ndim - 1) {
        for (size_t i = 0; i < tensor->shape[dim]; i++) {
            if (i > 0) printf(", ");
            printf("%.6g", tensor->data[offset + i]);
        }
    } else {
        size_t stride = 1;

        for (size_t j = dim + 1; j < tensor->ndim; j++) {
            stride *= tensor->shape[j];
        }

        for (size_t i = 0; i < tensor->shape[dim]; i++) {
            if (i > 0) {
                printf(",\n");
                for (size_t j = 0; j <= dim; j++) {
                    printf(" ");
                }
            }

            tensor_print_recursive(
                tensor,
                dim + 1,
                offset + i * stride
            );
        }
    }

    printf("]");
}

void tensor_print(const Tensor *tensor)
{
    if (tensor == NULL ||
        tensor->data == NULL ||
        tensor->shape == NULL ||
        tensor->ndim == 0) {
        printf("Tensor(NULL)\n");
        return;
    }

    printf("Tensor(shape=[");

    for (size_t i = 0; i < tensor->ndim; i++) {
        if (i > 0) printf(", ");
        printf("%zu", tensor->shape[i]);
    }

    printf("])\n");

    tensor_print_recursive(tensor, 0, 0);
    printf("\n");
}


static bool tensor_same_shape(
    const Tensor *a,
    const Tensor *b
)
{
    if (a == NULL || b == NULL ||
        a->data == NULL || b->data == NULL ||
        a->shape == NULL || b->shape == NULL) {
        return false;
    }

    if (a->ndim == 0 || a->ndim != b->ndim ||
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



Tensor *tensor_add(const Tensor *a, const Tensor *b)
{
    if (!tensor_same_shape(a, b)) {
        return NULL;
    }

    Tensor *result = tensor_create(a->ndim, a->shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < a->numel; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}



Tensor *tensor_mul(const Tensor *a, const Tensor *b)
{
    if (!tensor_same_shape(a, b)) {
        return NULL;
    }

    Tensor *result = tensor_create(a->ndim, a->shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < a->numel; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}


Tensor *tensor_exp(const Tensor *src)
{
    if (src == NULL || src->data == NULL ||
        src->shape == NULL) {
        return NULL;
    }

    Tensor *result = tensor_create(src->ndim, src->shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < src->numel; i++) {
        result->data[i] = expf(src->data[i]);
    }

    return result;
}

Tensor *tensor_log(const Tensor *src)
{
    if (src == NULL || src->data == NULL ||
        src->shape == NULL) {
        return NULL;
    }

    Tensor *result = tensor_create(src->ndim, src->shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < src->numel; i++) {
        result->data[i] = logf(src->data[i]);
    }

    return result;
}

Tensor *tensor_gelu(const Tensor *src)
{
    if (src == NULL || src->data == NULL ||
        src->shape == NULL) {
        return NULL;
    }

    Tensor *result = tensor_create(src->ndim, src->shape);

    if (result == NULL) {
        return NULL;
    }

    const float GELU_COEFF = 0.7978845608f;
    const float GELU_CUBIC = 0.044715f;

    for (size_t i = 0; i < src->numel; i++) {
        float x = src->data[i];

        float x3 = x * x * x;

        float inner =
            GELU_COEFF * (x + GELU_CUBIC * x3);

        result->data[i] =
            0.5f * x * (1.0f + tanhf(inner));
    }

    return result;
}


Tensor *tensor_matmul(const Tensor *a, const Tensor *b)
{
    if (a == NULL || b == NULL ||
        a->data == NULL || b->data == NULL ||
        a->shape == NULL || b->shape == NULL) {
        return NULL;
    }

    // Only 2D matrices are supported
    if (a->ndim != 2 || b->ndim != 2) {
        return NULL;
    }

    size_t M = a->shape[0];
    size_t K = a->shape[1];

    if (K != b->shape[0]) {
        return NULL;
    }

    size_t N = b->shape[1];

    size_t result_shape[] = {M, N};

    Tensor *result = tensor_create(2, result_shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
                sum += a->data[i * K + k]
                     * b->data[k * N + j];
            }

            result->data[i * N + j] = sum;
        }
    }

    return result;
}



Tensor *tensor_transpose(const Tensor *src)
{
    if (src == NULL ||
        src->data == NULL ||
        src->shape == NULL) {
        return NULL;
    }

    if (src->ndim != 2) {
        return NULL;
    }

    size_t rows = src->shape[0];
    size_t cols = src->shape[1];

    size_t result_shape[] = {cols, rows};

    Tensor *result = tensor_create(2, result_shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {

            result->data[j * rows + i] =
                src->data[i * cols + j];

        }
    }

    return result;
}

bool tensor_sum(const Tensor *src, float *out_sum)
{
    if (src == NULL || src->data == NULL ||
        src->numel == 0 || out_sum == NULL) {
        return false;
    }

    double sum = 0.0;

    for (size_t i = 0; i < src->numel; i++) {
        sum += src->data[i];
    }

    *out_sum = (float)sum;

    return true;
}

bool tensor_mean(const Tensor *src, float *out_mean)
{
    if (src == NULL || src->data == NULL ||
        src->numel == 0 || out_mean == NULL) {
        return false;
    }

    double sum = 0.0;

    for (size_t i = 0; i < src->numel; i++) {
        sum += src->data[i];
    }

    *out_mean = (float)(sum / (double)src->numel);

    return true;
}

Tensor *tensor_softmax(const Tensor *src)
{
    if (src == NULL || src->data == NULL ||
        src->shape == NULL || src->ndim == 0) {
        return NULL;
    }

    size_t cols = src->shape[src->ndim - 1];

    if (cols == 0 || src->numel % cols != 0) {
        return NULL;
    }

    size_t rows = src->numel / cols;

    Tensor *result = tensor_create(src->ndim, src->shape);

    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < rows; i++) {
        size_t offset = i * cols;

        // Find maximum value
        float max_val = src->data[offset];

        for (size_t j = 0; j < cols; j++) {
            float x = src->data[offset + j];

            if (!isfinite(x)) {
                tensor_free(result);
                return NULL;
            }

            if (x > max_val) {
                max_val = x;
            }
        }

        // Calculate exponentials
        float sum = 0.0f;

        for (size_t j = 0; j < cols; j++) {
            float x = src->data[offset + j];

            float e = expf(x - max_val);

            result->data[offset + j] = e;
            sum += e;
        }

        // Normalize probabilities
        for (size_t j = 0; j < cols; j++) {
            result->data[offset + j] /= sum;
        }
    }

    return result;
}
