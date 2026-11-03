#include <stdbool.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
   float *data;
   size_t *shape;
   size_t ndim;
   size_t numel;
} Tensor;

Tensor *tensor_create(size_t ndim, const size_t *shape)
{
    if (ndim == 0 || shape == NULL) {
        return NULL;
    }

    size_t numel = 1;

    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] == 0) {
            return NULL;
        }

        numel *= shape[i];
    }

    Tensor *tensor = malloc(sizeof(Tensor));

    if (tensor == NULL) {
        return NULL;
    }

    tensor->shape = malloc(ndim * sizeof(size_t));

    if (tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(size_t));

    tensor->data = malloc(numel * sizeof(float));

    if (tensor->data == NULL) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->numel = numel;

    return tensor;
}

void tensor_free(Tensor *tensor) {
    if (tensor == NULL) {
        return;
    }

    free(tensor->data);
    free(tensor->shape);
    free(tensor);
}

void tensor_fill(Tensor *tensor, float value) {
    if (tensor == NULL || tensor->data == NULL) {
        return;
    }

    for (size_t i = 0; i < tensor->numel; i++) {
        tensor->data[i] = value;
    }
}

static bool tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL || a->shape == NULL || b->shape == NULL) {
        return false;
    }

    if (a->ndim == 0 || a->ndim != b->ndim || a->numel != b->numel) {
        return false;
    }

    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    return true;
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
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

Tensor *tensor_mul(const Tensor *a, const Tensor *b) {
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

int main() {
    size_t shape[] = {2, 3};
        
    Tensor *a = tensor_create(2, shape);
    Tensor *b = tensor_create(2, shape);

    if (a == NULL || b == NULL) {
        tensor_free(a);
        tensor_free(b);
        return 1;
    }

    for (size_t i = 0; i < a->numel; i++) {
        a->data[i] = (float)(i + 1);
        b->data[i] = (float)((i + 1) * 10);
    }

    Tensor *c = tensor_mul(a, b);
    Tensor *d = tensor_add(a, b);

    
    if (c == NULL || d == NULL) {
        printf("Tensor addition failed\n");
        tensor_free(a);
        tensor_free(b);
        tensor_free(d);
        tensor_free(b);
        return 1;
    }

    for (size_t i = 0; i < a->numel; i++) {
        printf("add[%zu] = %.1f\n", i, c->data[i]);
        printf("mul[%zu] = %.1f\n", i, d->data[i]);
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);

    return 0;
    
}
