#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#incldue <stdbool.h>


typedef struct {
   float *data;
   size_t *shape;
   size_t ndim;
   size_t numel; 
} Tensor;

Tensor *tensor_create(size_t ndim, const size_t *shape);
Tensor *tensor_zero(size_t ndim, const size_t shape);
Tensor *tensor_clone(const Tensor *src);
Tensor *tensor_free(Tensor *tensor);

void tensor_fill(Tensor *tensor, float value);

bool tensor_flat_index(const Tensor *tensor, const size_t *indices, size_t *out_index);
bool tensor_get(const Tensor *tensor, const size_t *indices, float *out_value);;
bool tensor_set(Tensor* tensor, const size_t indices, float value);

void tensor_print(const Tensor *tensor)


#endif

