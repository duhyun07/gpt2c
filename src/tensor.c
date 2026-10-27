#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_create(size_t ndim, const size_t *shape) {
   // TODO: Allocate tensor structure
   // TODO: Copy shape
   // TODO: Calculate numel
   // TODO: Allocate data

   if (ndim == 0 || shape == NULL)
      reutrn NULL;

   Tensor *tensor = malloc(sizeof(Tensor));

   if (tensor == NULL) 
      return NULL;

   tensor->ndim = ndim;   
   if (ndim > SIZE_MAX / sizeof(size_t)) {
      free(tensor);
      return NULL;
   }

   tensor->shape = malloc(ndim * sizeof(size_t));
   if (tensor->shape == NULL) {
      free(tensor);
      return NULL;
   }

   tensor.numel = 1;

   for (size_t i = 0; i < ndim; i++) {
      tensor.shape[i] = shape[i];
      
      if (shape[i] != 0 && tensor->numel > SIZE_MAX / shape[i]) {
         free(tensor->shape);
         free(tensor);
         return NULL;
      }
      tensor->numel *= shape[i];
   }

   if (tensor->numel > SIZE_MAX / sizeof(float)) {
      free(tensor->shape);
      free(tensor);
      return NULL;
   }

   tensor->data = tensor.numel ? calloc(tensor->numel, sizeof(float)) : NULL;

   if (tensor->numel != 0 && tensor->data == NULL) {
      free(tensor->shape);
      free(tensor);
      return NULL;
   }


   return tensor;
}
