
#include "tensor.h"

#include <assert.h>
#include <stdio.h>

void test_tensor_indexing(void)
{
    size_t shape[] = {2, 3, 4};

    Tensor *tensor = tensor_zeros(3, shape);
    assert(tensor != NULL);

    // 1. Test flat index
    size_t indices[] = {1, 2, 3};
    size_t flat_index = 0;

    assert(tensor_flat_index(tensor, indices, &flat_index));
    assert(flat_index == 23);

    // 2. Test set
    assert(tensor_set(tensor, indices, 42.0f));
    assert(tensor->data[23] == 42.0f);

    // 3. Test get
    float value = 0.0f;

    assert(tensor_get(tensor, indices, &value));
    assert(value == 42.0f);

    // 4. Test first element
    size_t first[] = {0, 0, 0};

    assert(tensor_set(tensor, first, 10.0f));
    assert(tensor_get(tensor, first, &value));
    assert(value == 10.0f);

    // 5. Test out-of-bounds access
    size_t invalid[] = {2, 0, 0};

    assert(!tensor_flat_index(tensor, invalid, &flat_index));
    assert(!tensor_get(tensor, invalid, &value));
    assert(!tensor_set(tensor, invalid, 99.0f));

    // 6. Test NULL handling
    assert(!tensor_get(NULL, indices, &value));
    assert(!tensor_set(NULL, indices, 1.0f));
    assert(!tensor_get(tensor, indices, NULL));

    tensor_free(tensor);

    printf("Tensor indexing tests passed!\n");
}

int main(void)
{
    test_tensor_indexing();
    return 0;
}
