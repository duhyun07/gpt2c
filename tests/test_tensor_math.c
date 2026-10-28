
#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static bool almost_equal(float a, float b)
{
    return fabsf(a - b) < 1e-4f;
}

int main(void)
{
    size_t shape[] = {2, 3};

    Tensor *a = tensor_create(2, shape);
    assert(a != NULL);

    for (size_t i = 0; i < a->numel; i++) {
        a->data[i] = (float)(i + 1);
    }

    // 1. Sum / Mean
    float sum, mean;

    assert(tensor_sum(a, &sum));
    assert(tensor_mean(a, &mean));

    assert(almost_equal(sum, 21.0f));
    assert(almost_equal(mean, 3.5f));

    // 2. Exp / Log
    Tensor *exp_result = tensor_exp(a);
    assert(exp_result != NULL);

    Tensor *log_result = tensor_log(exp_result);
    assert(log_result != NULL);

    for (size_t i = 0; i < a->numel; i++) {
        assert(almost_equal(
            log_result->data[i],
            a->data[i]
        ));
    }

    // 3. Softmax
    Tensor *softmax = tensor_softmax(a);
    assert(softmax != NULL);

    for (size_t row = 0; row < 2; row++) {
        float row_sum = 0.0f;

        for (size_t col = 0; col < 3; col++) {
            float p = softmax->data[row * 3 + col];

            assert(p >= 0.0f && p <= 1.0f);
            row_sum += p;
        }

        assert(almost_equal(row_sum, 1.0f));
    }

    // 4. Softmax numerical stability
    size_t small_shape[] = {2};

    Tensor *large = tensor_create(1, small_shape);
    assert(large != NULL);

    large->data[0] = 1000.0f;
    large->data[1] = 1001.0f;

    Tensor *stable = tensor_softmax(large);
    assert(stable != NULL);

    assert(isfinite(stable->data[0]));
    assert(isfinite(stable->data[1]));

    assert(almost_equal(
        stable->data[0] + stable->data[1],
        1.0f
    ));

    // 5. GELU
    Tensor *gelu_input = tensor_create(1, small_shape);
    assert(gelu_input != NULL);

    gelu_input->data[0] = -1.0f;
    gelu_input->data[1] = 1.0f;

    Tensor *gelu = tensor_gelu(gelu_input);
    assert(gelu != NULL);

    assert(fabsf(gelu->data[0] - (-0.1588f)) < 1e-3f);
    assert(fabsf(gelu->data[1] - 0.8412f) < 1e-3f);

    // Cleanup
    tensor_free(a);
    tensor_free(exp_result);
    tensor_free(log_result);
    tensor_free(softmax);
    tensor_free(large);
    tensor_free(stable);
    tensor_free(gelu_input);
    tensor_free(gelu);

    printf("All tensor math tests passed!\n");

    return 0;
}
