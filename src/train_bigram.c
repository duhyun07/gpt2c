#include "bigram.h"
#include "dataset.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <dataset.txt>\n", argv[0]);
        return 1;
    }

    const size_t BATCH_SIZE = 8;
    const size_t CONTEXT_LENGTH = 32;
    const size_t TRAIN_STEPS = 1000;
    const size_t GENERATE_LENGTH = 200;

    const float LEARNING_RATE = 0.5f;

    uint64_t train_seed = 12345;
    uint64_t val_seed = 54321;
    uint64_t gen_seed = 99999;

    int status = EXIT_FAILURE;

    Dataset *dataset = NULL;
    TokenBatch *train_batch = NULL;
    TokenBatch *val_batch = NULL;
    Bigram *model = NULL;

    dataset = dataset_load(argv[1]);

    if (dataset == NULL) {
        fprintf(stderr, "Failed to load dataset\n");
        goto cleanup;
    }

    printf("Vocabulary size: %zu\n",
           dataset->tokenizer.vocab_size);

    model = bigram_create(
        dataset->tokenizer.vocab_size,
        42
    );

    train_batch = token_batch_create(
        BATCH_SIZE, CONTEXT_LENGTH
    );

    val_batch = token_batch_create(
        BATCH_SIZE, CONTEXT_LENGTH
    );

    if (model == NULL ||
        train_batch == NULL ||
        val_batch == NULL) {
        fprintf(stderr, "Initialization failed\n");
        goto cleanup;
    }

    for (size_t step = 1; step <= TRAIN_STEPS; step++) {
        float train_loss = 0.0f;

        if (!dataset_sample_batch(
                dataset,
                DATASET_TRAIN,
                train_batch,
                &train_seed
            )) {
            fprintf(stderr, "Train batch failed\n");
            goto cleanup;
        }

        if (!bigram_loss_backward(
                model, train_batch, &train_loss
            )) {
            fprintf(stderr, "Forward/backward failed\n");
            goto cleanup;
        }

        if (!bigram_sgd(model, LEARNING_RATE)) {
            fprintf(stderr, "SGD update failed\n");
            goto cleanup;
        }

        if (step == 1 || step % 100 == 0) {
            float val_loss = 0.0f;

            if (!dataset_sample_batch(
                    dataset,
                    DATASET_VAL,
                    val_batch,
                    &val_seed
                ) ||
                !bigram_loss(
                    model, val_batch, &val_loss
                )) {
                fprintf(stderr, "Validation failed\n");
                goto cleanup;
            }

            printf(
                "Step %4zu | Train Loss: %.4f"
                " | Val Loss: %.4f\n",
                step, train_loss, val_loss
            );
        }
    }

    uint32_t generated[201];
    char output[202];

    generated[0] = dataset->tokens[0];

    if (!bigram_generate(
            model,
            generated[0],
            GENERATE_LENGTH,
            1.0f,
            &gen_seed,
            &generated[1]
        )) {
        fprintf(stderr, "Generation failed\n");
        goto cleanup;
    }

    if (!tokenizer_decode(
            &dataset->tokenizer,
            generated,
            GENERATE_LENGTH + 1,
            output,
            sizeof(output)
        )) {
        fprintf(stderr, "Decoding failed\n");
        goto cleanup;
    }

    printf("\n=== Generated Text ===\n");
    printf("%s\n", output);

    status = EXIT_SUCCESS;

cleanup:
    token_batch_free(train_batch);
    token_batch_free(val_batch);
    bigram_free(model);
    dataset_free(dataset);

    return status;
}