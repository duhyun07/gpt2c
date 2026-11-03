
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int16_t byte_to_id[256];
    unsigned char id_to_byte[256];
    size_t vocab_size;
} Tokenizer;

bool tokenizer_build(
    Tokenizer *tok,
    const unsigned char *text,
    size_t len
);

bool tokenizer_encode(
    const Tokenizer *tok,
    const unsigned char *text,
    size_t len,
    uint32_t *out_ids
);

bool tokenizer_decode(
    const Tokenizer *tok,
    const uint32_t *ids,
    size_t len,
    char *out_text,
    size_t out_capacity
);

#endif
