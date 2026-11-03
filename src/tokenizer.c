#include "tokenizer.h"
#include <string.h>

bool tokenizer_build(
    Tokenizer *tok,
    const unsigned char *text,
    size_t len
)
{
    if (tok == NULL || text == NULL || len == 0) {
        return false;
    }

    bool seen[256] = {false};

    for (size_t i = 0; i < 256; i++) {
        tok->byte_to_id[i] = -1;
        tok->id_to_byte[i] = 0;
    }

    tok->vocab_size = 0;

    for (size_t i = 0; i < len; i++) {
        unsigned char c = text[i];

        if (c >= 128) {
            return false;
        }

        seen[c] = true;
    }

    for (size_t c = 0; c < 128; c++) {
        if (!seen[c]) {
            continue;
        }

        size_t id = tok->vocab_size;

        tok->byte_to_id[c] = (int16_t)id;
        tok->id_to_byte[id] = (unsigned char)c;

        tok->vocab_size++;
    }

    return tok->vocab_size > 0;
}

bool tokenizer_encode(
    const Tokenizer *tok,
    const unsigned char *text,
    size_t len,
    uint32_t *out_ids
)
{
    if (tok == NULL ||
        (len > 0 && (text == NULL || out_ids == NULL))) {
        return false;
    }

    for (size_t i = 0; i < len; i++) {
        if (tok->byte_to_id[text[i]] < 0) {
            return false;
        }
    }

    for (size_t i = 0; i < len; i++) {
        out_ids[i] = (uint32_t)tok->byte_to_id[text[i]];
    }

    return true;
}

bool tokenizer_decode(
    const Tokenizer *tok,
    const uint32_t *ids,
    size_t len,
    char *out_text,
    size_t out_capacity
)
{
    if (tok == NULL || out_text == NULL ||
        (len > 0 && ids == NULL)) {
        return false;
    }

    if (out_capacity <= len) {
        return false;
    }

    for (size_t i = 0; i < len; i++) {
        if (ids[i] >= tok->vocab_size) {
            return false;
        }
    }

    for (size_t i = 0; i < len; i++) {
        out_text[i] = (char)tok->id_to_byte[ids[i]];
    }

    out_text[len] = '\0';

    return true;
}