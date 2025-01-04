#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <stdint.h>

struct entry {
    uint32_t k;
    uint64_t v;
} __attribute__ ((packed));

void dict_setup(uint64_t size);
void dict_insert(uint64_t key, uint64_t value);
int dict_probe(uint64_t key, int maxval, uint64_t values[]);

void dictionary_get_usage(uint64_t *p_insert_calls, uint64_t *p_probe_calls);

#endif
