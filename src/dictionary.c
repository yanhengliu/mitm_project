#include "dictionary.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define EMPTY 0xffffffff
#define PRIME 0xfffffffbULL

static uint64_t dict_size;
static struct entry *A;

/* Used to record the number of times Insert and Probe actually occur in the hash table */
static uint64_t dictionary_insert_calls = 0;
static uint64_t dictionary_probe_calls = 0;

void dict_setup(uint64_t size)
{
    dict_size = size;
    A = malloc(sizeof(*A)*dict_size);
    if(!A) {
        fprintf(stderr, "Could not allocate dictionary of size %"PRIu64"\n", dict_size);
        exit(1);
    }
    for(uint64_t i=0; i<dict_size; i++)
        A[i].k = EMPTY;
}

void dict_insert(uint64_t key, uint64_t value)
{
    // Record an insert operation
    dictionary_insert_calls++;

    uint64_t h = murmur64(key) % dict_size;
    for(;;) {
        if (A[h].k == EMPTY) {
            A[h].k = (uint32_t)(key % PRIME);
            A[h].v = value;
            return;
        }
        h += 1;
        if(h == dict_size) h=0;
    }
}

int dict_probe(uint64_t key, int maxval, uint64_t values[])
{
    // Record an search operation
    dictionary_probe_calls++;

    uint32_t k = (uint32_t)(key % PRIME);
    uint64_t h = murmur64(key) % dict_size;
    int nval=0;
    for(;;) {
        if(A[h].k == EMPTY) {
            return nval;
        }
        if(A[h].k == k) {
            if(nval == maxval) return -1;
            values[nval++] = A[h].v;
        }
        h+=1;
        if(h == dict_size) h=0;
    }
}

void dictionary_get_usage(uint64_t *p_insert_calls, uint64_t *p_probe_calls)
{
    // return the value of a local static variable to the caller
    *p_insert_calls = dictionary_insert_calls;
    *p_probe_calls  = dictionary_probe_calls;
}
