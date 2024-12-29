#ifndef DISTRIBUTED_DICT_H
#define DISTRIBUTED_DICT_H

#include <stdint.h>

void distributed_dict_setup(uint64_t n, uint64_t mask, uint32_t C[2][2]);
void distributed_dict_build();
int distributed_dict_search(int maxres, uint64_t k1[], uint64_t k2[]);

#endif
