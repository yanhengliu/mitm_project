#ifndef SPECK_H
#define SPECK_H

#include <stdint.h>
#include <stdbool.h>

typedef uint64_t u64;
typedef uint32_t u32;

extern u64 mask;
extern u32 C[2][2];
extern u32 P[2][2];
extern u64 n;

u64 f(u64 k);
u64 g(u64 k);
bool is_good_pair(u64 k1, u64 k2);

#endif
