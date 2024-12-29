#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>

double wtime();
void human_format(uint64_t n, char *target);
uint64_t murmur64(uint64_t x);

void usage(char **argv);
void process_command_line_options(int argc, char **argv, uint64_t *n, uint64_t *mask, uint32_t C[2][2]);

#endif
