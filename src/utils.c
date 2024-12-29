#include "utils.h"
#include <sys/time.h>
#include <stdlib.h>
#include <getopt.h>
#include <err.h>

double wtime()
{
    struct timeval ts;
    gettimeofday(&ts, NULL);
    return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

void human_format(uint64_t n, char *target)
{
    if (n < 1000) {
        sprintf(target, "%" PRIu64, n);
        return;
    }
    if (n < 1000000) {
        sprintf(target, "%.1fK", n / 1e3);
        return;
    }
    if (n < 1000000000) {
        sprintf(target, "%.1fM", n / 1e6);
        return;
    }
    if (n < 1000000000000ULL) {
        sprintf(target, "%.1fG", n / 1e9);
        return;
    }
    sprintf(target, "%.1fT", n / 1e12);
}

/************************** Hash for sharding rule ****************************/
uint64_t murmur64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/************************** command-line options ****************************/
void usage(char **argv)
{
    printf("%s [OPTIONS]\n\n", argv[0]);
    printf("Options:\n");
    printf("--n N                       block size [default 24]\n");
    printf("--C0 N                      1st ciphertext (in hex)\n");
    printf("--C1 N                      2nd ciphertext (in hex)\n");
    printf("\n");
    printf("All arguments are required\n");
    exit(0);
}

void process_command_line_options(int argc, char ** argv, uint64_t *n, uint64_t *mask, uint32_t C[2][2])
{
    struct option longopts[4] = {
        {"n", required_argument, NULL, 'n'},
        {"C0", required_argument, NULL, '0'},
        {"C1", required_argument, NULL, '1'},
        {NULL, 0, NULL, 0}
    };
    char ch;
    int set = 0;
    uint64_t N=0;
    uint32_t LocalC[2][2];

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
        switch (ch) {
        case 'n':
            N = (uint64_t)atoi(optarg);
            break;
        case '0':
        {
            set |= 1;
            uint64_t c0 = strtoull(optarg, NULL, 16);
            LocalC[0][0] = (uint32_t)(c0 & 0xffffffff);
            LocalC[0][1] = (uint32_t)(c0 >> 32);
            break;
        }
        case '1':
        {
            set |= 2;
            uint64_t c1 = strtoull(optarg, NULL, 16);
            LocalC[1][0] = (uint32_t)(c1 & 0xffffffff);
            LocalC[1][1] = (uint32_t)(c1 >> 32);
            break;
        }
        default:
            errx(1, "Unknown option\n");
        }
    }

    if (N == 0 || set != 3) {
        usage(argv);
        exit(1);
    }

    *n = N;
    *mask = ((uint64_t)1 << N) - 1;
    C[0][0] = LocalC[0][0]; C[0][1] = LocalC[0][1];
    C[1][0] = LocalC[1][0]; C[1][1] = LocalC[1][1];
}
