#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <mpi.h>
#include "speck.h"
#include "utils.h"
#include "distributed_dict.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    uint64_t N;
    uint64_t M;
    uint32_t CC[2][2];
    process_command_line_options(argc, argv, &N, &M, CC);

    int world_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    
    if(world_rank==0) {
        printf("Running with n=%" PRIu64 ", C0=(%08x,%08x), C1=(%08x,%08x)\n",
            N, CC[0][0], CC[0][1], CC[1][0], CC[1][1]);
    }

    distributed_dict_setup(N, M, CC);

    double start = wtime();
    distributed_dict_build();
    double mid = wtime();

    if(world_rank==0)
        printf("Build dict time: %.2fs\n", mid - start);

    uint64_t k1[16], k2[16];
    int nkey = distributed_dict_search(16, k1, k2);

    if(world_rank==0) {
        double end = wtime();
        printf("Search time: %.2fs\n", end - mid);
    }

    MPI_Finalize();
    return 0;
}
