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
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Declare variables for command-line options
    uint64_t N;
    uint64_t M;
    uint32_t CC[2][2];
    process_command_line_options(argc, argv, &N, &M, CC);

    // Get the rank of the current process in the MPI world
    int world_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    
    // If this is the master process (rank 0), print the initial configuration
    if(world_rank==0) {
        printf("Running with n=%" PRIu64 ", C0=(%08x,%08x), C1=(%08x,%08x)\n",
            N, CC[0][0], CC[0][1], CC[1][0], CC[1][1]);
    }

    // Set up the distributed dictionary with the provided parameters
    distributed_dict_setup(N, M, CC);

    // Start timing for dictionary build
    double start = wtime_ms();

    // Build the distributed dictionary
    distributed_dict_build();

    // Record the time after building the dictionary
    double mid = wtime_ms();

    // If this is the master process, print the time taken to build the dictionary
    if(world_rank==0)
        printf("Build dict time: %.2fms\n", mid - start);

    // Declare key arrays for the search operation
    uint64_t k1[16], k2[16];
    
    // Perform a search in the distributed dictionary
    distributed_dict_search(16, k1, k2);

    // If this is the master process, record and print the search time
    if(world_rank==0) {
        double end = wtime_ms();
        printf("Search time: %.2fms\n", end - mid);
    }
    
    // Output statistics
    distributed_dict_print_stats();
    
    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
