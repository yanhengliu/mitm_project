#include "distributed_dict.h"
#include "dictionary.h"
#include "speck.h"
#include "utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

static int world_size, world_rank;
static uint64_t N;

typedef struct {
    uint64_t key; // y
    uint64_t z;   // z
} query_t;

/* Sharding rule: fragment = murmur64(key) % world_size based on key */
static inline int shard_of_key(uint64_t key) {
    return (int)(murmur64(key) % world_size);
}

void distributed_dict_setup(uint64_t nbits, uint64_t m, uint32_t CC[2][2])
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    n = nbits;
    mask = m;
    C[0][0] = CC[0][0]; C[0][1] = CC[0][1];
    C[1][0] = CC[1][0]; C[1][1] = CC[1][1];
    N = (1ULL << n);

    // Set up space for the dictionary, with an estimated total size of 1.125 * N
    // Approximate after averaging (1.125 * N / world_size)
    uint64_t global_dict_size = (uint64_t)(1.125 * N);
    uint64_t per_proc = global_dict_size / world_size + 100; 
    dict_setup(per_proc);
}

/**
 * Distribute f(x) results using MPI_Alltoallv:
 * Steps:
 * 1. locally compute f(x) for all x's in range and store (key=z,value=x) into an array
 * 2. according to shard_of_key to determine the target process to send, count the amount of data per process
 * 3. Use Alltoallv to send data to the corresponding process.
 * 4. receive the data in the local dict insert
 */
void distributed_dict_build()
{
    uint64_t start_x = world_rank * (N / world_size);
    uint64_t end_x = (world_rank == world_size-1) ? N : (world_rank+1)*(N/world_size);
    uint64_t local_count = end_x - start_x;

    uint64_t *local_pairs = malloc(sizeof(uint64_t)*2*local_count);
    if(!local_pairs) {
        fprintf(stderr, "Rank %d: Failed to allocate local_pairs\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    uint64_t idx=0;
    for(uint64_t x = start_x; x<end_x; x++){
        uint64_t z = f(x);
        local_pairs[idx++] = z; // key
        local_pairs[idx++] = x; // value
    }

    // Counting the number of copies each target process needs to send
    int *send_counts = calloc(world_size, sizeof(int));
    int *recv_counts = calloc(world_size, sizeof(int));
    if(!send_counts || !recv_counts) {
        fprintf(stderr, "Rank %d: Failed to allocate send/recv_counts\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    // Count the number of pairs for each process (2 uint64_t per pair)
    for (uint64_t i=0; i<local_count; i++) {
        uint64_t key = local_pairs[2*i];
        int dest = shard_of_key(key);
        send_counts[dest]+=2; 
    }

    // Calculate offsets for Alltoallv
    int *sdispls = calloc(world_size, sizeof(int));
    int *rdispls = calloc(world_size, sizeof(int));
    if(!sdispls || !rdispls) {
        fprintf(stderr, "Rank %d: Failed to allocate sdispls/rdispls\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Exchange size information
    MPI_Alltoall(send_counts,1,MPI_INT,recv_counts,1,MPI_INT,MPI_COMM_WORLD);

    // Calculate send and receive displacements
    {
        int soff=0, roff=0;
        for(int i=0; i<world_size; i++){
            sdispls[i]=soff; soff+=send_counts[i];
            rdispls[i]=roff; roff+=recv_counts[i];
        }
    }

    // Allocate send and receive buffers
    uint64_t *sendbuf = malloc(sizeof(uint64_t)*(sdispls[world_size-1]+send_counts[world_size-1]));
    uint64_t *recvbuf = malloc(sizeof(uint64_t)*(rdispls[world_size-1]+recv_counts[world_size-1]));
    if(!sendbuf || !recvbuf) {
        fprintf(stderr, "Rank %d: Failed to allocate sendbuf/recvbuf\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Fill sendbuf with local_pairs sorted by destination process
    int *current_pos = calloc(world_size,sizeof(int));
    if(!current_pos) {
        fprintf(stderr, "Rank %d: Failed to allocate current_pos\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    for(uint64_t i=0; i<local_count; i++){
        uint64_t key=local_pairs[2*i];
        uint64_t val=local_pairs[2*i+1];
        int dest = shard_of_key(key);
        int pos = sdispls[dest]+current_pos[dest];
        sendbuf[pos]=key;
        sendbuf[pos+1]=val;
        current_pos[dest]+=2;
    }

    free(current_pos);
    free(local_pairs);

    // Perform Alltoallv to distribute the data
    MPI_Alltoallv(sendbuf, send_counts, sdispls, MPI_UINT64_T,
                  recvbuf, recv_counts, rdispls, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    // Insert received pairs into the local dictionary
    int total_recv = rdispls[world_size-1]+recv_counts[world_size-1];
    for(int i=0; i<total_recv; i+=2){
        uint64_t key = recvbuf[i];
        uint64_t val = recvbuf[i+1];
        dict_insert(key,val);
    }

    free(send_counts); free(recv_counts);
    free(sdispls); free(rdispls);
    free(sendbuf); free(recvbuf);
}

/**
 * Modify query logic to not directly assign query_t to uint64_t array.
 * Use pure uint64_t arrays in send and receive sessions and extract key, z values when needed.
 */

/**
 * Search Phase:
 * Each process is responsible for a range of z, calculates y=g(z), 
 * and sends the query to the corresponding process based on the shard.
 * After querying, the recipient process returns a list of matching x values,
 * which are validated locally using is_good_pair and the results are collected.
 *
 * Use point-to-point communication (MPI_Send/MPI_Recv).
 *
 * Process flow:
 * 1. Each process computes the local [g(z)] set and organizes requests into per-rank query lists based on sharding.
 * 2. Each process sends requests (asynchronously or synchronously) to the target rank. The recipient performs 
 *    `dict_probe` on the queries and sends back responses.
 * 3. Once all responses are received, validate is_good_pair locally and aggregate all solutions to rank 0.
 */
int distributed_dict_search(int maxres, uint64_t k1[], uint64_t k2[])
{
    uint64_t start_z = world_rank * (N / world_size);
    uint64_t end_z = (world_rank == world_size-1) ? N : (world_rank+1)*(N/world_size);
    uint64_t local_count = end_z - start_z;

    // Temporary storage for all queries (no sorting required)
    query_t *queries = malloc(sizeof(query_t)*local_count);
    if(!queries) {
        fprintf(stderr,"Rank %d: queries alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Count how many queries need to be sent to each process
    int *req_counts = calloc(world_size,sizeof(int));
    if(!req_counts) {
        fprintf(stderr,"Rank %d: req_counts alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Compute y = g(z) and determine target rank for each query
    for(uint64_t i=0; i<local_count; i++){
        uint64_t z = start_z + i;
        uint64_t y = g(z);
        int dest = shard_of_key(y);
        queries[i].key = y;
        queries[i].z = z;
        // Store the destination rank in the upper 16 bits of z, lower 48 bits store the actual z value
        req_counts[dest]++;
    }

    int total_req=0;
    for(int i=0;i<world_size;i++){
        total_req+=req_counts[i];
    }

    // Reorganize queries into a per-destination array and prepare for MPI_Send
    int *offsets = calloc(world_size,sizeof(int));
    if(!offsets) {
        fprintf(stderr,"Rank %d: offsets alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Compute prefix sums for offsets
    for(int i=1;i<world_size;i++){
        offsets[i] = offsets[i-1]+req_counts[i-1];
    }

    // Organise queries into send buffers (uint64_t arrays)
    // 2 uint64_t to send (y,z) per request
    uint64_t *req_buffer = malloc(sizeof(uint64_t)*2*total_req);
    if(!req_buffer) {
        fprintf(stderr,"Rank %d: req_buffer alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int *cur_pos = calloc(world_size,sizeof(int));
    if(!cur_pos) {
        fprintf(stderr,"Rank %d: cur_pos alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    for(uint64_t i=0;i<local_count;i++){
        int dest = shard_of_key(queries[i].key);
        int pos = (offsets[dest]+cur_pos[dest])*2; 
        req_buffer[pos] = queries[i].key;
        req_buffer[pos+1] = queries[i].z;
        cur_pos[dest]++;
    }

    free(cur_pos);

    // Exchange query sizes with other processes to determine how many queries to receive
    int *recv_counts = calloc(world_size,sizeof(int));
    MPI_Alltoall(req_counts,1,MPI_INT,recv_counts,1,MPI_INT,MPI_COMM_WORLD);

    int total_recv=0;
    for(int i=0;i<world_size;i++)
        total_recv+=recv_counts[i];

    uint64_t *recv_buffer = malloc(sizeof(uint64_t)*2*total_recv);
    if(!recv_buffer){
        fprintf(stderr,"Rank %d: recv_buffer alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    // Perform P2P communication to send and receive queries
    // Allocate space for displacements
    int *rdispls = calloc(world_size,sizeof(int));
    if(!rdispls) {
        fprintf(stderr,"Rank %d: rdispls alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    for(int i=1;i<world_size;i++){
        rdispls[i] = rdispls[i-1]+recv_counts[i-1];
    }

    int *sdispls = calloc(world_size,sizeof(int));
    if(!sdispls){
        fprintf(stderr,"Rank %d: sdispls alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    for(int i=1;i<world_size;i++){
        sdispls[i] = sdispls[i-1]+req_counts[i-1];
    }

    // P2P communication requests
    // Send and receive symmetrically using non-blocking communication
    MPI_Request *reqs = malloc(sizeof(MPI_Request)*2*world_size);
    int reqi=0;

    // Post non-blocking receives
    for(int i=0;i<world_size;i++){
        if(recv_counts[i]>0){
            MPI_Irecv(&recv_buffer[rdispls[i]*2], recv_counts[i]*2, MPI_UINT64_T,
                      i, 100, MPI_COMM_WORLD, &reqs[reqi++]);
        }
    }

    // Send query data
    for(int i=0;i<world_size;i++){
        if(req_counts[i]>0){
            MPI_Isend(&req_buffer[sdispls[i]*2], req_counts[i]*2, MPI_UINT64_T,
                      i, 100, MPI_COMM_WORLD, &reqs[reqi++]);
        }
    }

    MPI_Waitall(reqi, reqs, MPI_STATUSES_IGNORE);
    free(reqs);

    // The recv_buffer now contains (y, z) pairs that this process needs to query.
    // For each query, perform `dict_probe`, and send the results back to the original process.
    // The original process corresponds to the source process `i` during the MPI receive.
    // For responses, perform another round of point-to-point communication.

    // Group queries by the rank of their origin and prepare for response.
    int *resp_counts = calloc(world_size,sizeof(int));
    if(!resp_counts){
        fprintf(stderr,"Rank %d: resp_counts alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    // Allocate space for holding potential x values from `dict_probe`.
    uint64_t *probe_vals = malloc(sizeof(uint64_t)*256);
    if(!probe_vals) {
        fprintf(stderr,"Rank %d: probe_vals alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Allocate displacements for sending responses
    int *back_rdispls = calloc(world_size,sizeof(int));
    int *back_sdispls = calloc(world_size,sizeof(int));
    if(!back_rdispls||!back_sdispls){
        fprintf(stderr,"Rank %d: back_rdispls/back_sdispls alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Count the number of response entries for each process
    for(int r=0; r<world_size; r++){
        // `recv_counts[r]` queries originated from rank `r`
        // Offset of these queries in `recv_buffer` is `rdispls[r]`
        for(int q=0; q<recv_counts[r]; q++){
            uint64_t y = recv_buffer[(rdispls[r]+q)*2]; 
            uint64_t z = recv_buffer[(rdispls[r]+q)*2+1];
            int nx = dict_probe(y, 256, probe_vals);
            if(nx<0) nx=256;
            resp_counts[r] += nx*2; // Each match (x, z) contributes 2 uint64_t
        }
    }

    // Compute response displacements for sending back results
    for(int i=1;i<world_size;i++){
        back_sdispls[i]=back_sdispls[i-1]+resp_counts[i-1];
    }

    // Allocate buffer for sending responses
    uint64_t *resp_buffer = malloc(sizeof(uint64_t)*(back_sdispls[world_size-1]+resp_counts[world_size-1]));
    if(!resp_buffer){
        fprintf(stderr,"Rank %d: resp_buffer alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int *resp_cur = calloc(world_size,sizeof(int));
    if(!resp_cur){
        fprintf(stderr,"Rank %d: resp_cur alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Populate the response buffer
    for(int r=0; r<world_size; r++){
        for(int q=0; q<recv_counts[r]; q++){
            uint64_t y = recv_buffer[(rdispls[r]+q)*2]; 
            uint64_t z = recv_buffer[(rdispls[r]+q)*2+1];
            int nx = dict_probe(y, 256, probe_vals);
            if(nx<0) nx=256;
            for(int i=0;i<nx;i++){
                int pos = back_sdispls[r]+resp_cur[r];
                resp_buffer[pos]=probe_vals[i];
                resp_buffer[pos+1]=z;
                resp_cur[r]+=2;
            }
        }
    }

    // Clean up temporary buffers used for query processing
    free(recv_buffer);
    free(resp_cur);
    free(probe_vals);

    // Exchange the response sizes with all processes
    int *back_recv_counts = calloc(world_size,sizeof(int));
    MPI_Alltoall(resp_counts,1,MPI_INT,back_recv_counts,1,MPI_INT,MPI_COMM_WORLD);

    // Calculate total size of responses to receive
    int back_total=0;
    for(int i=0;i<world_size;i++)
        back_total+=back_recv_counts[i];

    uint64_t *back_buffer= malloc(sizeof(uint64_t)*back_total);
    if(!back_buffer) {
        fprintf(stderr,"Rank %d: back_buffer alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Calculate receive displacements for responses
    for(int i=1;i<world_size;i++){
        back_rdispls[i] = back_rdispls[i-1]+back_recv_counts[i-1];
    }

    int *all_counts=calloc(world_size,sizeof(int));
    int *all_displs=calloc(world_size,sizeof(int));

    if(!all_counts||!all_displs){
        fprintf(stderr,"Rank %d: all_counts/all_displs alloc failed\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Send responses back to their respective source processes
    // Do P2P again.
    MPI_Request *reqs2 = malloc(sizeof(MPI_Request)*2*world_size);
    int reqj=0;

    // Post non-blocking receives for responses
    for(int i=0;i<world_size;i++){
        if(back_recv_counts[i]>0){
            MPI_Irecv(&back_buffer[back_rdispls[i]], back_recv_counts[i], MPI_UINT64_T,
                      i, 200, MPI_COMM_WORLD, &reqs2[reqj++]);
        }
    }

    // Send responses
    for(int i=0;i<world_size;i++){
        if(resp_counts[i]>0){
            MPI_Isend(&resp_buffer[back_sdispls[i]], resp_counts[i], MPI_UINT64_T,
                      i, 200, MPI_COMM_WORLD, &reqs2[reqj++]);
        }
    }

    MPI_Waitall(reqj, reqs2, MPI_STATUSES_IGNORE);

    free(reqs2);
    free(resp_counts);
    free(back_recv_counts);
    free(back_rdispls);
    free(back_sdispls);
    free(resp_buffer);

    // The back_buffer now contains (x, z) pairs. Validate is_good_pair locally.
    int found=0;
    for(int i=0;i<back_total;i+=2){
        uint64_t X = back_buffer[i];
        uint64_t Z = back_buffer[i+1];
        if(is_good_pair(X,Z)){
            if(found<maxres) {
                k1[found]=X;
                k2[found]=Z;
            }
            found++;
        }
    }
    free(back_buffer);

    // Aggregate the total number of solutions found across all processes
    int total_found=0;
    MPI_Reduce(&found,&total_found,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    // Collect all solutions for rank 0
    int send_sol = (found>maxres?maxres:found)*2;
    MPI_Gather(&send_sol,1,MPI_INT,all_counts,1,MPI_INT,0,MPI_COMM_WORLD);

    uint64_t *my_sol_buffer = malloc(sizeof(uint64_t)*send_sol);
    for(int i=0;i<send_sol/2;i++){
        my_sol_buffer[2*i]=k1[i];
        my_sol_buffer[2*i+1]=k2[i];
    }

    if(world_rank==0){
        int total_sol=0;
        for(int i=0;i<world_size;i++){
            total_sol+=all_counts[i];
        }
        for(int i=1;i<world_size;i++){
            all_displs[i]=all_displs[i-1]+all_counts[i-1];
        }

        uint64_t *all_solutions=malloc(sizeof(uint64_t)*total_sol);
        MPI_Gatherv(my_sol_buffer, send_sol, MPI_UINT64_T,
                    all_solutions, all_counts, all_displs, MPI_UINT64_T,
                    0, MPI_COMM_WORLD);

        if(total_found>0) {
            printf("Total solutions found: %d\n", total_found);
            for(int i=0;i<total_sol;i+=2) {
                uint64_t X=all_solutions[i];
                uint64_t Z=all_solutions[i+1];
                assert(f(X)==g(Z));
                assert(is_good_pair(X,Z));
                printf("Solution: (%" PRIx64 ", %" PRIx64 ")\n", X, Z);
            }
        } else {
            // 没有解
            MPI_Gatherv(my_sol_buffer, send_sol, MPI_UINT64_T,
                        all_solutions, all_counts, all_displs, MPI_UINT64_T,
                        0, MPI_COMM_WORLD);
            printf("No solution found.\n");
        }
        free(all_solutions);
    } else {
        MPI_Gatherv(my_sol_buffer, send_sol, MPI_UINT64_T,
                    NULL, NULL, NULL, MPI_UINT64_T,
                    0, MPI_COMM_WORLD);
    }

    free(my_sol_buffer);
    free(all_counts);
    free(all_displs);
    free(req_buffer);
    free(queries);
    free(req_counts);
    free(recv_counts);
    free(rdispls);
    free(sdispls);
    free(offsets);

    return total_found;
}
