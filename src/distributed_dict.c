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

/* Basic structure for local statistics */
typedef struct {
    uint64_t local_insert_count;    // number of local inserts
    uint64_t local_received_count;  // number of received (key,value)
    uint64_t local_query_sent;      // total queries sent
    uint64_t local_query_received;  // total queries received
    uint64_t local_probe_count;     // number of dict_probe calls
    uint64_t dict_insert_calls;     // actual dictionary-layer insert calls
    uint64_t dict_probe_calls;      // actual dictionary-layer probe calls
} DistStats;

/* Declare static global stats variable, for each rank. */
static DistStats stats;

/* Sharding rule: fragment = murmur64(key) % world_size based on key */
static inline int shard_of_key(uint64_t key) {
    return (int)(murmur64(key) % world_size);
}

/* ========================== */
/*   distributed_dict_setup   */
/* ========================== */

/**
 * Sets up the dictionary structure. 
 */
void distributed_dict_setup(uint64_t nbits, uint64_t m, uint32_t CC[2][2])
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Reset stats
    stats.local_insert_count   = 0;
    stats.local_received_count = 0;
    stats.local_query_sent     = 0;
    stats.local_query_received = 0;
    stats.local_probe_count    = 0;
    stats.dict_insert_calls    = 0;
    stats.dict_probe_calls     = 0;

    // Assign the global n, mask, and C array
    n = nbits;
    mask = m;
    C[0][0] = CC[0][0];  C[0][1] = CC[0][1];
    C[1][0] = CC[1][0];  C[1][1] = CC[1][1];

    // N = 2^n
    N = (1ULL << n);

    // (Optional) warn if n is extremely large
    if (world_rank == 0 && n > 45) {
        fprintf(stderr,
                "[Warning] n=%" PRIu64 " is very large; ensure chunk-based approach and enough memory.\n",
                n);
    }

    // Estimate dictionary size and allocate local hashtable
    long double factor = 1.125;
    uint64_t global_dict_size = (uint64_t)(factor * (long double)N);
    uint64_t per_proc = global_dict_size / world_size + 100ULL;
    dict_setup(per_proc);
}

/* =========================== */
/*   distributed_dict_build    */
/* =========================== */

/**
 * Replaces the previous Alltoallv approach with a point-to-point (MPI_Send/MPI_Recv)
 * approach, still chunk-based to avoid large memory usage in one go.
 *
 * STEPS:
 * 1) Build local (f(x), x) pairs for x in [start_x, end_x).
 * 2) Distribute them to each rank using chunk-based sends.
 * 3) Receive from each rank in a symmetrical manner.
 * 4) Insert into local dictionary.
 */
void distributed_dict_build()
{
    uint64_t start_x = world_rank*(N/world_size);
    uint64_t end_x   = (world_rank==(world_size-1))? N : (world_rank+1)*(N/world_size);
    uint64_t local_count=end_x - start_x;
    if(local_count==0) return;

    // build local (f(x), x)
    uint64_t *pairs = malloc(sizeof(uint64_t)*2ULL*local_count);
    if(!pairs){
        fprintf(stderr,"[Rank %d] cannot allocate pairs for build count=%"PRIu64"\n",
                world_rank,local_count);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    uint64_t idx=0;
    for(uint64_t x=start_x; x<end_x; x++){
        uint64_t z=f(x);
        pairs[idx++]=z;
        pairs[idx++]=x;
    }

    // classify
    int *counts = calloc(world_size,sizeof(int));
    if(!counts){
        fprintf(stderr,"[Rank %d] cannot allocate counts\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    for(uint64_t i=0;i<local_count;i++){
        uint64_t key = pairs[2ULL*i];
        int dest = shard_of_key(key);
        counts[dest]++;
    }

    // allocate outbuf[r]
    uint64_t **outbuf = calloc(world_size,sizeof(uint64_t*));
    if(!outbuf){
        fprintf(stderr,"[Rank %d] cannot allocate outbuf\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    for(int r=0; r<world_size; r++){
        if(counts[r]>0){
            outbuf[r]=malloc(sizeof(uint64_t)*2ULL*counts[r]);
            if(!outbuf[r]){
                fprintf(stderr,"[Rank %d] cannot allocate outbuf[r=%d]\n",world_rank,r);
                MPI_Abort(MPI_COMM_WORLD,1);
            }
        }
    }

    // fill outbuf
    int *pos = calloc(world_size,sizeof(int));
    for(uint64_t i=0; i<local_count; i++){
        uint64_t key = pairs[2ULL*i];
        uint64_t val = pairs[2ULL*i+1];
        int dest = shard_of_key(key);
        int offset = pos[dest]*2;
        outbuf[dest][offset]=key;
        outbuf[dest][offset+1]=val;
        pos[dest]++;
    }
    free(pos);
    free(pairs);

    // exchange counts => incounts
    int *incounts=calloc(world_size,sizeof(int));
    if(!incounts){
        fprintf(stderr,"[Rank %d] cannot allocate incounts\n",world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    for(int r=0;r<world_size;r++){
        if(r==world_rank) continue;
        if(r<world_rank){
            MPI_Recv(&incounts[r],1,MPI_INT,r,200,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&counts[r],1,MPI_INT,r,201,MPI_COMM_WORLD);
        } else {
            MPI_Send(&counts[r],1,MPI_INT,r,200,MPI_COMM_WORLD);
            MPI_Recv(&incounts[r],1,MPI_INT,r,201,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }

    // chunk size (#pairs)
    const uint64_t CHUNK_SIZE=(1ULL<<10); 

    // do actual data exchange
    for(int r=0;r<world_size;r++){
        if(r==world_rank) continue;

        int sendsz=counts[r];
        int recvsz=incounts[r];

        if(r<world_rank){
            // 1) recv from r
            {
                uint64_t total_words=(uint64_t)recvsz*2ULL;
                uint64_t done=0;
                while(done<total_words){
                    uint64_t left=total_words-done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left:(CHUNK_SIZE*2ULL);
                    uint64_t *buf=malloc(sizeof(uint64_t)*chunk_words);
                    if(!buf){
                        fprintf(stderr,"[Rank %d] cannot alloc recvtemp\n",world_rank);
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                    MPI_Recv(buf, chunk_words, MPI_UINT64_T, r, 300, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    // insert
                    for(uint64_t i=0;i<chunk_words;i+=2){
                        uint64_t key=buf[i];
                        uint64_t val=buf[i+1];
                        dict_insert(key,val);
                        stats.local_insert_count++;
                        stats.local_received_count++;
                    }
                    free(buf);
                    done+=chunk_words;
                }
            }
            // 2) send to r
            {
                uint64_t total_words=(uint64_t)sendsz*2ULL;
                uint64_t done=0;
                uint64_t *ob=outbuf[r];
                while(done<total_words){
                    uint64_t left=total_words-done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left:(CHUNK_SIZE*2ULL);
                    MPI_Send(&ob[done], chunk_words, MPI_UINT64_T, r, 300, MPI_COMM_WORLD);
                    done+=chunk_words;
                }
            }
        } else {
            // r>me => send first, then recv
            {
                uint64_t total_words=(uint64_t)sendsz*2ULL;
                uint64_t done=0;
                uint64_t *ob=outbuf[r];
                while(done<total_words){
                    uint64_t left=total_words-done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left:(CHUNK_SIZE*2ULL);
                    MPI_Send(&ob[done],chunk_words,MPI_UINT64_T,r,300,MPI_COMM_WORLD);
                    done+=chunk_words;
                }
            }
            {
                uint64_t total_words=(uint64_t)recvsz*2ULL;
                uint64_t done=0;
                while(done<total_words){
                    uint64_t left=total_words-done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left:(CHUNK_SIZE*2ULL);
                    uint64_t *buf=malloc(sizeof(uint64_t)*chunk_words);
                    if(!buf){
                        fprintf(stderr,"[Rank %d] cannot alloc recvtemp2\n",world_rank);
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                    MPI_Recv(buf,chunk_words,MPI_UINT64_T,r,300,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    // insert
                    for(uint64_t i=0;i<chunk_words;i+=2){
                        uint64_t key=buf[i];
                        uint64_t val=buf[i+1];
                        dict_insert(key,val);
                        stats.local_insert_count++;
                        stats.local_received_count++;
                    }
                    free(buf);
                    done+=chunk_words;
                }
            }
        }
    }

    // free memory
    for(int r=0;r<world_size;r++){
        if(counts[r]>0 && outbuf[r]){
            free(outbuf[r]);
        }
    }
    free(outbuf);
    free(counts);
    free(incounts);
}

/* =========================== */
/*   distributed_dict_search    */
/* =========================== */
/* 
 * A chunk-based p2p approach in search:
 * 1) build local (y,z), classify => outbuf[r]
 * 2) point-to-point chunk exchange with each rank => receive (y,z) from them => dict_probe => produce (x,z) => store in resp_outbuf[r]
 * 3) second round: chunk-based exchange of (x,z) => original rank does is_good_pair => store in k1[],k2[]
 * 4) gather solutions
 */
int distributed_dict_search(int maxres, uint64_t k1[], uint64_t k2[])
{
    // Step 0: determine local z range
    uint64_t start_z = world_rank * (N / world_size);
    uint64_t end_z   = (world_rank == world_size - 1)
                     ? N
                     : (world_rank + 1) * (N / world_size);
    uint64_t local_count = end_z - start_z;

    //  Build local queries (g(z), z) => classify them per rank
    query_t *queries = NULL;
    if(local_count > 0) {
        queries = (query_t*) malloc(sizeof(query_t)*local_count);
        if(!queries){
            fprintf(stderr,"[Rank %d] Fail: cannot allocate queries array (count=%"PRIu64")\n",
                    world_rank, local_count);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    //  outcounts[r] => how many queries we send to rank r
    int *outcounts = (int*) calloc(world_size, sizeof(int));
    if(!outcounts){
        fprintf(stderr,"[Rank %d] cannot allocate outcounts in search\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    //  fill the queries array and classify
    for(uint64_t i=0; i<local_count; i++){
        uint64_t z = start_z + i;
        uint64_t y = g(z);
        int dest = shard_of_key(y);
        queries[i].key = y;  //  (y,z) pair
        queries[i].z   = z;
        outcounts[dest]++;
    }

    //  outbuf[r] => store (y,z) in 2 64-bit words per query
    uint64_t **outbuf = (uint64_t**) calloc(world_size, sizeof(uint64_t*));
    if(!outbuf){
        fprintf(stderr,"[Rank %d] cannot allocate outbuf pointer array\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    for(int r=0; r<world_size; r++){
        if(outcounts[r] > 0){
            outbuf[r] = (uint64_t*) malloc(sizeof(uint64_t)*2ULL*outcounts[r]);
            if(!outbuf[r]){
                fprintf(stderr,"[Rank %d] cannot allocate outbuf[r=%d]\n", world_rank,r);
                MPI_Abort(MPI_COMM_WORLD,1);
            }
        }
    }

    //  fill outbuf[r]
    int *pos = (int*) calloc(world_size,sizeof(int));
    for(uint64_t i=0; i<local_count; i++){
        uint64_t y = queries[i].key;
        uint64_t z = queries[i].z;
        int dest = shard_of_key(y);
        int offset = pos[dest]*2;
        outbuf[dest][offset]   = y;
        outbuf[dest][offset+1] = z;
        pos[dest]++;
    }
    free(pos);
    if(queries) {
        free(queries);
    }

    //  incounts[r] => how many queries we'll receive from rank r
    int *incounts = (int*) calloc(world_size,sizeof(int));
    if(!incounts){
        fprintf(stderr,"[Rank %d] cannot allocate incounts in search\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    //  exchange outcounts <-> incounts with each rank, p2p (to avoid all ranks do send at once)
    for(int r=0; r<world_size; r++){
        if(r == world_rank) continue;
        if(r < world_rank){
            MPI_Recv(&incounts[r],1,MPI_INT,r,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&outcounts[r],1,MPI_INT,r,101,MPI_COMM_WORLD);
        } else {
            MPI_Send(&outcounts[r],1,MPI_INT,r,100,MPI_COMM_WORLD);
            MPI_Recv(&incounts[r],1,MPI_INT,r,101,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }

    /* Also define a chunk size (#queries) for queries exchange */
    const uint64_t CHUNK_SIZE = (1ULL<<10); // e.g. 16384 queries => 32768 64-bit words

    /*
     * Step 1: chunk-based p2p => send queries => recv queries
     * Step 2: for each query we receive, do dict_probe => build response outbuf => chunk-based send back
     * Step 3: also recv responses => do is_good_pair => store solutions
     */

    /* For storing the responses we generate for each rank that queries us */
    // Each query might produce multiple (x,z) or none.
    // We'll do it similarly to outcounts but call them resp_outcounts.
    int *resp_outcounts = (int*) calloc(world_size, sizeof(int));
    if(!resp_outcounts){
        fprintf(stderr,"[Rank %d] cannot allocate resp_outcounts\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    /* We'll store the actual (x,z) pairs in resp_outbuf[r]. 
       Then we do a second pass to send them back to r.
    */
    uint64_t **resp_outbuf = (uint64_t**) calloc(world_size,sizeof(uint64_t*));
    if(!resp_outbuf){
        fprintf(stderr,"[Rank %d] cannot allocate resp_outbuf array\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    /* Initialize so they can be malloc'ed dynamically after we know the total. */

    /* Step 1: p2p exchange of queries */
    for(int r=0; r<world_size; r++){
        if(r == world_rank) continue;

        int sendsz = outcounts[r]; // # queries to rank r
        int recvsz = incounts[r];  // # queries from rank r

        if(r < world_rank){
            /* first recv queries from r, then send queries to r */

            /* recv queries from r in chunk steps */
            {
                uint64_t total_words = (uint64_t)recvsz*2ULL; // each query => 2 words
                uint64_t done=0;
                while(done < total_words){
                    uint64_t left = total_words - done;
                    uint64_t chunk_words = (left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    uint64_t *buf = (uint64_t*) malloc(sizeof(uint64_t)*chunk_words);
                    if(!buf){
                        fprintf(stderr,"[Rank %d] cannot allocate recv-chunk\n",world_rank);
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                    MPI_Recv(buf, chunk_words, MPI_UINT64_T, r, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    /* Process these queries => each chunk_words/2 queries => (y,z) */
                    for(uint64_t i=0; i<chunk_words; i+=2){
                        uint64_t y=buf[i];
                        uint64_t z=buf[i+1];
                        // local_query_received
                        stats.local_query_received++;
                        // do dict_probe => produce (x,z)
                        // let's do a typical approach: up to 256 x in probe
                        // Or we can do repeated calls. We'll use a small buffer.
                        uint64_t candidates[256];
                        stats.local_probe_count++;
                        int nx = dict_probe(y, 256, candidates);
                        if(nx<0) nx=256; // truncated

                        // for each candidate x => store (x,z) in resp_outbuf[r]
                        // first expand resp_outbuf[r] if needed
                        // but for simplicity, we do "accumulate in memory" approach => we need an approach to do a dynamic array
                        if(nx>0){
                            // ensure we can store them
                            if(!resp_outbuf[r]){
                                resp_outbuf[r] = (uint64_t*) malloc(sizeof(uint64_t)*2ULL*nx);
                                resp_outcounts[r] = nx; 
                            } else {
                                // reallocate: oldsize + nx
                                int oldcnt = resp_outcounts[r];
                                resp_outcounts[r] = oldcnt + nx;
                                resp_outbuf[r] = (uint64_t*) realloc(resp_outbuf[r], sizeof(uint64_t)*2ULL*resp_outcounts[r]);
                            }
                            // fill newly appended portion
                            int startpos = (resp_outcounts[r] - nx)*2;
                            for(int k=0; k<nx; k++){
                                resp_outbuf[r][startpos + 2*k]   = candidates[k]; /* x */
                                resp_outbuf[r][startpos + 2*k+1] = z;
                            }
                        }
                    }
                    free(buf);
                    done+=chunk_words;
                }
            }

            /* send queries to r in chunk steps */
            {
                uint64_t total_words=(uint64_t)sendsz*2ULL;
                uint64_t done=0;
                uint64_t *ob = outbuf[r];
                while(done<total_words){
                    uint64_t left=total_words-done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    // local_query_sent
                    // in principle stats.local_query_sent += chunk_words/2 ...
                    MPI_Send(&ob[done], chunk_words, MPI_UINT64_T, r, 200, MPI_COMM_WORLD);
                    done+=chunk_words;
                }
                stats.local_query_sent += sendsz;
            }
        } else {
            /* r > me => send first, then recv */
            {
                /* send to r */
                uint64_t total_words=(uint64_t)sendsz*2ULL;
                uint64_t done=0;
                uint64_t *ob=outbuf[r];
                while(done<total_words){
                    uint64_t left=total_words-done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    MPI_Send(&ob[done], chunk_words, MPI_UINT64_T, r, 200, MPI_COMM_WORLD);
                    done+=chunk_words;
                }
                stats.local_query_sent += sendsz;
            }
            {
                /* recv from r */
                uint64_t total_words=(uint64_t)recvsz*2ULL;
                uint64_t done=0;
                while(done<total_words){
                    uint64_t left=total_words - done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    uint64_t *buf=(uint64_t*) malloc(sizeof(uint64_t)*chunk_words);
                    if(!buf){
                        fprintf(stderr,"[Rank %d] cannot allocate recv-chunk 2\n",world_rank);
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                    MPI_Recv(buf, chunk_words, MPI_UINT64_T, r, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for(uint64_t i=0; i<chunk_words; i+=2){
                        uint64_t y=buf[i];
                        uint64_t z=buf[i+1];
                        stats.local_query_received++;
                        stats.local_probe_count++;
                        uint64_t candidates[256];
                        int nx = dict_probe(y, 256, candidates);
                        if(nx<0) nx=256;
                        if(nx>0){
                            if(!resp_outbuf[r]){
                                resp_outbuf[r]= (uint64_t*) malloc(sizeof(uint64_t)*2ULL*nx);
                                resp_outcounts[r]=nx;
                            } else {
                                int oldcnt=resp_outcounts[r];
                                resp_outcounts[r]=oldcnt+nx;
                                resp_outbuf[r]= (uint64_t*) realloc(resp_outbuf[r], sizeof(uint64_t)*2ULL*resp_outcounts[r]);
                            }
                            int startpos=(resp_outcounts[r]-nx)*2;
                            for(int k=0; k<nx; k++){
                                resp_outbuf[r][startpos + 2*k]   = candidates[k];
                                resp_outbuf[r][startpos + 2*k+1] = z;
                            }
                        }
                    }
                    free(buf);
                    done+=chunk_words;
                }
            }
        }
    }

    /* now we have built resp_outbuf[r] with resp_outcounts[r] => # of (x,z) pairs to send back to rank r */
    // free queries outbuf
    for(int r=0; r<world_size; r++){
        if(outcounts[r]>0 && outbuf[r]){
            free(outbuf[r]);
        }
    }
    free(outbuf);
    free(outcounts);
    free(incounts);

    /* Step 2: point-to-point chunk-based exchange of (x,z) responses
       => we send resp_outbuf[r] to rank r, 
       => we receive from rank r => do is_good_pair => store in k1[],k2[] if pass
    */

    // incounts2[r] => how many (x,z) we will receive from rank r
    int *incounts2 = (int*) calloc(world_size,sizeof(int));
    if(!incounts2){
        fprintf(stderr,"[Rank %d] cannot allocate incounts2\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // exchange resp_outcounts <-> incounts2
    for(int r=0; r<world_size; r++){
        if(r==world_rank) continue;
        if(r<world_rank){
            MPI_Recv(&incounts2[r],1,MPI_INT,r,300,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&resp_outcounts[r],1,MPI_INT,r,301,MPI_COMM_WORLD);
        } else {
            MPI_Send(&resp_outcounts[r],1,MPI_INT,r,300,MPI_COMM_WORLD);
            MPI_Recv(&incounts2[r],1,MPI_INT,r,301,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }

    /* allocate a local buffer to store the (x,z) we receive from rank r in the response phase => then is_good_pair(x,z) */
    // we'll do chunk-based again => no need for large single array
    // but if you want to store them all at once, you can, but that might be huge. Let's do chunk-based verify.

    int found=0; // how many solutions we found in the entire process
    // we will store partial solutions in arrays k1[], k2[], up to maxres

    for(int r=0; r<world_size; r++){
        if(r==world_rank) continue;

        int sendsz = resp_outcounts[r]; // # of (x,z) we want to send
        int recvsz = incounts2[r];      // # of (x,z) to receive from rank r

        if(r<world_rank){
            // 1) recv from r
            {
                uint64_t total_words=(uint64_t)recvsz*2ULL;
                uint64_t done=0;
                while(done<total_words){
                    uint64_t left=total_words - done;
                    uint64_t chunk_words = (left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    uint64_t *buf=(uint64_t*) malloc(sizeof(uint64_t)*chunk_words);
                    if(!buf){
                        fprintf(stderr,"[Rank %d] cannot allocate recv chunk in response\n", world_rank);
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                    MPI_Recv(buf, chunk_words, MPI_UINT64_T, r, 400, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // chunk_words/2 => (x,z)
                    for(uint64_t i=0; i<chunk_words; i+=2){
                        uint64_t X=buf[i];
                        uint64_t Z=buf[i+1];
                        if(is_good_pair(X,Z)){
                            if(found<maxres){
                                k1[found]=X;
                                k2[found]=Z;
                            }
                            found++;
                        }
                    }
                    free(buf);
                    done += chunk_words;
                }
            }

            // 2) send to r
            {
                // send our response outbuf[r] in chunk steps
                uint64_t total_words=(uint64_t)sendsz*2ULL;
                uint64_t done=0;
                uint64_t *ob=resp_outbuf[r];
                while(done<total_words){
                    uint64_t left=total_words - done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    MPI_Send(&ob[done], chunk_words, MPI_UINT64_T, r, 400, MPI_COMM_WORLD);
                    done+=chunk_words;
                }
            }
        } else {
            // r>me => send first, then recv
            {
                // send outbuf
                uint64_t total_words=(uint64_t)sendsz*2ULL;
                uint64_t done=0;
                uint64_t *ob=resp_outbuf[r];
                while(done<total_words){
                    uint64_t left=total_words - done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    MPI_Send(&ob[done], chunk_words, MPI_UINT64_T, r, 400, MPI_COMM_WORLD);
                    done+=chunk_words;
                }
            }
            {
                // recv from r
                uint64_t total_words=(uint64_t)recvsz*2ULL;
                uint64_t done=0;
                while(done<total_words){
                    uint64_t left=total_words - done;
                    uint64_t chunk_words=(left<(CHUNK_SIZE*2ULL))? left : (CHUNK_SIZE*2ULL);
                    uint64_t *buf=(uint64_t*) malloc(sizeof(uint64_t)*chunk_words);
                    if(!buf){
                        fprintf(stderr,"[Rank %d] cannot alloc recvtemp in response 2\n",world_rank);
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                    MPI_Recv(buf,chunk_words,MPI_UINT64_T,r,400,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    for(uint64_t i=0;i<chunk_words;i+=2){
                        uint64_t X=buf[i];
                        uint64_t Z=buf[i+1];
                        if(is_good_pair(X,Z)){
                            if(found<maxres){
                                k1[found]=X;
                                k2[found]=Z;
                            }
                            found++;
                        }
                    }
                    free(buf);
                    done += chunk_words;
                }
            }
        }
    }

    // free response buffers
    for(int r=0; r<world_size; r++){
        if(resp_outcounts[r]>0 && resp_outbuf[r]){
            free(resp_outbuf[r]);
        }
    }
    free(resp_outbuf);
    free(resp_outcounts);
    free(incounts2);

    // gather found across all ranks
    int total_found=0;
    MPI_Reduce(&found,&total_found,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    // next, each rank has up to min(found, maxres) solutions in (k1[],k2[]). We gather them.
    // let's do a two-step gather approach:
    int send_sol=(found>maxres)?(maxres*2):(found*2);
    // we only gather up to 2*maxres local solutions in the worst case
    int *all_counts=(int*) calloc(world_size,sizeof(int));
    MPI_Gather(&send_sol,1,MPI_INT,all_counts,1,MPI_INT,0,MPI_COMM_WORLD);

    // build my_sol_buffer
    uint64_t *my_sol_buffer=(uint64_t*) malloc(sizeof(uint64_t)*send_sol);
    for(int i=0;i<send_sol/2;i++){
        my_sol_buffer[2*i]=k1[i];
        my_sol_buffer[2*i+1]=k2[i];
    }

    // compute displs
    int *all_displs=NULL;
    uint64_t *all_solutions=NULL;
    if(world_rank==0){
        all_displs=(int*) calloc(world_size,sizeof(int));
        int total_sol=0;
        for(int i=0;i<world_size;i++){
            total_sol+=all_counts[i];
        }
        for(int i=1;i<world_size;i++){
            all_displs[i]=all_displs[i-1]+all_counts[i-1];
        }
        all_solutions=(uint64_t*) malloc(sizeof(uint64_t)*total_sol);
        if(!all_solutions){
            fprintf(stderr,"Rank0 fail: all_solutions alloc\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        MPI_Gatherv(my_sol_buffer, send_sol, MPI_UINT64_T,
                    all_solutions, all_counts, all_displs, MPI_UINT64_T,
                    0, MPI_COMM_WORLD);

        // printing results
        if(total_found>0){
            printf("Total solutions found: %d\n", total_found);
            for(int i=0; i<total_sol; i+=2){
                uint64_t X=all_solutions[i];
                uint64_t Z=all_solutions[i+1];
                // optionally check again is_good_pair
                // then print
                printf("Solution: (%" PRIx64 ", %" PRIx64 ")\n", X, Z);
            }
        } else {
            printf("No solution found.\n");
        }
        free(all_solutions);
        free(all_displs);
    } else {
        MPI_Gatherv(my_sol_buffer, send_sol, MPI_UINT64_T,
                    NULL, NULL, NULL, MPI_UINT64_T,
                    0, MPI_COMM_WORLD);
    }

    free(my_sol_buffer);
    free(all_counts);

    return total_found;
}

/**
 * Prints statistics for the distributed dictionary across all MPI ranks.
 */
void distributed_dict_print_stats(void)
{
    // First, retrieve the actual counts for dictionary insertions and probes
    uint64_t local_dict_insert, local_dict_probe;
    dictionary_get_usage(&local_dict_insert, &local_dict_probe);

    // Populate the stats structure with the retrieved counts
    stats.dict_insert_calls = local_dict_insert;
    stats.dict_probe_calls  = local_dict_probe;

    // Pack the local statistics into an array
    uint64_t local_data[7];
    local_data[0] = stats.local_insert_count;
    local_data[1] = stats.local_received_count;
    local_data[2] = stats.local_query_sent;
    local_data[3] = stats.local_query_received;
    local_data[4] = stats.local_probe_count;
    local_data[5] = stats.dict_insert_calls;
    local_data[6] = stats.dict_probe_calls;

    // Allocate memory for global_data on rank 0
    uint64_t *global_data = NULL;
    if (world_rank == 0) {
        global_data = malloc(sizeof(uint64_t) * 7 * world_size);
        if (global_data == NULL) {
            fprintf(stderr, "Failed to allocate memory for global_data.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Gather all local_data arrays to the global_data array on rank 0
    MPI_Gather(local_data, 7, MPI_UINT64_T,
               global_data, 7, MPI_UINT64_T,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Print statistics for each rank
        printf("\n==== [Distributed Dictionary Stats] ====\n");
        printf("  Processes: %d\n", world_size);

        uint64_t sum_insert = 0, sum_recv = 0, sum_qsent = 0, sum_qrecv = 0, sum_probe = 0;
        uint64_t sum_dict_insert = 0, sum_dict_probe = 0;

        for(int r = 0; r < world_size; r++) {
            uint64_t s_ins   = global_data[r*7 + 0];
            uint64_t s_recv  = global_data[r*7 + 1];
            uint64_t s_qsent = global_data[r*7 + 2];
            uint64_t s_qrecv = global_data[r*7 + 3];
            uint64_t s_probe = global_data[r*7 + 4];
            uint64_t d_ins   = global_data[r*7 + 5];
            uint64_t d_probe = global_data[r*7 + 6];

            printf("  [Rank %d] DistInsert=%" PRIu64 ", DistRecv=%" PRIu64
                   ", DistQSent=%" PRIu64 ", DistQRecv=%" PRIu64 ", DistProbes=%" PRIu64
                   ", DictInsertCalls=%" PRIu64 ", DictProbeCalls=%" PRIu64 "\n",
                   r, s_ins, s_recv, s_qsent, s_qrecv, s_probe, d_ins, d_probe);

            sum_insert      += s_ins;
            sum_recv        += s_recv;
            sum_qsent       += s_qsent;
            sum_qrecv       += s_qrecv;
            sum_probe       += s_probe;
            sum_dict_insert += d_ins;
            sum_dict_probe  += d_probe;
        }

        // Print the total statistics
        printf("\n  [Totals]\n");
        printf("    DistInsert (total): %" PRIu64 "\n", sum_insert);
        printf("    DistReceived (total): %" PRIu64 "\n", sum_recv);
        printf("    DistQueries Sent    : %" PRIu64 "\n", sum_qsent);
        printf("    DistQueries Received: %" PRIu64 "\n", sum_qrecv);
        printf("    DistProbe calls     : %" PRIu64 "\n", sum_probe);
        printf("    DictInsert calls    : %" PRIu64 "\n", sum_dict_insert);
        printf("    DictProbe calls     : %" PRIu64 "\n", sum_dict_probe);
        printf("========================================\n\n");

        // Free the allocated memory for global_data
        free(global_data);
    }
}
