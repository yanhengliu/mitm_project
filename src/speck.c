#include "speck.h"
#include <assert.h>

#define ROTL32(x,r) (((x)<<(r)) | (x>>(32-(r))))
#define ROTR32(x,r) (((x)>>(r)) | ((x)<<(32-(r))))
#define ER32(x,y,k) (x=ROTR32(x,8), x+=y, x^=k, y=ROTL32(y,3), y^=x)
#define DR32(x,y,k) (y^=x, y=ROTR32(y,3), x^=k, x-=y, x=ROTL32(x,8))

static void Speck64128KeySchedule(const u32 K[],u32 rk[])
{
    u32 i,D=K[3],C=K[2],B=K[1],A=K[0];
    for(i=0;i<27;){
        rk[i]=A; ER32(B,A,i++);
        rk[i]=A; ER32(C,A,i++);
        rk[i]=A; ER32(D,A,i++);
    }
}

static void Speck64128Encrypt(const u32 Pt[], u32 Ct[], const u32 rk[])
{
    u32 i;
    Ct[0]=Pt[0]; Ct[1]=Pt[1];
    for(i=0;i<27;)
        ER32(Ct[1],Ct[0],rk[i++]);
}

static void Speck64128Decrypt(u32 Pt[], const u32 Ct[], u32 const rk[])
{
    int i;
    Pt[0]=Ct[0]; Pt[1]=Ct[1];
    for(i=26;i>=0;)
        DR32(Pt[1],Pt[0],rk[i--]);
}

/* Define global variables */
u64 mask;
u32 C[2][2];
u32 P[2][2] = {{0,0},{0xffffffff,0xffffffff}};
u64 n;

/* f(k) */
u64 f(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {(u32)(k & 0xffffffffU), (u32)(k >> 32), 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Ct[2];
    Speck64128Encrypt(P[0], Ct, rk);
    return (((u64)Ct[0]) ^ (((u64)Ct[1]) << 32)) & mask;
}

/* g(k) */
u64 g(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {(u32)(k & 0xffffffffU), (u32)(k >> 32), 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Pt[2];
    Speck64128Decrypt(Pt, C[0], rk);
    return (((u64)Pt[0]) ^ (((u64)Pt[1]) << 32)) & mask;
}

bool is_good_pair(u64 k1, u64 k2)
{
    u32 Ka[4] = {(u32)(k1 & 0xffffffffU), (u32)(k1 >> 32),0,0};
    u32 Kb[4] = {(u32)(k2 & 0xffffffffU), (u32)(k2 >> 32),0,0};
    u32 rka[27], rkb[27];
    Speck64128KeySchedule(Ka, rka);
    Speck64128KeySchedule(Kb, rkb);
    u32 mid[2], Ct_t[2];
    Speck64128Encrypt(P[1], mid, rka);
    Speck64128Encrypt(mid, Ct_t, rkb);
    return (Ct_t[0] == C[1][0]) && (Ct_t[1] == C[1][1]);
}
