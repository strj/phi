#ifndef PHI_BARRIER_H_
#define PHI_BARRIER_H_
#include <pthread.h>

typedef struct {
    int num_threads;
    int called;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} barrier_t;

void BarrierDestroy(barrier_t *barrier);
void BarrierInit(barrier_t *barrier, int num_threads);
void BarrierWait(barrier_t *barrier);

#endif  // PHI_BARRIER_H_

