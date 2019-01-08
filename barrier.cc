#include "barrier.h"

void BarrierInit(barrier_t *barrier, int num_threads) {
  barrier->num_threads = num_threads;
  barrier->called = 0;
  pthread_mutex_init(&barrier->mutex, NULL);
  pthread_cond_init(&barrier->cond, NULL);
}

void BarrierDestroy(barrier_t *barrier) {
  pthread_mutex_destroy(&barrier->mutex);
  pthread_cond_destroy(&barrier->cond);
}

void BarrierWait(barrier_t *barrier) {
  pthread_mutex_lock(&barrier->mutex);
  barrier->called++;
  if (barrier->called == barrier->num_threads) {
    barrier->called = 0;
    pthread_cond_broadcast(&barrier->cond);
  } else {
    pthread_cond_wait(&barrier->cond, &barrier->mutex);
  }
  pthread_mutex_unlock(&barrier->mutex);
}
