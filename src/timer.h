/*
 * The timer class assumes that timers of each type are stopped before the
 * following timer is started.
 */
#ifndef PHI_TIMER_H_
#define PHI_TIMER_H_

#include <sys/time.h>

#include <ctime>
#include <string>

// Set up a macro to enable the timers at compile time.
#ifdef TIMERS
#define RECORD_START(timer, type, step) timer->Start(type, step)
#define RECORD_STOP(timer, type, step) timer->Stop(type, step)
#define RECORD_SUMMARY(timer, output) output = timer->GetSummary()
#else
#define RECORD_START(timer, type, step) do {} while (0)
#define RECORD_STOP(timer, type, step) do {} while (0)
#define RECORD_SUMMARY(timer, output) do {} while (0)
#endif  // TIMERS

namespace timer {

enum Type {
  kStart,
  kDrhoUpdate,
  kRhoUpdate,
  kHamiltonianUpdate,
  kTimelocalUpdate,
  kTimestepUpdate,
  kTruncatedSpaceUpdate,
  kHierarchyTruncationUpdate,
  kFileOutput,
  kRestartOutput
};

const int NUM_TYPES = 10;

class TimerTest;

class Timer {
 public:
  explicit Timer(int num_steps);
  ~Timer();
  void Start(Type type, int step);
  void Stop(Type type, int step);
  void Reset();
  std::string GetSummary();

 private:
  friend class TimerTest;

  float GetTotalComputeTime();
  float GetTotalWaitTime();
  float GetTotalTimeFor(const int type, float **time_store) const;

  timeval* current_timer_;
  timeval** timer_;
  float** compute_time_;
  float** wait_time_;
  int** compute_count_;
  int** wait_count_;
  int num_steps_;  // The number of Runga-Kutta steps.
};

}  // namespace timer
#endif  // PHI_TIMER_H_

