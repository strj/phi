#include "timer.h"

#include <sstream>

namespace timer {

Timer::Timer(int num_steps): num_steps_(num_steps) {
  timer_ = new timeval*[NUM_TYPES];
  compute_time_ = new float*[NUM_TYPES];
  wait_time_ = new float*[NUM_TYPES];
  compute_count_ = new int*[NUM_TYPES];
  wait_count_ = new int*[NUM_TYPES];
  for (int i = 0; i < NUM_TYPES; ++i) {
    timer_[i] = new timeval[num_steps_];
    compute_time_[i] = new float[num_steps_];
    wait_time_[i] = new float[num_steps_];
    compute_count_[i] = new int[num_steps_];
    wait_count_[i] = new int[num_steps_];
    for (int j = 0; j < num_steps_; ++j) {
      compute_time_[i][j] = 0;
      wait_time_[i][j] = 0;
      compute_count_[i][j] = 0;
      wait_count_[i][j] = 0;
    }
  }
  current_timer_ = NULL;
}

Timer::~Timer() {
  for (int i = 0; i < NUM_TYPES; ++i) {
    delete []timer_[i];
    delete []compute_time_[i];
    delete []wait_time_[i];
    delete []compute_count_[i];
    delete []wait_count_[i];
  }
  delete []timer_;
  delete []compute_time_;
  delete []wait_time_;
  delete []compute_count_;
  delete []wait_count_;
  delete current_timer_;
}

float GetTimeDifference(const timeval &t1, const timeval &t2) {
  return t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6;
}

/*
 * Notes the current time for the specified type.
 */
void Timer::Start(Type type, int step) {
  if (step < num_steps_) {
    if (current_timer_ != NULL) {
      gettimeofday(&(timer_[type][step]), NULL);
      wait_time_[type][step] += GetTimeDifference(*current_timer_,
                                                  timer_[type][step]);
      ++wait_count_[type][step];
    } else {
      current_timer_ = new timeval;
    }
    gettimeofday(current_timer_, NULL);
  }
}

void Timer::Stop(Type type, int step) {
  if (step < num_steps_ && current_timer_ != NULL) {
    gettimeofday(&(timer_[type][step]), NULL);
    compute_time_[type][step] += GetTimeDifference(*current_timer_,
                                                  timer_[type][step]);
    ++compute_count_[type][step];
    gettimeofday(current_timer_, NULL);
  }
}

std::string Timer::GetSummary() {
  float total_compute_time = GetTotalComputeTime();
  float total_wait_time = GetTotalWaitTime();

  std::stringstream ss;
  ss << "Timing Summary:\n"
     << "Total time: " << total_compute_time + total_wait_time << " s\n"
     << "Compute time: " << total_compute_time << " s\n"
     << "Wait time: " << total_wait_time << " s\n"
     << "drho update time: " << GetTotalTimeFor(
         Type::kDrhoUpdate, compute_time_) << " s\n"
     << "rho update time: " << GetTotalTimeFor(
         Type::kRhoUpdate, compute_time_) << " s\n"
     << "Hamiltonian update time: " << GetTotalTimeFor(
         Type::kHamiltonianUpdate, compute_time_) << " s\n"
     << "Timestep update time: " << GetTotalTimeFor(
         Type::kTimestepUpdate, compute_time_) << " s\n"
     << "Truncated space update time: " << GetTotalTimeFor(
         Type::kTruncatedSpaceUpdate, compute_time_) << " s\n"
     << "Adaptive truncation update time: " << GetTotalTimeFor(
         Type::kHierarchyTruncationUpdate, compute_time_) << " s\n"
     << "File output time: " << GetTotalTimeFor(
         Type::kFileOutput, compute_time_) << " s\n"
     << "Restart output time: " << GetTotalTimeFor(
         Type::kRestartOutput, compute_time_) << " s\n";
  return ss.str();
}

float Timer::GetTotalComputeTime() {
  float total_compute_time_ = 0;
  for (int j = 0; j < NUM_TYPES; ++j) {
    total_compute_time_ += GetTotalTimeFor(j, compute_time_);
  }
  return total_compute_time_;
}

float Timer::GetTotalWaitTime() {
  float total_wait_time_ = 0;
  for (int j = 0; j < NUM_TYPES; ++j) {
    total_wait_time_ += GetTotalTimeFor(j, wait_time_);
  }
  return total_wait_time_;
}

float Timer::GetTotalTimeFor(const int type, float **time_store) const {
  float time = 0;
  for (int i = 0; i < num_steps_; ++i) {
    time += time_store[type][i];
  }
  return time;
}

void Timer::Reset() {
  for (int i = 0; i < NUM_TYPES; ++i) {
    for (int j = 0; j < num_steps_; ++j) {
      compute_time_[i][j] = 0;
      wait_time_[i][j] = 0;
      compute_count_[i][j] = 0;
      wait_count_[i][j] = 0;
    }
  }
}

}  // namespace timer
