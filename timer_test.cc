#include "timer.h"

#include <stdlib.h>

#include "testing/base/public/gunit.h"


namespace timer {

float kDelta = 1e-3;

class TimerTest : public ::testing::Test {
 protected:
  Timer *timer_;

  TimerTest() {
    timer_ = new Timer(2 /* integration sub-steps */);
  }

  virtual ~TimerTest() {
    delete timer_;
  }

  float GetTotalComputeTime() {
    return timer_->GetTotalComputeTime();
  }

  float GetTotalWaitTime() {
    return timer_->GetTotalWaitTime();
  }
};

TEST_F(TimerTest, TestTimerStartStop) {
  timer_->Start(Type::kStart, 0);
  sleep(2);
  timer_->Stop(Type::kStart, 0);

  EXPECT_NEAR(2, GetTotalComputeTime(), kDelta);
  EXPECT_NEAR(0, GetTotalWaitTime(), kDelta);
}

TEST_F(TimerTest, TestWaitingTimer) {
  timer_->Start(Type::kStart, 0);
  sleep(2);
  timer_->Stop(Type::kStart, 0);
  sleep(1);
  timer_->Start(Type::kDrhoUpdate, 0);
  sleep(2);
  timer_->Stop(Type::kDrhoUpdate, 0);
  sleep(1);
  timer_->Start(Type::kDrhoUpdate, 1);
  sleep(2);
  timer_->Stop(Type::kDrhoUpdate, 1);

  EXPECT_NEAR(6, GetTotalComputeTime(), kDelta);
  EXPECT_NEAR(2, GetTotalWaitTime(), kDelta);
}

TEST_F(TimerTest, TestTimerMacroNoOp) {
  RECORD_START(timer_, Type::kStart, 0);
  sleep(2);
  RECORD_STOP(timer_, Type::kStart, 0);
  sleep(1);
  RECORD_START(timer_, Type::kDrhoUpdate, 0);
  sleep(2);
  RECORD_STOP(timer_, Type::kDrhoUpdate, 0);
  sleep(1);
  RECORD_START(timer_, Type::kDrhoUpdate, 1);
  sleep(2);
  RECORD_STOP(timer_, Type::kDrhoUpdate, 1);

  EXPECT_NEAR(0, GetTotalComputeTime(), kDelta);
  EXPECT_NEAR(0, GetTotalWaitTime(), kDelta);
}
}  // namespace timer
