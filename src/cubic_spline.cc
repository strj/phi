#include "cubic_spline.h"
#include "lapack_wrapper.h"

CubicSpline::CubicSpline(Float *y, phi_size_t num_values) {
  size_ = num_values;
  x_ = new Float[size_];
  Float ds = 1./(size_ - 1);
  for (phi_size_t i = 0; i < size_; ++i) {
    x_[i] = i * ds;
  }
  Initialize(y);
}

CubicSpline::CubicSpline(Float *x,
                         Float *y,
                         phi_size_t num_values) {
  size_ = num_values;
  x_ = new Float[size_];
  for (phi_size_t i = 0; i < size_; ++i) {
    x_[i] = x[i];
  }
  Initialize(y);
}

CubicSpline::~CubicSpline() {
  delete []a_;
  delete []b_;
  delete []c_;
  delete []d_;
  delete []x_;
}

Float CubicSpline::Get(Float x) {
  if (x > x_[size_ - 1]) {
    return a_[size_ - 1];
  }
  if (x < x_[0]) {
    return a_[0];
  }
  phi_size_t i = Locate(x);
  Float t = (x - x_[i]) / (x_[i + 1] - x_[i]);
  return a_[i] + b_[i] * t + c_[i] * t * t + d_[i] * t * t * t;
}

void CubicSpline::Initialize(Float *y) {
  a_ = new Float[size_];
  b_ = new Float[size_];
  c_ = new Float[size_];
  d_ = new Float[size_];
  for (phi_size_t i = 0; i < size_; ++i) {
    a_[i] = y[i];
  }
  b_[0] = 3. * (y[1] - y[0]);
  b_[size_ - 1] = 3. * (y[size_ - 1] - y[size_ - 2]);
  for (phi_size_t i = 1; i < size_ - 1; ++i) {
    b_[i] = 3. * (y[i + 1] - y[i]);
  }

  SolveTridiagonal();

  for (phi_size_t i = 0; i < size_ - 1; ++i) {
    c_[i] = 3. * (a_[i + 1] - a_[i]) - 2. * b_[i] - b_[i + 1];
    d_[i] = 2. * (a_[i] - a_[i + 1]) + b_[i] + b_[i + 1];
  }
  last_index_ = 0;
}

phi_size_t CubicSpline::Locate(Float x) {
  if (x > x_[last_index_] && x < x_[last_index_ + 1]) {
    return last_index_;
  }
  phi_size_t i = last_index_;
  while (x > x_[i + 1]) {
    ++i;
  }
  while (x < x_[i]) {
    --i;
  }
  last_index_ = i;
  return i;
}

void CubicSpline::SolveTridiagonal() {
  lapack_int n = size_;
  lapack_int one = 1;
  lapack_int info;
  Float *diagonal = new Float[size_];
  Float *off_diagonal = new Float[size_ - 1];
  diagonal[0] = diagonal[size_ - 1] = 2.;
  for (phi_size_t i = 1; i < size_ - 1; ++i) {
    diagonal[i] = 4.;
  }
  for (phi_size_t i = 0; i < size_ - 1; ++i) {
    off_diagonal[i] = 1.;
  }
  Float *b = new Float[size_];
  for (phi_size_t i = 0; i < size_; ++i) {
    b[i] = b_[i];
  }
  phi_real_xpttrf(
      &n,
      diagonal,
      off_diagonal,
      &info);
  phi_real_xpttrs(
      &n,
      &one,
      diagonal,
      off_diagonal,
      b,
      &n,
      &info);
  for (phi_size_t i = 0; i < size_; ++i) {
    b_[i] = b[i];
  }
  delete []diagonal;
  delete []off_diagonal;
  delete []b;
}
