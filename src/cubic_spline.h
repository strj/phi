#ifndef PHI_CUBIC_SPLINE_H_
#define PHI_CUBIC_SPLINE_H_

#include "numeric_types.h"

class CubicSpline {
 public:
  CubicSpline(Float *, phi_size_t);
  CubicSpline(Float *, Float *, phi_size_t);
  ~CubicSpline();
  Float Get(Float x);
 private:
  phi_size_t size_;
  Float *a_;
  Float *b_;
  Float *c_;
  Float *d_;
  Float *x_;
  phi_size_t last_index_;

  void Initialize(Float *);
  void SolveTridiagonal();
  phi_size_t Locate(Float);
};

#endif  // PHI_CUBIC_SPLINE_H_

