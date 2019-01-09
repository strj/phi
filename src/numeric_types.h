#ifndef PHI_NUMERIC_TYPES_H_
#define PHI_NUMERIC_TYPES_H_

#include <complex>

#ifdef SINGLEPRECISION
typedef float Float;
typedef std::complex<float> Complex
#else
typedef double Float;
typedef std::complex<double> Complex;
#endif  // SINGLEPRECISION

typedef int phi_size_t;
typedef int phi_big_size_t;
const Complex kOne(1.0, 0.0);
const Complex kZero(0.0, 0.0);
const Complex kTwo(2.0, 0.0);
const Complex kNegativeOne(-1.0, 0.0);
const Complex kImaginary(0.0, 1.0);
const Complex kNegativeImaginary(0.0, -1.0);
const Float kZerof(0.0);
#endif  // PHI_NUMERIC_TYPES_H_
