#ifndef PHI_BLAS_WRAPPER_H_
#define PHI_BLAS_WRAPPER_H_

#include "numeric_types.h"

#ifndef NOBLAS
extern "C" {
#include "cblas.h"
}
#else
#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113,
  AtlasConj = 114
};
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
#endif
#endif

namespace blas {

void Gemm(const int size, const Complex *alpha, const Complex *matrix_a,
          const Complex *matrix_b, const Complex *beta, Complex *matrix_c);

void Hemm(const enum CBLAS_SIDE Side, const int size, const Complex *alpha,
          const Complex *matrix_a, const Complex *matrix_b, const Complex *beta,
          Complex *matrix_c);

void Symm(const enum CBLAS_SIDE Side, const int size, const Complex *alpha,
          const Complex *matrix_a, const Complex *matrix_b, const Complex *beta,
          Complex *matrix_c);

void Her(const int size, const Float alpha, const Complex *x,
         Complex *matrix_a);

void Gerc(const int size, const Complex *alpha, const Complex *x,
          const Complex *y, Complex *matrix_a);

void Copy(const int size, const Complex *x, Complex *y);

void Copy(const int size, const Complex *x, const int incx, Complex *y,
          const int incy);

void Axpy(const int size, const Complex *x, Complex *y);
void Axpy(const int size, const Complex *alpha, const Complex *x, Complex *y);
void Axpy(const int size, const Complex *alpha, const Complex *x,
          const int incx, Complex *y, const int incy);

void DotcSub(const int size, const Complex *x, const int incx, const Complex *y,
             const int incy, Complex *dotc);

void DotuSub(const int size, const Complex *x, const int incx, const Complex *y,
             const int incy, Complex *dotu);

void Scal(const int size, const Complex *alpha, Complex *x, const int incx);

void Hemv(const int size, const Complex *alpha, const Complex *matrix_a,
          const Complex *x, const Complex *beta, Complex *y);

void Gemv(const int size, const Complex *alpha, const Complex *matrix_a,
          const Complex *x, const Complex *beta, Complex *y);
}  // namespace blas
#endif  // PHI_BLAS_WRAPPER_H_
