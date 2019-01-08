#include "blas_wrapper.h"

/////////////////////////////////////////////////////////////////////
//         REPLACEMENT FUNCTIONS IN CASE BLAS NOT AVAILABLE        //
//      THESE ARE SLOW - NO ATTEMPTS AT OPTIMISATION WERE MADE     //
//           THESE ARE *NOT* THE GENERAL BLAS FUNCTIONS:           //
//        THEY **ONLY** APPLY TO SQUARE ROW-MAJOR MATRICES!!!      //
/////////////////////////////////////////////////////////////////////
namespace blas {

void Gemm(const int size, const Complex *alpha, const Complex *matrix_a,
              const Complex *matrix_b, const Complex *beta, Complex *matrix_c) {
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, alpha, matrix_a, size, matrix_b, size, beta, matrix_c, size);
    #else
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, alpha, matrix_a, size, matrix_b, size, beta, matrix_c, size);
    #endif
#else

    int i, j, n;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        matrix_c[i + j * size] *= (*beta);
        for (n = 0; n < size; ++n) {
          matrix_c[i + j * size] += (*alpha) * (matrix_a[i + n * size] * matrix_b[j * size + n]);
        }
      }
    }
#endif
}


void Hemm(const enum CBLAS_SIDE Side,
              const int size,
              const Complex *alpha, const Complex *matrix_a,
              const Complex *matrix_b, const Complex *beta,
              Complex *matrix_c) {
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_chemm(CblasColMajor, Side, CblasUpper,
                size, size, alpha, matrix_a, size, matrix_b, size, beta, matrix_c, size);
    #else
    cblas_zhemm(CblasColMajor, Side, CblasUpper,
                size, size, alpha, matrix_a, size, matrix_b, size, beta, matrix_c, size);
    #endif
#else
    if (Side == CblasLeft) {
      Gemm(size, alpha, matrix_a, matrix_b, beta, matrix_c);
    } else if (Side == CblasRight) {
      Gemm(size, alpha, matrix_b, matrix_a, beta, matrix_c);
    }
#endif
}

void Symm(const enum CBLAS_SIDE Side,
              const int size,
                 const Complex *alpha, const Complex *matrix_a,
                 const Complex *matrix_b, const Complex *beta,
                 Complex *matrix_c){
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_csymm(CblasColMajor, Side, CblasUpper,
                size, size, alpha, matrix_a, size, matrix_b, size, beta, matrix_c, size);
    #else
    cblas_zsymm(CblasColMajor, Side, CblasUpper,
                size, size, alpha, matrix_a, size, matrix_b, size, beta, matrix_c, size);
    #endif
#else
    if (Side == CblasLeft) {
        Gemm(size, alpha, matrix_a, matrix_b, beta, matrix_c);
    } else if (Side == CblasRight) {
        Gemm(size, alpha, matrix_b, matrix_a, beta, matrix_c);
    }
#endif
}

void Gemv(const int size, const Complex *alpha,
                 const Complex *matrix_a, const Complex *x,
                 const Complex *beta, Complex *y){
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cgemv(CblasColMajor, CblasNoTrans,
                size, size, alpha, matrix_a, size, x, 1, beta, y, 1);
    #else
    cblas_zgemv(CblasColMajor, CblasNoTrans,
                size, size, alpha, matrix_a, size, x, 1, beta, y, 1);
    #endif
#else
    int i, n;

    //multiply beta:
    for(i = 0; i < size; ++i)
        y[i] *= (*beta);

    for(i = 0; i < size; ++i) {
        for (n = 0; n < size; ++n)
            y[i] += (*alpha) * matrix_a[n * size + i] * x[n];
    }
#endif
}

void Hemv(const int size, const Complex *alpha, const Complex *matrix_a,
              const Complex *x,
              const Complex *beta, Complex *y){
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cgemv(CblasColMajor, CblasNoTrans,
                size, size, alpha, matrix_a, size, x, 1, beta, y, 1);
    #else
    cblas_zgemv(CblasColMajor, CblasNoTrans,
                size, size, alpha, matrix_a, size, x, 1, beta, y, 1);
    #endif
#else
    Gemv(size, alpha, matrix_a, x, beta, y);
#endif
}

void Her(const int size, const Float alpha, const Complex *x,
                Complex *matrix_a){
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cher(CblasColMajor, CblasUpper, size, alpha, x, 1, matrix_a, size);
    #else
    cblas_zher(CblasColMajor, CblasUpper, size, alpha, x, 1, matrix_a, size);
    #endif
#else
    int i, j;
    for(i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            matrix_a[i * size + j] += alpha * x[i] * conj(x[j]);
        }
    }
#endif
}

void Gerc(const int size,
              const Complex *alpha, const Complex *x,
              const Complex *y, Complex *matrix_a){
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cgerc(CblasColMajor, size, size, alpha, x, 1, y, 1, matrix_a, size);
    #else
    cblas_zgerc(CblasColMajor, size, size, alpha, x, 1, y, 1, matrix_a, size);
    #endif
#else
    int i, j;
    for(i = 0; i < size; ++i){
        for(j = 0; j < size; ++j){
            matrix_a[i * size + j] += (*alpha) * x[i] * y[j];
        }
    }
#endif

}

void Copy(const int size, const Complex *x, Complex *y) {
  Copy(size, x, 1, y, 1);
}

void Copy(const int size, const Complex *x, const int incx,
                 Complex *y, const int incy){
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_ccopy(size, x, incx, y, incy);
    #else
    cblas_zcopy(size, x, incx, y, incy);
    #endif
#else
    int i;
    for(i = 0; i < size; ++i){
        y[i * incy] = x[i * incx];
    }
#endif
}

void Axpy(const int size, const Complex *x, Complex *y) {
  Axpy(size, &kOne, x, 1, y, 1);
}

void Axpy(const int size, const Complex *alpha, const Complex *x, Complex *y) {
  Axpy(size, alpha, x, 1, y, 1);
}

void Axpy(const int size, const Complex *alpha, const Complex *x,
          const int incx, Complex *y, const int incy) {
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_caxpy(size, alpha, x, incx, y, incy);
    #else
    cblas_zaxpy(size, alpha, x, incx, y, incy);
    #endif
#else
    int i;
    for(i = 0; i < size; ++i){
        y[i * incy] = (*alpha) * x[i * incx] + y[i * incy];
    }
#endif
}

/*
 * Conjugated dot product dotc = <x|y>
 */
void DotcSub(const int size, const Complex *x, const int incx,
                  const Complex *y, const int incy, Complex *dotc) {
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cdotc_sub(size, x, incx, y, incy, dotc);
    #else
    cblas_zdotc_sub(size, x, incx, y, incy, dotc);
    #endif
#else
    int i;
    *dotc = 0;
    for (i = 0; i < size; ++i, x += incx, y += incy) {
        *dotc += (*x) * conj(*y);
    }
#endif
}

/*
 * Unconjugated dot product dotu = <x^*|y>
 */
void DotuSub(const int size, const Complex *x, const int incx,
                  const Complex *y, const int incy, Complex *dotu) {
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cdotu_sub(size, x, incx, y, incy, dotu);
    #else
    cblas_zdotu_sub(size, x, incx, y, incy, dotu);
    #endif
#else
    int i;
    *dotu = 0;
    for (i = 0; i < size; ++i, x += incx, y += incy) {
      *dotu += (*x) * (*y);
    }
#endif
}

void Scal(const int size, const Complex *alpha, Complex *x, const int incx) {
#ifndef NOBLAS
    #ifdef SINGLEPRECISION
    cblas_cscal(size, alpha, x, incx);
    #else
    cblas_zscal(size, alpha, x, incx);
    #endif
#else
    int i;
    for (i = 0; i < size; ++i, x += incx) {
      *x *= (*alpha);
    }
#endif
}
}  // namespace blas
