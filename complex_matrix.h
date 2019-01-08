#ifndef PHI_COMPLEX_MATRIX_H_
#define PHI_COMPLEX_MATRIX_H_
#include <fstream>

#include "numeric_types.h"
#include "lapack_wrapper.h"

struct Index2D {
  int i;
  int j;
};

void ConjugateTranspose(const int size,
                        const Complex *matrix,
                        Complex *adjoint);
void SumMatrices(Complex **matrix_array,
                 const phi_size_t array_size,
                 const phi_size_t matrix_size,
                 Complex *sum);
void DoEigenDecomposition(ZheevrParameters *p,
                          const Complex *matrix,
                          Complex *eigenvalues,
                          Complex *eigenvectors);
void SetElementsToZero(const phi_big_size_t size, Complex *matrix);
void SetElementsToZero(const phi_big_size_t size, Float *matrix);
void Copy(const phi_big_size_t size, const Complex *from, Complex *to);
void Copy(const phi_big_size_t size, const Float *from, Complex *to);

// Equal size square matrices only.
Complex* OuterProduct(const Complex *matrix_a,
                      const Complex *matrix_b,
                      const int size);
Complex* ToLiouville(const Complex *matrix, const int size);
void PrintMatrix(const Complex *matrix, const int size);
void PrintMatrix(const Complex *matrix, const Index2D *idx, const int size);

#endif  // PHI_COMPLEX_MATRIX_H_

