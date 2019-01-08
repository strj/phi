#include <complex>
#include <iomanip>
#include <iostream>

#include "blas_wrapper.h"
#include "complex_matrix.h"

// Place a copy of A multiplied by each element in B.
// A = NxN complex matrix, B = NxN complex matrix
// returns N^2xN^2 complex matrix
Complex *OuterProduct(const Complex *matrix_a,
                      const Complex *matrix_b,
                      const int size) {
  Complex *result;
  result = new Complex[size * size * size * size];
  int ii_size = 0;
  int jj_size = 0;
  for (int ii = 0; ii < size; ++ii, ii_size += size) {
    jj_size = 0;
    for (int jj = 0; jj < size; ++jj, jj_size += size) {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          result[(ii_size + i) * size * size + jj_size + j] =
              matrix_a[i * size + j] * matrix_b[ii_size + jj];
        }
      }
    }
  }
  return result;
}

/*
 * Conjugate transpose of a square matrix.
 */
void ConjugateTranspose(const int size,
                        const Complex *matrix,
                        Complex *adjoint) {
  // TODO(johanstr): Optimize.
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      adjoint[i * size + j] = conj(matrix[j * size + i]);
    }
  }
}

/*
 * Sums an array of matrices.
 */
void SumMatrices(Complex **matrix_array,
                 const phi_size_t array_size,
                 const phi_size_t matrix_size,
                 Complex *sum) {
  blas::Copy(matrix_size, matrix_array[0], sum);
  for (phi_size_t i = 1; i < array_size; ++i) {
    blas::Axpy(matrix_size, matrix_array[i], sum);
  }
}

void DoEigenDecomposition(ZheevrParameters *p,
                          const Complex *matrix,
                          Complex *eigenvalues,
                          Complex *eigenvectors) {
  // Need to make a copy since zheevr destroys the input.
  blas::Copy(p->size * p->size, matrix, p->local_matrix);
  phi_xheevr(p->job,
             p->range,
             p->format,
             &p->size,
             p->local_matrix,
             &p->lda,
             &p->vl,
             &p->vu,
             &p->il,
             &p->iu,
             &p->tolerance,
             &p->number_eigenvalues_returned,
             p->eigenvalues_real,
             eigenvectors,
             &p->ldz,
             p->eigenvalues_support,
             p->complex_workspace,
             &p->size_complex_workspace,
             p->real_workspace,
             &p->size_real_workspace,
             p->integer_workspace,
             &p->size_integer_workspace,
             &p->info);
  Copy(p->number_eigenvalues_returned, p->eigenvalues_real, eigenvalues);
}

void SetElementsToZero(const phi_big_size_t size, Complex *matrix) {
  for (phi_big_size_t i = 0; i < size; ++i) {
    matrix[i] = kZero;
  }
}

void SetElementsToZero(const phi_big_size_t size, Float *matrix) {
  for (phi_big_size_t i = 0; i < size; ++i) {
    matrix[i] = kZerof;
  }
}

void Copy(const phi_big_size_t size, const Float *from, Complex *to) {
  for (phi_big_size_t i = 0; i < size; ++i) {
    to[i] = Complex(from[i], 0);
  }
}

// Computes I_N (*) A - A^H (*) I_N where
// (*) is the outer-product,
// I_N is the NxN identity and
// ^H is the Hermitian conjugate
Complex *ToLiouville(const Complex *matrix, const int size) {
  Complex *result;
  result = new Complex[size * size * size * size];
  int ii_n = 0;
  int jj_n = 0;
  for (int ii = 0; ii < size * size * size * size; ++ii) {
    result[ii] = 0;
  }
  for (int ii = 0; ii < size; ++ii, ii_n += size) {
    jj_n = 0;
    for (int jj = 0; jj < size; ++jj, jj_n += size) {
      for (int i = 0; i < size; ++i) {
        result[(ii_n + i) * size * size + jj_n + i] = matrix[ii_n + jj];
      }
    }
  }
  ii_n = 0;
  for (int ii = 0; ii < size; ++ii, ii_n += size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        result[(ii_n + i) * size * size + ii_n + j] -=
            conj(matrix[j * size + i]);
      }
    }
  }
  return result;
}

void PrintMatrix(const Complex *matrix, const int size) {
  for (int i = 0; i < size * size; i += size) {
    for (int j = 0; j < size; ++j) {
      std::cout << "(" << std::ios_base::showpos << std::setw(5)
                << std::ios_base::scientific
                << std::setprecision(2) << real(matrix[i + j]) << ","
                << imag(matrix[i + j]) << ") ";
    }
    std::cout << "\n";
  }
  std::cout.unsetf(std::ios_base::fixed);
  std::cout.unsetf(std::ios_base::showpos);
}

void PrintMatrix(const Complex *matrix, const Index2D *idx, const int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << std::setw(5) << idx[i].i << " " << idx[i].j << " "
              << std::ios_base::showpos << std::setw(4) << std::ios_base::fixed
              << std::setprecision(1) << "(" << real(matrix[i]) << ","
              << imag(matrix[i]) << ")\n";
    std::cout.unsetf(std::ios_base::fixed);
    std::cout.unsetf(std::ios_base::showpos);
  }
}
