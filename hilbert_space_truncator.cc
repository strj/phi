#include "hilbert_space_truncator.h"

#include "blas_wrapper.h"
#include "complex_matrix.h"
#include "lapack_wrapper.h"
#include "numeric_types.h"

HilbertSpaceTruncator::HilbertSpaceTruncator(
    phi_size_t full_size,
    phi_size_t truncated_size): full_size_(full_size),
                                truncated_size_(truncated_size) {
  decomposition_parameters_ = new ZheevrParameters(full_size, truncated_size_);
  eigenvalues_ = new Complex[truncated_size_];
  eigenvectors_ = new Complex[full_size_ * truncated_size_];
  SetElementsToZero(truncated_size_, eigenvalues_);
  SetElementsToZero(full_size_ * truncated_size_, eigenvectors_);
  work_space_ = new Complex[full_size_ * full_size_];
  rotator_ = new Complex[truncated_size_ * truncated_size_];
  rotator_adjoint_ = new Complex[truncated_size_ * truncated_size_];
}

HilbertSpaceTruncator::~HilbertSpaceTruncator() {
  delete decomposition_parameters_;
  delete []eigenvectors_;
  delete []eigenvalues_;
  delete []work_space_;
  delete []rotator_;
  delete []rotator_adjoint_;
}

/*
 * Truncates a Hermitian operator.
 */
void HilbertSpaceTruncator::Truncate(const Complex *from, Complex *to) {
  for (phi_size_t i = 0; i < truncated_size_; ++i) {
    // Calculate from | eigenvectors[i] >
    blas::Hemv(full_size_,
               &kOne,
               from,
               &(eigenvectors_[i * full_size_]),
               &kZero,
               work_space_);
    blas::DotcSub(full_size_,
                  &(eigenvectors_[i * full_size_]),
                  1,
                  work_space_,
                  1,
                  &(to[i * truncated_size_ + i]));
    for (phi_size_t j = 0; j < i; ++j) {
      // Calculate to[i, j] = <eigenvectors[j] | from | eigenvectors[i] >
      blas::DotcSub(full_size_,
                    &(eigenvectors_[j * full_size_]),
                    1,
                    work_space_,
                    1,
                    &(to[i * truncated_size_ + j]));
      // Assign hermitian conjugate element.
      to[j * truncated_size_ + i] = conj(to[i * truncated_size_ + j]);
    }
  }
}

/*
 * Rotates a Hermitian matrix from the previous basis to the current basis.
 */
void HilbertSpaceTruncator::Rotate(Complex *matrix) {
  blas::Hemm(CblasLeft, truncated_size_,
             &kOne,
             matrix,
             rotator_adjoint_,
             &kZero,
             work_space_);
  blas::Gemm(truncated_size_,
             &kOne,
             rotator_,
             work_space_,
             &kZero,
             matrix);
}

/*
 * Untruncates a Hermitian operator (eigenexpansion).
 */
void HilbertSpaceTruncator::UnTruncate(const Complex *from, Complex *to) {
  for (phi_size_t i = 0; i < truncated_size_; ++i) {
    for (phi_size_t j = 0; j < truncated_size_; ++j) {
      blas::Gerc(full_size_,
                 &(from[i * truncated_size_ + j]),
                 &(eigenvectors_[j * full_size_]),
                 &(eigenvectors_[i * full_size_]),
                 to);
    }
  }
}

/*
 * Updates the truncator with a new Hamiltonian.
 */
void HilbertSpaceTruncator::UpdateWith(const Complex *hamiltonian) {
  // Store old eigenvectors.
  blas::Copy(full_size_ * truncated_size_, eigenvectors_, work_space_);
  // Update eigenvectors with new hamiltonian.
  DoEigenDecomposition(decomposition_parameters_,
                       hamiltonian,
                       eigenvalues_,
                       eigenvectors_);

  // Update the rotator matrix elements.
  for (phi_size_t i = 0; i < truncated_size_; ++i) {
    for (phi_size_t j = 0; j < truncated_size_; ++j) {
      blas::DotcSub(full_size_,
                    &(work_space_[i * full_size_]), 1,
                    &(eigenvectors_[j * full_size_]), 1,
                    &(rotator_[i * truncated_size_ + j]));
    }
  }
  ConjugateTranspose(truncated_size_, rotator_, rotator_adjoint_);
}

/*
 * Updates the truncator from another truncator.
 */
void HilbertSpaceTruncator::UpdateFrom(const HilbertSpaceTruncator *from) {
  blas::Copy(truncated_size_ * truncated_size_, from->rotator_, rotator_);
  blas::Copy(truncated_size_ * truncated_size_, from->rotator_adjoint_,
             rotator_adjoint_);
  blas::Copy(truncated_size_, from->eigenvalues_, eigenvalues_);
  blas::Copy(truncated_size_ * full_size_, from->eigenvectors_, eigenvectors_);
}

/*
 * Fills the Hamiltonian eigenvalues to the 'to' matrix.
 */
void HilbertSpaceTruncator::FillDiagonal(Complex *to) {
  SetElementsToZero(truncated_size_ * truncated_size_, to);
  for (phi_size_t i = 0; i < truncated_size_; ++i) {
    to[i * truncated_size_ + i] = eigenvalues_[i];
  }
}
