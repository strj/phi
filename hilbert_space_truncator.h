#ifndef PHI_HILBERT_SPACE_TRUNCATOR_H_
#define PHI_HILBERT_SPACE_TRUNCATOR_H_

#include "lapack_wrapper.h"
#include "numeric_types.h"

class HilbertSpaceTruncator {
 public:
  HilbertSpaceTruncator(phi_size_t full_size,
                        phi_size_t truncated_size);
  ~HilbertSpaceTruncator();

  void UpdateWith(const Complex* hamiltonian);
  void UpdateFrom(const HilbertSpaceTruncator *from);
  void FillDiagonal(Complex* to);
  void Truncate(const Complex* from, Complex* to);
  void Rotate(Complex* matrix);
  void UnTruncate(const Complex* from, Complex* to);
 private:
  ZheevrParameters *decomposition_parameters_;
  phi_size_t full_size_;
  phi_size_t truncated_size_;
  Complex *eigenvectors_;
  Complex *eigenvalues_;
  Complex *work_space_;
  Complex *rotator_;
  Complex *rotator_adjoint_;
};

#endif  // PHI_HILBERT_SPACE_TRUNCATOR_H_
