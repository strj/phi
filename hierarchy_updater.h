/*
 * The classes here contain the methods that implement the HEOM. They are the
 * inner-most loops of the integration so extra care has been taken to avoid
 * variable construction and branching, leading to somewhat odd looking code.
 */
#ifndef PHI_HIERARCHY_UPDATER_H_
#define PHI_HIERARCHY_UPDATER_H_

#include "blas_wrapper.h"
#include "complex_matrix.h"
#include "hierarchy_node.h"
#include "numeric_types.h"
#include "phi_parameters.h"

namespace hierarchy_updater {

const Complex kZero(0, 0);
const Complex kOne(1, 0);
const Complex kTwo(2, 0);
const Complex kNegativeOne(-1, 0);
const Complex kImaginary(0, 1);
const Complex kNegativeImaginary(0, -1);

class HierarchyUpdater {
 public:
  explicit HierarchyUpdater(const PhiParameters* parameters,
                            Complex liouville_prefactor):
      parameters_(parameters),
      size_(parameters->num_states),
      num_elements_(parameters->num_states * parameters->num_states),
      num_bath_operators_(parameters->num_bath_couplings),
      liouville_prefactor_(Complex(0, -1. / parameters->hbar)),
      negative_liouville_prefactor_(Complex(0, 1. / parameters->hbar)) {}
  virtual ~HierarchyUpdater() {}

  void Get(const HierarchyNode &node, Complex* drho) {
    // drho = -i/hbar [H, rho]
    DoUnitary(node, drho);
    // drho += -sum_m (sum_k n_ak nu_ak) rho.
    DoDephasing(node, drho);
    // -sum_ alpha_a [F_a, [F_a, rho]]
    DoMatsubara(node, drho);
    if (node.is_timelocal_truncated) {
      // -sum_m [ V_m (Q_m rho - rho Q_m) - (Q_m rho - rho Q_m) V_m ]
      DoTimelocal(node, drho);
    } else {
      // -i sum_a sum_k [F_a, rho_n(ak)+]
      DoNext(node, drho);
    }
    // -i sum_a sum_k n_ak (c_ak F_a rho_n(ak)- - c_ak^* rho_n(ak)- F_a)
    DoPrevious(node, drho);
  }

 protected:
  const PhiParameters* parameters_;
  const phi_size_t size_;
  const phi_size_t num_elements_;
  const phi_size_t num_bath_operators_;
  const Complex liouville_prefactor_;
  const Complex negative_liouville_prefactor_;

  phi_size_t n_;

  inline void DoUnitary(const HierarchyNode &node, Complex* drho) const {
    // Unitary: calculates drho = -i / hbar [H, rho].
    blas::Hemm(CblasRight, size_, &liouville_prefactor_,
               node.density_matrix, node.hamiltonian, &kZero, drho);
    blas::Hemm(CblasLeft, size_, &negative_liouville_prefactor_,
               node.density_matrix, node.hamiltonian_adjoint, &kOne, drho);
  }
  inline void DoDephasing(const HierarchyNode &node, Complex* drho) {
    for (n_ = 0; n_ < num_elements_; ++n_) {
      drho[n_] -= node.dephasing_prefactor * node.density_matrix[n_];
    }
  }
  virtual void DoMatsubara(const HierarchyNode &node, Complex* drho) = 0;
  virtual void DoTimelocal(const HierarchyNode &node, Complex* drho) = 0;
  virtual void DoNext(const HierarchyNode &node, Complex* drho) = 0;
  virtual void DoPrevious(const HierarchyNode &node, Complex* drho) = 0;
};

/*
 * Calculates the update to the density matrix assuming arbitrary system-bath
 * coupling.
 */
class FullBath: public HierarchyUpdater {
 public:
  FullBath(const PhiParameters* parameters,
           Complex liouville_prefactor,
           const Complex* const* timelocal,
           const Complex* const* timelocal_adjoint,
           const Complex* same,
           const Complex* next,
           const Complex* prev):
      HierarchyUpdater(parameters, liouville_prefactor),
      timelocal_(timelocal),
      timelocal_adjoint_(timelocal_adjoint),
      same_(same),
      next_(next),
      prev_(prev) {
    temp_ = new Complex[parameters->num_states * parameters->num_states];
  }
  ~FullBath() {
    delete[] temp_;
  }

 protected:
  const Complex* const* timelocal_;
  const Complex* const* timelocal_adjoint_;
  // Pointer to bath operators for coupling to different hierarchy levels.
  const Complex* same_;
  const Complex* next_;
  const Complex* prev_;

  phi_size_t m_;

  const Complex* bath_;
  const Complex* density_;
  Complex* temp_;
  Complex prefactor_;

  void DoMatsubara(const HierarchyNode &node, Complex* drho);
  void DoTimelocal(const HierarchyNode &node, Complex* drho);
  void DoNext(const HierarchyNode &node, Complex* drho);
  void DoPrevious(const HierarchyNode &node, Complex* drho);
};

/*
 * Calculates the integration update for a system with diagonal system-bath
 * coupling operators.
 */
class DiagonalBath: public HierarchyUpdater {
 public:
  DiagonalBath(const PhiParameters* parameters,
               Complex liouville_prefactor,
               const Complex* const* timelocal,
               const Complex* const* timelocal_adjoint,
               const Complex* same,
               const Complex* next,
               const Complex* prev):
      HierarchyUpdater(parameters, liouville_prefactor),
      timelocal_(timelocal),
      timelocal_adjoint_(timelocal_adjoint),
      same_(same),
      next_(next),
      prev_(prev) {
    temp_ = new Complex[parameters->num_states * parameters->num_states];
  }
  ~DiagonalBath() {
    delete[] temp_;
  }

 protected:
  const Complex* const* timelocal_;
  const Complex* const* timelocal_adjoint_;
  // Pointer to bath operators for coupling to different hierarchy levels.
  const Complex* same_;
  const Complex* next_;
  const Complex* prev_;

  phi_size_t m_, j_;  // Loop counters;

  // Temporary work space.
  Complex* temp_;
  const Complex* density_;
  const Complex* bath_;
  Complex prefactor_;

  void DoMatsubara(const HierarchyNode &node, Complex* drho);
  void DoTimelocal(const HierarchyNode &node, Complex* drho);
  void DoNext(const HierarchyNode &node, Complex* drho);
  void DoPrevious(const HierarchyNode &node, Complex* drho);
};

/*
 * Calculates the update to the density matrix. This version assumes
 * that the bath DOF are independently coupled to each state |i> and that the
 * hierarchy is always present (no adaptive truncation).
 */
class IndependentBath: public HierarchyUpdater {
 public:
  IndependentBath(const PhiParameters* parameters,
                  Complex liouville_prefactor,
                  const Complex* const* timelocal,
                  const Complex* const* timelocal_adjoint):
      HierarchyUpdater(parameters, liouville_prefactor),
      timelocal_(timelocal),
      timelocal_adjoint_(timelocal_adjoint) {
    temp_ = new Complex[parameters->num_states * parameters->num_states];
  }
  ~IndependentBath() {
    delete []temp_;
  }

 protected:
  const Complex* const* timelocal_;
  const Complex* const* timelocal_adjoint_;

  Complex* temp_;

  const Complex* density_;
  Complex* update_;
  phi_size_t m_, j_;

  void DoMatsubara(const HierarchyNode &node, Complex* drho);
  void DoTimelocal(const HierarchyNode &node, Complex* drho);
  void DoMultiBathMatsubara(const HierarchyNode &node, Complex* drho);
  void DoMultiBathTimelocal(const HierarchyNode &node, Complex* drho);
  void DoNext(const HierarchyNode &node, Complex* drho);
  void DoPrevious(const HierarchyNode &node, Complex* drho);
};

/*
 * Calculates the integration update for a hierarchy node with independently
 * coupled environments to each state using adaptive hierarchy truncation.
 */
class IndependentBathAdaptiveTruncation: public IndependentBath {
 public:
  IndependentBathAdaptiveTruncation(const PhiParameters* parameters,
                             Complex liouville_prefactor,
                             const Complex* const* timelocal,
                             const Complex* const* timelocal_adjoint):
      IndependentBath(parameters,
                      liouville_prefactor,
                      timelocal,
                      timelocal_adjoint) {}
  void Get(const HierarchyNode &node, Complex* drho) {
    if (node.is_active) {
      DoUnitary(node, drho);
      DoDephasing(node, drho);
      DoMatsubara(node, drho);
      if (node.is_timelocal_truncated) {
        DoTimelocal(node, drho);
      }
    }
    if (!node.is_timelocal_truncated) {
      DoNext(node, drho);
    }
    DoPrevious(node, drho);
  }

 protected:
  void DoNext(const HierarchyNode &node, Complex* drho);
  void DoPrevious(const HierarchyNode &node, Complex* drho);
};

/*
 * Calculates the update to the density matrix that assumes no environment
 * coupling.
 */
class Vacuum : public HierarchyUpdater {
 public:
    Vacuum(const PhiParameters* parameters,
           Complex liouville_prefactor):
        HierarchyUpdater(parameters, liouville_prefactor) {}
  void Get(const HierarchyNode &node, Complex* drho) {
    DoUnitary(node, drho);
  }

 protected:
  void DoMatsubara(const HierarchyNode &node, Complex* drho) {}
  void DoTimelocal(const HierarchyNode &node, Complex* drho) {}
  void DoNext(const HierarchyNode &node, Complex* drho) {}
  void DoPrevious(const HierarchyNode &node, Complex* drho) {}
};

/*
 * Calculates the update to the density matrix. This version assumes the
 * hierarchy is truncated at the first tier and time-local truncation is applied
 * (otherwise there would be no bath interaction), and there are abritrary
 * system-bath coupling terms.
 */
class NoHierarchyFullBath: public FullBath {
 public:
  NoHierarchyFullBath(const PhiParameters* parameters,
                      Complex liouville_prefactor,
                      const Complex* const* timelocal,
                      const Complex* const* timelocal_adjoint,
                      const Complex* same,
                      const Complex* next):
      FullBath(parameters,
               liouville_prefactor,
               timelocal,
               timelocal_adjoint,
               same,
               next,
               NULL /* prev not used */) {}

  void Get(const HierarchyNode &node, Complex* drho) {
    DoUnitary(node, drho);
    DoMatsubara(node, drho);
    DoTimelocal(node, drho);
  }
};

/*
 * Calculates the update to the density matrix. This version assumes the
 * hierarchy is truncated at the first tier and time-local truncation is applied
 * (otherwise there would be no bath interaction), and there are abritrary
 * (diagonal) system-bath coupling terms.
 */
class NoHierarchyDiagonalBath: public DiagonalBath {
 public:
  NoHierarchyDiagonalBath(const PhiParameters* parameters,
                          Complex liouville_prefactor,
                          const Complex* const* timelocal,
                          const Complex* const* timelocal_adjoint,
                          const Complex* same,
                          const Complex* next):
      DiagonalBath(parameters,
                   liouville_prefactor,
                   timelocal,
                   timelocal_adjoint,
                   same,
                   next,
                   NULL /* prev not used */) {}

  void Get(const HierarchyNode &node, Complex* drho) {
    DoUnitary(node, drho);
    DoMatsubara(node, drho);
    DoTimelocal(node, drho);
  }
};

/*
 * Calculates the update to the density matrix. This version assumes the
 * hierarchy is truncated at the first tier and time-local truncation is applied
 * (otherwise there would be no bath interaction), and each state is
 * independently coupled to the environment.
 */
class NoHierarchyIndependentBath: public IndependentBath {
 public:
  NoHierarchyIndependentBath(const PhiParameters* parameters,
                             Complex liouville_prefactor,
                             const Complex* const* timelocal,
                             const Complex* const* timelocal_adjoint):
    IndependentBath(parameters,
                    liouville_prefactor,
                    timelocal,
                    timelocal_adjoint) {}
  void Get(const HierarchyNode &node, Complex* drho) {
    DoUnitary(node, drho);
    DoMatsubara(node, drho);
    DoTimelocal(node, drho);
  }
};

class SpectrumHierarchyUpdater: public HierarchyUpdater {
 public:
  SpectrumHierarchyUpdater(const PhiParameters* parameters,
                           Complex liouville_prefactor):
      HierarchyUpdater(parameters, liouville_prefactor) {}

 protected:
  phi_size_t n_;

  void DoUnitary(const HierarchyNode &node, Complex* drho) {
    blas::Hemv(size_, &liouville_prefactor_,
               node.hamiltonian, node.density_matrix, &kZero, drho);
  }
  void DoDephasing(const HierarchyNode &node, Complex* drho) {
    for (n_ = 0; n_ < size_; ++n_) {
      drho[n_] -= node.dephasing_prefactor * node.density_matrix[n_];
    }
  }
};

class SpectrumIndependentBath: public SpectrumHierarchyUpdater {
 public:
  SpectrumIndependentBath(const PhiParameters* parameters,
                          Complex liouville_prefactor,
                          const Complex* const* timelocal,
                          const Complex* const* timelocal_adjoint):
      SpectrumHierarchyUpdater(parameters, liouville_prefactor),
      timelocal_(timelocal),
      timelocal_adjoint_(timelocal_adjoint) {
    temp_ = new Complex[parameters_->num_states];
  }
  ~SpectrumIndependentBath() {
    delete []temp_;
  }

 protected:
  const Complex* const* timelocal_;
  const Complex* const* timelocal_adjoint_;

  phi_size_t m_;
  Complex* temp_;

  void DoMatsubara(const HierarchyNode &node, Complex* drho);
  void DoTimelocal(const HierarchyNode &node, Complex* drho);
  void DoNext(const HierarchyNode &node, Complex* drho);
  void DoPrevious(const HierarchyNode &node, Complex* drho);
};

class SpectrumDiagonalBath: public SpectrumHierarchyUpdater {
 public:
  SpectrumDiagonalBath(const PhiParameters* parameters,
                       Complex liouville_prefactor,
                       const Complex* const* timelocal,
                       const Complex* const* timelocal_adjoint,
                       const Complex* same,
                       const Complex* next,
                       const Complex* prev):
      SpectrumHierarchyUpdater(parameters, liouville_prefactor),
      timelocal_(timelocal),
      timelocal_adjoint_(timelocal_adjoint),
      same_(same),
      next_(next),
      prev_(prev) {
    temp_ = new Complex[parameters_->num_states];
  }
  ~SpectrumDiagonalBath() {
    delete []temp_;
  }

 protected:
  const Complex* const* timelocal_;
  const Complex* const* timelocal_adjoint_;
  const Complex* same_;
  const Complex* next_;
  const Complex* prev_;

  phi_size_t m_, j_;
  const Complex* bath_;
  const Complex* density_;
  Complex* temp_;
  Complex prefactor_;

  void DoMatsubara(const HierarchyNode &node, Complex* drho);
  void DoTimelocal(const HierarchyNode &node, Complex* drho);
  void DoNext(const HierarchyNode &node, Complex* drho);
  void DoPrevious(const HierarchyNode &node, Complex* drho);
};

}  // namespace hierarchy_updater

#endif  // PHI_HIERARCHY_UPDATER_H_
