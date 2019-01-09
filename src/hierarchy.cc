#include "hierarchy_integrator.h"

#include "complex_matrix.h"

Hierarchy::Hierarchy():
  parameters_(NULL),
  verbose_(1),
  num_states_(0),
  num_elements_(0),
  matsubara_truncation_(0),
  num_bath_coupling_terms_(0),
  num_correlation_fn_terms_(0),
  matrix_count_(0),
  hierarchy_level_(NULL),
  matrix_count_by_level_(NULL),
  nodes_(NULL),
  nodes_tmp_(NULL),
  bath_operators_(NULL),
  full_bath_operators_(NULL) {
}

Hierarchy::Hierarchy(PhiParameters *p):
  parameters_(p),
  verbose_(p->verbose),
  num_states_(p->num_states),
  num_elements_(p->num_states * p->num_states),
  matsubara_truncation_(p->num_matsubara_terms),
  num_bath_coupling_terms_(p->num_bath_couplings),
  num_correlation_fn_terms_(p->num_bath_couplings * p->num_matsubara_terms),
  matrix_count_(0),
  hierarchy_level_(NULL),
  matrix_count_by_level_(NULL),
  nodes_(NULL),
  nodes_tmp_(NULL),
  bath_operators_(NULL),
  full_bath_operators_(NULL)  {

  identity_matrix_ = new Complex[num_elements_];
  SetElementsToZero(num_elements_, identity_matrix_);
  for (int i = 0; i < num_states_; ++i) {
    identity_matrix_[i * (num_states_ + 1)] = kOne;
  }
}

void Hierarchy::ConstructI() {
  if (verbose_ > 0) {
    std::cout << "Constructing Index graph.\n";
  }

  const phi_size_t truncation_level = parameters_->hierarchy_truncation_level;

  // Dummy indices to populate I[.].
  MultiIndex index_0(num_correlation_fn_terms_);
  MultiIndex index_1(num_correlation_fn_terms_);

  hierarchy_level_ = new phi_size_t[truncation_level];
  matrix_count_by_level_ = new phi_big_size_t[truncation_level];
  for (phi_size_t i = 0; i < truncation_level; i++) {
    hierarchy_level_[i] = 0;
  }

  hierarchy_level_[0] = 1;
  matrix_count_by_level_[0] = 1;

  // Create 0th hierarchy tier, containing just the system density matrix.
  multi_index_map_lookup_[index_0] = 0;
  multi_index_map_[0].Create(num_correlation_fn_terms_);
  multi_index_map_[0] = index_0;
  if (verbose_ > 1) {
    std::cout << "\nHierarchy Level: " << 0 << "\n";
    if (verbose_ > 2) {
      std::cout << "" << matrix_count_ << ": "
           << multi_index_map_[0].GetString();
    }
  }
  matrix_count_++;
  // Create 1st hierarchy tier.
  if (truncation_level >= 2) {
    if (verbose_ > 1) std::cout << "\nHierarchy Level: " << 1 << "\n";
    hierarchy_level_[1] = num_correlation_fn_terms_;
    matrix_count_by_level_[1] = 1 + num_correlation_fn_terms_;
    for (int i = 0; i < num_correlation_fn_terms_; ++i) {
      const phi_big_size_t node_index = i + 1;
      multi_index_map_[node_index].Create(num_correlation_fn_terms_);
      multi_index_map_[node_index] = multi_index_map_[0];
      multi_index_map_[node_index].Inc(node_index);
      multi_index_map_[node_index].UpdateWeight();
      multi_index_map_lookup_[multi_index_map_[node_index]] = node_index;
      if (verbose_ > 2) {
        std::cout << "" << matrix_count_ << ": "
             << multi_index_map_[node_index].GetString();
      }
      matrix_count_++;
    }
     if (verbose_ > 1) std::cout << "\t" << hierarchy_level_[1] << " Matrices\n";
  }

  // FOLLOWING ORDERS
  if (truncation_level > 2) {
    int index_prev_n = 0;
    for (int level = 2; level < truncation_level; ++level) {
      if (verbose_ > 1) {
        std::cout << "\nHierarchy Level: " << level << "\n";
      }
      // For each position.
      for (int k = 0; k < num_correlation_fn_terms_; k++) {
        // For each entry in the previous level.
        for (int n = 0; n < hierarchy_level_[level - 1]; ++n) {
          index_prev_n = matrix_count_ - hierarchy_level_[level - 1] + n;
          // Get previous index.
          index_1 = multi_index_map_[index_prev_n];
          // UPdate I[n+k] by 1.
          index_1.Inc((n + k) % num_correlation_fn_terms_ + 1);
          index_1.UpdateWeight();
          // Check for same Index in the current level of the hierarchy and if
          // not found, insert it.
          if (multi_index_map_lookup_.count(index_1) == 0) {
            const phi_big_size_t node_index =
                matrix_count_ + hierarchy_level_[level];
            multi_index_map_[node_index].Create(num_correlation_fn_terms_);
            multi_index_map_[node_index] = multi_index_map_[index_prev_n];
            multi_index_map_[node_index].Inc((n + k) %
                                             num_correlation_fn_terms_ + 1);
            multi_index_map_[node_index].UpdateWeight();
            if (verbose_ > 2) {
              std::cout << "" << node_index << ": ";
            }
            multi_index_map_lookup_[multi_index_map_[node_index]] = node_index;
            if (verbose_ > 2) {
              std::cout << multi_index_map_[node_index].GetString();
            }
            ++hierarchy_level_[level];
          }
        }
      }
      if (verbose_ > 1) {
        std::cout << "\t" << hierarchy_level_[level] << " Matrices\n";
      }
      matrix_count_ += hierarchy_level_[level];
      matrix_count_by_level_[level] = matrix_count_by_level_[level - 1] +
                                      hierarchy_level_[level];
    }
  }
  // The above looping will create, e.g., a 2-hierarchy with numbering:
  //               01
  //             02  03
  //           04  06  05
  //         07  09  10  08
  //       11  13  15  14  12
  //     16  18  20  21  19  17
  //   22  24  26  28  27  25  23
  if (verbose_ > 0) {
    std::cout << "\nTotal: " <<  matrix_count_ << " matrices\n\n";
  }
}

void Hierarchy::FillDiagonalBathOperators(Complex *same,
                                          Complex *next,
                                          Complex *prev) const {
  const bool is_spectrum_calculation = parameters_->is_spectrum_calculation;
  const Complex *matsubara_factor = nodes_[0].matsubara_prefactor;
  for (int m = 0; m < num_bath_coupling_terms_; ++m) { 
    if (!is_spectrum_calculation) {
      for (int i = 0; i < num_states_; ++i) {
        for (int j = 0; j < num_states_; ++j) {
          next[m * num_elements_ + i * num_states_ + j] = kNegativeImaginary *
              (bath_operators_[m][j] - bath_operators_[m][i]);

          same[m * num_elements_ + i * num_states_ + j] = matsubara_factor[m] *
              pow(bath_operators_[m][j] - bath_operators_[m][i], 2);
        }
      }
      for (int k = 0; k < matsubara_truncation_; ++k) {
        for (int i = 0; i < num_states_; ++i) {
          for (int j = 0; j < num_states_; ++j) {
            const int prev_idx = m * matsubara_truncation_ * num_elements_ +
                                 k * num_elements_ + i * num_states_ + j;
            const int op_idx = m * matsubara_truncation_ + k;
            prev[prev_idx] = kNegativeImaginary *
                (bath_operators_[m][j] * correlation_fn_constants_[op_idx] -
                 bath_operators_[m][i] * conj(correlation_fn_constants_[op_idx]));
          }
        }
      }
    } else {
      for (int i = 0; i < num_states_; ++i) {
          next[m * num_states_ + i] = kNegativeImaginary *
                                      bath_operators_[m][i];
          same[m * num_states_ + i] = matsubara_factor[m] *
                                      pow(bath_operators_[m][i], 2);
      }
      for (int k = 0; k < matsubara_truncation_; ++k) {
        for (int i = 0; i < num_states_; ++i) {
          prev[m * matsubara_truncation_ * num_states_ + k * num_states_ + i] =
              kNegativeImaginary * bath_operators_[m][i] *
              correlation_fn_constants_[m * matsubara_truncation_ + k];
        }
      }
    }
  }
}

void Hierarchy::FillFullBathOperators(Complex *same,
                                      Complex *next,
                                      Complex *prev) const {
  const Complex *matsubara_factor = nodes_[0].matsubara_prefactor;
  // Loop over coupling index.
  for (int m = 0; m < num_bath_coupling_terms_; ++m) {   
    for (int i = 0; i < num_elements_; ++i) {
      same[m * num_elements_ + i] = sqrt(matsubara_factor[m]) *
                                    bath_operators_[m][i];
      next[m * num_elements_ + i] = kNegativeImaginary * bath_operators_[m][i];
      prev[m * num_elements_ + i] = -Complex(bath_operators_[m][i]);
    }
  }
}

void Hierarchy::FillBathOperators(Complex *same,
                                  Complex *next,
                                  Complex *prev) const {
  if (parameters_->is_full_bath_coupling) {
    FillFullBathOperators(same, next, prev);
  }
  else if (parameters_->is_diagonal_bath_coupling) {
    FillDiagonalBathOperators(same, next, prev);
  }
}

void Hierarchy::FillPropagator(const Float timestep,
                               const Complex *hamiltonian,
                               Complex *propagator,
                               Complex *propagator_inverse) const {
  for (int i = 0; i < num_elements_; ++i) {
    propagator[i] =
        kNegativeImaginary * hamiltonian[i] * timestep + identity_matrix_[i];
  }
  ConjugateTranspose(num_states_, propagator, propagator_inverse);
}

void Hierarchy::AddFullBathOperator(const Complex *factor,
                                    const Complex *bath_operator,
                                    Complex *matrix) const {
  blas::Axpy(num_elements_, factor, bath_operator, 1, matrix, 1);
}

void Hierarchy::AddDiagonalBathOperator(const Complex *factor,
                                        const Complex *bath_operator,
                                        Complex *matrix) const {
  for (phi_size_t i = 0; i < num_states_; ++i) {
    matrix[i * num_states_ + i] += (*factor) * bath_operator[i];
  }
}

void Hierarchy::AddIndependentBathOperator(const Complex *factor,
                                           const Complex *bath_operator,
                                           Complex *element) const {
  *element += (*factor);
}

/*
 * Calculating the time local truncation matrices.
 * TODO: lots of optimizations.
 */
void Hierarchy::UpdateTimeLocalTruncationMatrices(
    const Float timestep, const int coupling_index,
    const Complex *propagator, const Complex *propagator_inverse,
    Complex **time_local_truncation_matrix,
    Complex * summed_truncation_matrix,
    Complex * summed_truncation_matrix_dagger) const {

  // TODO(johanstr): Objectify the Time Local truncation and factor out the time
  // independent parts of the calculation.
  if (timestep < 1e-10) {
    // Assume that we're at pretty much the same time and the time local
    // truncation matrices don't need to be updated - this can be dangerous if
    // called multiple times.
    return;
  }

  const Float *gamma = parameters_->gamma;
  const Float hbar = parameters_->hbar;
  Complex * temp_matrix;
  temp_matrix = new Complex[num_elements_];

  Complex correlation_factor = timestep * 
      correlation_fn_constants_[coupling_index * matsubara_truncation_];
  Complex exponential_damping = Complex(exp(-gamma[coupling_index] * timestep));
  if (parameters_->is_diagonal_bath_coupling) {
    AddDiagonalBathOperator(&correlation_factor,
                            bath_operators_[coupling_index],
                            time_local_truncation_matrix[0]);
  } else if (parameters_->is_full_bath_coupling) {
    AddFullBathOperator(&correlation_factor,
                        bath_operators_[coupling_index],
                        time_local_truncation_matrix[0]);
  } else {
    int i = coupling_index * num_states_ + coupling_index;
    AddIndependentBathOperator(&correlation_factor,
                               NULL,  // Bath operators not needed.
                               &(time_local_truncation_matrix[0][i]));
  }
  blas::Gemm(num_states_, &exponential_damping, propagator_inverse,
             time_local_truncation_matrix[0], &kZero, temp_matrix);

  blas::Gemm(num_states_, &kOne, temp_matrix, propagator, &kZero,
             time_local_truncation_matrix[0]);

  const Float thermal_energy =
      parameters_->boltzmann_constant * parameters_->temperature;
  const Float nu = 2 * PI * thermal_energy / hbar;
  for (int k = 1; k < matsubara_truncation_; ++k) {
    correlation_factor = timestep *
        correlation_fn_constants_[coupling_index * matsubara_truncation_ + k];
    exponential_damping = exp(-nu * k * timestep);
    if (parameters_->is_diagonal_bath_coupling) {
      AddDiagonalBathOperator(&correlation_factor,
                              bath_operators_[coupling_index],
                              time_local_truncation_matrix[k]);
    } else if (parameters_->is_full_bath_coupling) {
      AddFullBathOperator(&correlation_factor,
                          bath_operators_[coupling_index],
                          time_local_truncation_matrix[k]);
    } else {
      int i = coupling_index * num_states_ + coupling_index;
      AddIndependentBathOperator(&correlation_factor,
                                 NULL,  // Bath operators not needed.
                                 &(time_local_truncation_matrix[k][i]));
    }
    blas::Gemm(num_states_, &exponential_damping, propagator_inverse,
               time_local_truncation_matrix[k], &kZero, temp_matrix);

    blas::Gemm(num_states_, &kOne, temp_matrix, propagator, &kZero,
               time_local_truncation_matrix[k]);
  }
  SumMatrices(time_local_truncation_matrix,
              matsubara_truncation_, num_elements_, summed_truncation_matrix);
  ConjugateTranspose(
      num_states_, summed_truncation_matrix, summed_truncation_matrix_dagger);

  delete []temp_matrix;
}

Hierarchy::~Hierarchy() {
  delete []nodes_;
  delete []hierarchy_level_;
}
