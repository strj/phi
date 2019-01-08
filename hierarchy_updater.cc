#include "hierarchy_updater.h"

#include "blas_wrapper.h"
#include "complex_matrix.h"
#include "hierarchy_node.h"
#include "numeric_types.h"

namespace hierarchy_updater {

void FullBath::DoMatsubara(const HierarchyNode &node, Complex* drho) {
  // -sum_ alpha_a [F_a, [F_a, rho]]
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    // sqrt(alpha_a) F_a
    bath_ = &(same_[m_ * num_elements_]);
    // temp = (sqrt(alpha_a) F_a) rho
    blas::Hemm(CblasRight, size_,
               &kOne, node.density_matrix, bath_, &kZero, temp_);
    // temp -= rho (sqrt(alpha_a) F_a)
    blas::Hemm(CblasLeft, size_,
               &kNegativeOne, node.density_matrix, bath_, &kOne, temp_);
    // drho -= (sqrt(alpha_a) F_a) temp
    blas::Hemm(CblasLeft, size_, &kNegativeOne, bath_, temp_, &kOne, drho);
    // drho += temp (sqrt(alpha_a) F_a)
    blas::Hemm(CblasRight, size_, &kOne, bath_, temp_, &kOne, drho);
  }
}

void FullBath::DoTimelocal(const HierarchyNode &node, Complex* drho) {
  // -sum_m [V_m (Q_m rho - rho Q_m^H) - (Q_m rho - rho Q_m^H) V_m]
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    // temp = -i (Q_m * rho)
    blas::Hemm(CblasRight, size_, &kNegativeImaginary,
               node.density_matrix, timelocal_[m_], &kZero, temp_);
    // temp += i rho * Q_m^H).
    blas::Hemm(CblasLeft, size_, &kImaginary,
               node.density_matrix, timelocal_adjoint_[m_], &kOne, temp_);
    // -i V_m
    bath_ = &(next_[m_ * num_elements_]);
    // drho += (-i V_m) temp
    blas::Hemm(CblasRight, size_, &kOne, temp_, bath_, &kOne, drho);
    // drho += - temp (-i V_m)
    blas::Hemm(CblasLeft, size_, &kNegativeOne, temp_, bath_, &kOne, drho);
  }
}

void FullBath::DoNext(const HierarchyNode &node, Complex* drho) {
  // -i sum_a sum_k [F_a, rho_n(ak)+]
  for (n_ = 0; n_ < node.num_next_hierarchy_nodes; ++n_) {
    m_ = node.next_coupling_op_index[n_];  // Runs from 0 to M.
    // rho_n(ak)+
    density_ = node.next_hierarchy_nodes[n_]->density_matrix;
    // -i F_a
    bath_ = &(next_[m_ * num_elements_]);
    // 1 or sqrt((n_ak + 1) |c_ak|)
    prefactor_ = kNegativeImaginary * node.next_prefactor[n_];
    // drho += (-i F_a) rho_n(ak)+
    blas::Hemm(CblasRight, size_, &prefactor_, density_, bath_, &kOne, drho);
    // -1 or -sqrt((n_ak + 1) |c_ak|)
    prefactor_ = kImaginary * node.next_prefactor[n_];
    // drho += -rho_n(ak)+ (-i F_a)
    blas::Hemm(CblasLeft, size_, &prefactor_, density_, bath_, &kOne, drho);
  }
}

void FullBath::DoPrevious(const HierarchyNode &node, Complex* drho) {
  // -i sum_a sum_k n_ak (c_ak F_a rho_n(ak)- - c_ak^* rho_n(ak)- F_a)
  for (n_ = 0; n_ < node.num_prev_hierarchy_nodes; ++n_) {
    m_ = node.prev_bath_coupling_op_index[n_];  // Runs from 0 to M.
    // rho_n(ak)-
    density_ = node.prev_hierarchy_nodes[n_]->density_matrix;
    // - F_a
    bath_ = &(prev_[m_ * num_elements_]);
    // i n_ak c_ak OR i c_ak sqrt(n_ak / |c_ak|)
    prefactor_ = node.prev_prefactor_row[n_];
    // drho += (i n_ak c_ak) (-F_a) rho_n(ak)-
    blas::Hemm(CblasRight, size_, &prefactor_, density_, bath_, &kOne, drho);
    // -i n_ak c_ak^* OR -i c_ak^* sqrt(n_ak / |c_ak|)
    prefactor_ = node.prev_prefactor_col[n_];
    // drho += (-i n_ak c_ak^*) rho_n(ak)- (-F_a)
    blas::Hemm(CblasLeft, size_, &prefactor_, density_, bath_, &kOne, drho);
  }
}

void DiagonalBath::DoMatsubara(const HierarchyNode &node, Complex* drho) {
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    bath_ = &(same_[m_ * num_elements_]);
    for (n_ = 0; n_ < num_elements_; ++n_) {
      drho[n_] -= bath_[n_] * node.density_matrix[n_];
    }
  }
}

void DiagonalBath::DoTimelocal(const HierarchyNode &node, Complex* drho) {
  // Note temp includes -i, and V_m includes -i
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    blas::Hemm(CblasRight, size_, &kNegativeImaginary,
               node.density_matrix, timelocal_[m_], &kZero, temp_);
    blas::Hemm(CblasLeft, size_, &kImaginary,
               node.density_matrix, timelocal_adjoint_[m_], &kOne, temp_);
    bath_ = &(next_[m_ * num_elements_]);
    for (n_ = 0; n_ < num_elements_; ++n_) {
      drho[n_] += bath_[n_] * temp_[n_];
    }
  }
}

void DiagonalBath::DoNext(const HierarchyNode &node, Complex* drho) {
  for (m_ = 0; m_ < node.num_next_hierarchy_nodes; ++m_) {  // Nnext runs from 0 to NsKt.
    density_ = node.next_hierarchy_nodes[m_]->density_matrix;
    bath_ = &(next_[node.next_coupling_op_index[m_] * num_elements_]);
    for (n_ = 0; n_ < num_elements_; ++n_) {
      drho[n_] += bath_[n_] * density_[n_];
    }
  }
}

void DiagonalBath::DoPrevious(const HierarchyNode &node, Complex* drho) {
  for (m_ = 0; m_ < node.num_prev_hierarchy_nodes; ++m_) {
    j_ = node.prev_node_index[m_];  // Runs from 0 to Ns * Kt - 1.
    density_ = node.prev_hierarchy_nodes[m_]->density_matrix;
    bath_ = &(prev_[j_ * num_elements_]);
    prefactor_ = Complex(node.index[j_]);
    for (n_ = 0; n_ < num_elements_; ++n_) {
      drho[n_] += prefactor_ * bath_[n_] * density_[n_];
    }
  }
}

void IndependentBath::DoMatsubara(const HierarchyNode &node, Complex* drho) {
  if (parameters_->is_multiple_independent_baths) {
    DoMultiBathMatsubara(node, drho);
    return;
  }
  for (m_ = 0; m_ < size_; ++m_) {
    update_ = &(drho[m_ * size_]);
    density_ = &(node.density_matrix[m_* size_]);
    for (n_ = 0; n_ < size_; ++n_) {
      update_[n_] -= density_[n_] *
          (node.matsubara_prefactor[n_] + node.matsubara_prefactor[m_]);
    }
    n_ = (size_ + 1) * m_;
    drho[n_] += kTwo * node.matsubara_prefactor[m_] * node.density_matrix[n_];
  }
}

void IndependentBath::DoMultiBathMatsubara(const HierarchyNode &node,
                                           Complex* drho) {
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    j_ = parameters_->multiple_independent_bath_indices[m_];  // j_ in range [0, size_).
    update_ = &(drho[size_ * j_]);
    density_ = &(node.density_matrix[size_ * j_]);
    for (n_ = 0; n_ < j_; ++n_) {
      update_[n_] -= node.matsubara_prefactor[m_] * density_[n_];
    }
    for (n_ = j_ + 1; n_ < size_; ++n_) {
      update_[n_] -= node.matsubara_prefactor[m_] * density_[n_];
    }
    update_ = &(node.density_matrix[size_ * j_]);
    density_ = &(drho[size_ * j_]);
    for (n_ = 0; n_ < j_; ++n_, update_ += size_, density_ += size_) {
      *update_ -= node.matsubara_prefactor[m_] * (*density_);
    }
    update_ += size_;
    density_ += size_;
    for (n_ = j_ + 1; n_ < size_;
         ++n_, update_ += size_, density_ += size_) {
      *update_ -= node.matsubara_prefactor[m_] * (*density_);
    }
  }
}

void IndependentBath::DoTimelocal(const HierarchyNode &node, Complex* drho) {
  if (parameters_->is_multiple_independent_baths) {
    DoMultiBathTimelocal(node, drho);
    return;
  }
  for (m_ = 0; m_ < size_; ++m_) {
    blas::Hemm(CblasRight, size_, &kOne,
               node.density_matrix, timelocal_[m_], &kZero, temp_);
    blas::Hemm(CblasLeft, size_, &kNegativeOne,
               node.density_matrix, timelocal_adjoint_[m_], &kOne, temp_);
    update_ = &(drho[m_ * size_]);
    density_ = &(temp_[m_ * size_]);
    for (n_ = 0; n_ < size_; ++n_, ++update_, ++density_) {
      *update_ += *density_;
    }
    update_ = &(drho[m_]);
    density_ = &(temp_[m_]);
    for (n_ = 0; n_ < size_; ++n_, update_ += size_, density_ += size_) {
      *update_ -= *density_;
    }
  }
}

void IndependentBath::DoMultiBathTimelocal(const HierarchyNode &node,
                                           Complex* drho) {
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    j_ = parameters_->multiple_independent_bath_indices[m_];
    blas::Hemm(CblasRight, size_, &kOne,
               node.density_matrix, timelocal_[m_], &kZero, temp_);
    blas::Hemm(CblasLeft, size_, &kNegativeOne,
               node.density_matrix, timelocal_adjoint_[m_], &kOne, temp_);
    update_ = &(drho[j_ * size_]);
    density_ = &(temp_[j_ * size_]);
    for (n_ = 0; n_ < size_; ++n_, ++update_, ++density_) {
      *update_ += *density_;
    }
    update_ = &(drho[j_]);
    density_ = &(temp_[j_]);
    for (n_ = 0; n_ < size_; ++n_, update_ += size_, density_ += size_) {
      *update_ -= *density_;
    }
  }
}

void IndependentBath::DoNext(const HierarchyNode &node, Complex* drho) {
  for (j_ = 0; j_ < node.num_next_hierarchy_nodes; ++j_) {  // This is sum over size_ & Kt.
    m_ = node.next_coupling_op_index[j_];
    update_ = &(drho[size_ * m_]);
    density_ = &(node.next_hierarchy_nodes[j_]->density_matrix[size_ * m_]);
    for (n_ = 0; n_ < size_; n_++) {
      update_[n_] += (node.next_prefactor[j_]) * density_[n_];
    }
  }
  for (n_ = 0; n_ < size_; ++n_) {
    update_ = &(drho[size_ * n_]);
    for (j_ = 0; j_ < node.num_next_hierarchy_nodes; ++j_) {
      m_ = node.next_coupling_op_index[j_];
      density_ = &(node.next_hierarchy_nodes[j_]->density_matrix[size_ * n_]);
      update_[m_] -= node.next_prefactor[j_] * density_[m_];
    }
  }
}

void IndependentBath::DoPrevious(const HierarchyNode &node, Complex* drho) {
  for (j_ = 0; j_ < node.num_prev_hierarchy_nodes; ++j_) {
    m_ = node.prev_bath_coupling_op_index[j_];
    update_ = &drho[size_ * m_];
    density_ = &(node.prev_hierarchy_nodes[j_]->density_matrix[size_ * m_]);
    for (n_ = 0; n_ < size_; n_++) {
      update_[n_] -= node.prev_prefactor_col[j_] * density_[n_];
    }
  }
  for (n_ = 0; n_ < size_; ++n_) {
    update_ = &drho[n_ * size_];
    for (j_ = 0; j_ < node.num_prev_hierarchy_nodes; ++j_) {
      // m_ runs from 0 to Ns
      // E.g. 0, 0, 0, 1, 1, 1, 2, 2, 2, ..., Ns, Ns, Ns for Kt = 2.
      m_ = node.prev_bath_coupling_op_index[j_];
      density_ = &(node.prev_hierarchy_nodes[j_]->density_matrix[n_ * size_]);
      update_[m_] -= node.prev_prefactor_row[j_] * density_[m_];
    }
  }
}

void IndependentBathAdaptiveTruncation::DoNext(const HierarchyNode &node,
                                               Complex* drho) {
  for (j_ = 0; j_ < node.num_next_hierarchy_nodes; ++j_) {  // This is sum over size_ & Kt.
    if (node.next_hierarchy_nodes[j_]->is_active) {
      m_ = node.next_coupling_op_index[j_];
      update_ = &(drho[size_ * m_]);
      density_ = &(node.next_hierarchy_nodes[j_]->density_matrix[size_ * m_]);
      for (n_ = 0; n_ < size_; n_++) {
        update_[n_] += (node.next_prefactor[j_]) * density_[n_];
      }
    }
  }
  for (n_ = 0; n_ < size_; ++n_) {
    update_ = &(drho[size_ * n_]);
    for (j_ = 0; j_ < node.num_next_hierarchy_nodes; ++j_) {
      if (node.next_hierarchy_nodes[j_]->is_active) {
        m_ = node.next_coupling_op_index[j_];
        density_ = &(node.next_hierarchy_nodes[j_]->density_matrix[size_ * n_]);
        update_[m_] -= node.next_prefactor[j_] * density_[m_];
      }
    }
  }
}

void IndependentBathAdaptiveTruncation::DoPrevious(const HierarchyNode &node,
                                                   Complex* drho) {
  for (j_ = 0; j_ < node.num_prev_hierarchy_nodes; ++j_) {
    if (node.prev_hierarchy_nodes[j_]->is_active) {
      m_ = node.prev_bath_coupling_op_index[j_];
      update_ = &drho[size_ * m_];
      density_ = &(node.prev_hierarchy_nodes[j_]->density_matrix[size_ * m_]);
      for (n_ = 0; n_ < size_; n_++) {
        update_[n_] -= node.prev_prefactor_col[j_] * density_[n_];
      }
    }
  }
  for (n_ = 0; n_ < size_; ++n_) {
    update_ = &drho[n_ * size_];
    for (j_ = 0; j_ < node.num_prev_hierarchy_nodes; ++j_) {
      if (node.prev_hierarchy_nodes[j_]->is_active) {
        // m_ runs from 0 to Ns
        // E.g. 0, 0, 0, 1, 1, 1, 2, 2, 2, ..., Ns, Ns, Ns for Kt = 2.
        m_ = node.prev_bath_coupling_op_index[j_];
        density_ = &(node.prev_hierarchy_nodes[j_]->density_matrix[n_ * size_]);
        update_[m_] -= node.prev_prefactor_row[j_] * density_[m_];
      }
    }
  }
}

void SpectrumIndependentBath::DoMatsubara(const HierarchyNode &node,
                                          Complex* drho) {
  if (!parameters_->is_multiple_independent_baths) {
    for (n_ = 0; n_ < size_; ++n_) {
      drho[n_] -= node.matsubara_prefactor[n_] * node.density_matrix[n_];
    }
  } else {
    for (n_ = 0; n_ < num_bath_operators_; ++n_) {
      m_ = parameters_->multiple_independent_bath_indices[n_];
      drho[m_] -= node.matsubara_prefactor[n_] * node.density_matrix[m_];
    }
  }
}

void SpectrumIndependentBath::DoTimelocal(const HierarchyNode &node,
                                          Complex* drho) {
  if (!parameters_->is_multiple_independent_baths) {
    if (!parameters_->is_restarted) {
      for (n_ = 0; n_ < size_; ++n_) {
        blas::Gemv(size_, &kOne,
                   timelocal_[n_], node.density_matrix, &kZero, temp_);
        drho[n_] -= temp_[n_];
      }
    } else {
      for (n_ = 0; n_ < size_; ++n_) {
        blas::Gemv(size_, &kOne,
                   timelocal_adjoint_[n_], node.density_matrix, &kZero, temp_);
        drho[n_] -= temp_[n_];
      }
    }
  } else {
    if (!parameters_->is_restarted) {
      for (m_ = 0; m_ < num_bath_operators_; ++m_) {
        blas::Gemv(size_, &kOne,
                   timelocal_[m_], node.density_matrix, &kZero, temp_);
        n_ = parameters_->multiple_independent_bath_indices[m_];
        drho[n_] -= temp_[n_];
      }
    } else {
      for (m_ = 0; m_ < num_bath_operators_; ++m_) {
        n_ = parameters_->multiple_independent_bath_indices[m_];
        blas::Gemv(size_, &kOne,
                   timelocal_adjoint_[m_], node.density_matrix, &kZero, temp_);
        drho[n_] -= temp_[n_];
      }
    }
  }
}

void SpectrumIndependentBath::DoNext(const HierarchyNode &node,
                                     Complex* drho) {
  for (n_ = 0; n_ < node.num_next_hierarchy_nodes; ++n_) {  // This is sum over M & Kt.
    m_ = node.next_coupling_op_index[n_];
    drho[m_] -= node.next_prefactor[n_] * node.next_hierarchy_nodes[n_]->density_matrix[m_];
  }
}

void SpectrumIndependentBath::DoPrevious(const HierarchyNode &node,
                                         Complex* drho) {
  for (n_ = 0; n_ < node.num_prev_hierarchy_nodes; ++n_) {
    m_ = node.prev_bath_coupling_op_index[n_];
    drho[m_] -= node.prev_prefactor_row[n_] * node.prev_hierarchy_nodes[n_]->density_matrix[m_];
  }
}

void SpectrumDiagonalBath::DoMatsubara(const HierarchyNode &node,
                                          Complex* drho) {
  for (m_ = 0; m_ < size_; ++m_) {
    bath_ = &(same_[m_ * size_]);
    for (n_ = 0; n_ < size_; ++n_) {
      drho[n_] -= bath_[n_] * node.density_matrix[n_];
    }
  }
}

/*
 * NOTE: Does not correctly account for emission spectrum.
 */
void SpectrumDiagonalBath::DoTimelocal(const HierarchyNode &node,
                                       Complex* drho) {
  //- sum_m [ V_m Q_m rho]
  // Note Qrho_rhoQ includes no -i, and V_m includes -i*-i (different to
  // no TL_truncation)
  for (m_ = 0; m_ < num_bath_operators_; ++m_) {
    blas::Gemv(size_, &kNegativeImaginary,
               timelocal_[m_], node.density_matrix, &kZero, temp_);
    bath_ = &(next_[m_ * size_]);
    for (n_ = 0; n_ < size_; ++n_) {
      drho[n_] += bath_[n_] * temp_[n_];
    }
  }
}

void SpectrumDiagonalBath::DoNext(const HierarchyNode &node,
                                  Complex* drho) {
  for (m_ = 0; m_ < node.num_next_hierarchy_nodes; ++m_) {  // Nnext runs from 0 to NsKt.
    density_ = node.next_hierarchy_nodes[m_]->density_matrix;
    bath_ = &(next_[node.next_coupling_op_index[m_] * size_]);
    for (n_ = 0; n_ < size_; ++n_) {
      drho[n_] += bath_[n_] * density_[n_];
    }
  }
}

void SpectrumDiagonalBath::DoPrevious(const HierarchyNode &node,
                                      Complex* drho) {
  for (m_ = 0; m_ < node.num_prev_hierarchy_nodes; ++m_) {
    j_ = node.prev_node_index[m_];  // Runs from 0 to Ns * Kt - 1.
    density_ = node.prev_hierarchy_nodes[m_]->density_matrix;
    bath_ = &(prev_[j_ * size_]);
    prefactor_ = Complex(node.index[j_]);
    for (n_ = 0; n_ < size_; ++n_) {
      drho[n_] += prefactor_ * bath_[n_] * density_[n_];
    }
  }
}


}  // namespace hierarchy_updater
