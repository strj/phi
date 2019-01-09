#include "hierarchy_node.h"

#include <string>

HierarchyNode::HierarchyNode() {
  num_states = 0;
  num_elements = 0;
  num_bath_couplings = 0;
  num_matsubara_terms = 0;
  hierarchy_truncation_level = 0;
  next_hierarchy_nodes = NULL;
  prev_hierarchy_nodes = NULL;
  index = NULL;
  num_prev_hierarchy_nodes = 0;
  num_next_hierarchy_nodes = 0;
  is_active = true;
  is_timelocal_truncated = false;
  dephasing_prefactor = 0;
  id = 0;
  k0 = k1 = k2 = k3 = k4 = k5 = NULL;
  density_matrix = NULL;
  matsubara_prefactor = NULL;
  prev_prefactor_row = NULL;
  prev_prefactor_col = NULL;
  next_prefactor = NULL;
  prev_node_index = NULL;
  prev_bath_coupling_op_index = NULL;
  next_node_index = NULL;
  next_coupling_op_index = NULL;
  matsubara_node_index = NULL;
  hamiltonian = NULL;
  hamiltonian_adjoint = NULL;
  prev_liouville_index = NULL;
  next_liouville_index = NULL;
  same_liouville = NULL;
  prev_liouville = NULL;
  next_liouville = NULL;
  steady_state_density_matrix = NULL;
  r = r0 = v = p = t = NULL;
}

void HierarchyNode::Create(int n, int m, int k, int* in, Complex* s) {
  num_states = n;
  num_matsubara_terms = k;
  num_bath_couplings = m;
  hierarchy_truncation_level = 0;
  dephasing_prefactor = 0;
  is_active = true;
  if (index != NULL){
    delete []index;
  }
  index = new int[num_bath_couplings * num_matsubara_terms];
  for (int i = 0; i < num_bath_couplings * num_matsubara_terms; i++) {
    index[i] = in[i];
    hierarchy_truncation_level += index[i];
  }
  matsubara_node_index = new int[num_bath_couplings];
  k0 = k1 = k2 = k3 = k4 = k5 = NULL;
}

void HierarchyNode::CreateVec(int n, int m, int k, int *In, Complex * s) {
  num_states = n;
  num_matsubara_terms = k;
  num_bath_couplings = m;
  hierarchy_truncation_level = 0;
  if (index != NULL) {
    delete []index;
  }
  index = new int[num_bath_couplings * num_matsubara_terms];
  for (int i = 0; i < num_bath_couplings * num_matsubara_terms; i++) {
    index[i] = In[i];
    hierarchy_truncation_level += index[i];
  }
  matsubara_node_index = new int[num_bath_couplings];
  k0 = k1 = k2 = k3 = k4 = k5 = NULL;
}

std::string HierarchyNode::GetIndexString() {
  ostringstream ss;
  for (int i = 0; i < num_bath_couplings * num_matsubara_terms; i++) {
    ss << index[i] << " ";
  }
  return ss.str();
}

void HierarchyNode::CreateSameLiouvilleOperator(
    const Complex* liouville_hamiltonian,
    const int* diag_num,
    const Index2D* diagonal_block,
    const int num_entries) {
  same_liouville = new Complex[num_entries];
  int n;
  memcpy(same_liouville, liouville_hamiltonian, num_entries * sizeof(Complex));
  for (int i = 0; i < num_states*num_states; ++i){
    n = diag_num[i];
    same_liouville[n] -= dephasing_prefactor;
    for (int m = 0; m < num_bath_couplings; ++m){
      if (diagonal_block[n].i != matsubara_node_index[m] &&
          diagonal_block[n].j == matsubara_node_index[m])
        same_liouville[n] -= matsubara_prefactor[m];
      if (diagonal_block[n].j != matsubara_node_index[m] &&
          diagonal_block[n].i == matsubara_node_index[m])
        same_liouville[n] -= matsubara_prefactor[m];
    }
  }
}

void HierarchyNode::CreateNextLiouvilleOperator(const int num_entries) {
  next_liouville = new Complex*[num_next_hierarchy_nodes];
  int m = 0;
  int n = 0;
  for (int nj = 0; nj < num_next_hierarchy_nodes; ++nj) {
    n = 0;
    m = matsubara_node_index[next_coupling_op_index[nj]];
    next_liouville[nj] = new Complex[num_entries];
    for (int i = 0; i < m; ++i) {
      next_liouville[nj][n++] = +next_prefactor[nj];
    }
    for (int i = 0 ; i < m; ++i){
      next_liouville[nj][n++] = -next_prefactor[nj];
    }
    for (int i = m+1 ; i < num_states; ++i){
      next_liouville[nj][n++] = -next_prefactor[nj];
    }
    for (int i = m+1; i < num_states; ++i) {
      next_liouville[nj][n++] = +next_prefactor[nj];
    }
  }
}

void HierarchyNode::CreatePrevLiouvilleOperator(const int num_entries) {
  prev_liouville = new Complex*[num_prev_hierarchy_nodes];
  int m = 0;
  int n = 0;
  for (int nj = 0; nj < num_prev_hierarchy_nodes; ++nj) {
    n = 0;
    m = matsubara_node_index[prev_bath_coupling_op_index[nj]];
    prev_liouville[nj] = new Complex[num_entries];

    for (int i = 0; i < m; ++i) {
      prev_liouville[nj][n++] = prev_prefactor_col[nj];
    }
    for (int i = 0; i < m; ++i) {
      prev_liouville[nj][n++] = -prev_prefactor_row[nj];
    }
    prev_liouville[nj][n++] = prev_prefactor_col[nj]-prev_prefactor_row[nj];
    for (int i = m+1; i < num_states; ++i) {
      prev_liouville[nj][n++] = -prev_prefactor_row[nj];
    }
    for (int i = m+1; i < num_states; ++i) {
      prev_liouville[nj][n++] = prev_prefactor_col[nj];
    }
  }
}

void HierarchyNode::DetachDensityMatrix() {
  density_matrix = NULL;
}

void HierarchyNode::BicgstabInit(){
  r = new Complex[num_states * num_states];
  r0 = new Complex[num_states * num_states];
  v = new Complex[num_states * num_states];
  p = new Complex[num_states * num_states];
  t = new Complex[num_states * num_states];
  steady_state_density_matrix = new Complex [num_states * num_states];
  for (int i = 0; i < num_states * num_states; ++i) {
    r[i] = 0;
    r0[i] =0;
    v[i] = 0;
    p[i] = 0;
    t[i] = 0;
    steady_state_density_matrix[i] = 0;
  }
}

void HierarchyNode::BicgstablInit(int l) {
  r = new Complex[(l+1) * num_states * num_states];
  v = new Complex[(l+1) * num_states * num_states];
  r0 = new Complex[num_states * num_states];
  steady_state_density_matrix = new Complex [num_states * num_states];
  for (int i = 0; i < num_states * num_states; ++i) {
    for (int k = 0; k <= l; ++k) {
      r[i * (l + 1) + k] = 0;
      v[i * (l + 1) + k] = 0;
    }
    r0[i] = 0;
    steady_state_density_matrix[i] = 0;
  }
}
