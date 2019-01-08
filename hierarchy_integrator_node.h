#ifndef PHI_HIERARCHY_NODE_H_
#define PHI_HIERARCHY_NODE_H_
#include <string.h>   // for memcpy
#include <sstream>
#include "complex_matrix.h"
using namespace std;

const double PI = 3.14159265358979323846264338327950288419716939937510;

struct MultiIndex{
  int *index_array;
  int size;
  int sum;
  int thread;
  int weight;
  MultiIndex(){
    index_array = 0;
    size = 0;
    sum = 0;
    weight = 0;
    thread = -1;
  }
  explicit MultiIndex(int m) {
    size = m;
    index_array = new int[m];
    for (int i = 0; i < size; i++) {
      index_array[i] = 0;
    }
    sum = 0;
    weight = 0;
    thread = -1;
  }
  void Create(int m){
    size = m;
    if (index_array == NULL)
      delete index_array;
    index_array = new int[m];
    for (int i = 0; i < size; i++) {
      index_array[i] = 0;
    }
    sum = 0;
    weight = 0;
    thread = -1;
  }
  void Erase(){
    if (size != 0 || index_array == NULL) {
      delete []index_array;
    }
    index_array = new int[size];
    for (int i = 0; i < size; i++) {
      index_array[i] = 0;
    }
    sum = 0;
    weight = 0;
  }
  int operator[](int n) {
    return index_array[n];
  }
  MultiIndex &operator=(const MultiIndex &rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (size != rhs.size) {
      if (size != 0) {
        delete []index_array;
      } else if (index_array == NULL) {
        delete index_array;
      }
      index_array = new int[size];
      size = rhs.size;
    }
    for (int i = 0; i < rhs.size; i++)
      index_array[i] = rhs.index_array[i];
    sum = rhs.sum;
    weight = rhs.weight;
    thread = rhs.thread;
    return *this;
  }
  std::string GetString() const {
    ostringstream ss;
    for (int i = 0; i < size - 1; i++) {
      ss << index_array[i] << ",";
    }
    ss << index_array[size - 1] << "\n";
    return ss.str();
  }
  void Inc(int n) {
    if (n < 0) {
      index_array[- n + 1] -= 1;
      sum -= 1;
    } else if (n > 0) {
      index_array[n - 1] += 1;
      sum += 1;
    }
  }
  void Dec(int n){
    if (n < 0) {
      index_array[- n + 1] += 1;
      sum += 1;
    } else if (n > 0) {
      index_array[n - 1] -= 1;
      sum -= 1;
    }
  }
  void UpdateWeight(){
    weight = 0;
    for (int i = 0; i < size; ++i) {
      if (index_array[i] > weight) {
        weight = index_array[i];
      }
    }
  }
};

struct CompareMultiIndex {
  bool operator()(const MultiIndex &index_1, const MultiIndex &index_2) const {
    bool res = false;
    if (index_1.size > 0 || index_2.size > 0) {
      if (index_1.size > index_2.size) {
        res = false;
      } else if (index_1.size < index_2.size) {
        res = true;
      } else if (index_1.sum < index_2.sum) {
        res = true;
      } else if (index_1.sum > index_2.sum) {
        res = false;
      } else if (index_1.index_array != NULL && index_2.index_array != NULL) {
        for (int i =0; i < index_1.size; i++) {
          if (index_1.index_array[i] != index_2.index_array[i]){
            res = (index_1.index_array[i] < index_2.index_array[i]);
            return res;
          }
        }
      }
    }
    return res;
  }
};

struct HierarchyNode {
  HierarchyNode();

  int *index;
  int hierarchy_truncation_level;
  int num_states;
  int num_bath_couplings;
  int num_matsubara_terms;
  int id;

  bool is_active;

  Complex dephasing_prefactor;
  Complex* matsubara_prefactor;
  Complex* prev_prefactor_row;
  Complex* prev_prefactor_col;
  Complex* next_prefactor;

  HierarchyNode **next_hierarchy_nodes;
  HierarchyNode **prev_hierarchy_nodes;
  // Entries are 0...num_bath_couplings * num_matsubara_terms
  int *prev_node_index;
  // Entries are 0...num_bath_couplings
  int *prev_bath_coupling_op_index;
  // Entries are 0...num_bath_couplings * num_matsubara_terms
  int *next_node_index;
  // Entries are 0...num_bath_couplings
  int *next_coupling_op_index;
  int num_prev_hierarchy_nodes;
  int num_next_hierarchy_nodes;
  int num_elements;

  bool is_timelocal_truncated;

  Complex *density_matrix;
  void Create(int n, int m, int k, int* in, Complex* s);
  void CreateVec(int n, int m, int k, int* in, Complex* s);
  std::string GetIndexString();

  // Integration Matrices.
  Complex *k0, *k1, *k2, *k3, *k4, *k5;
  Complex *hamiltonian, *hamiltonian_adjoint;

  // Steady-state Liouville-space Matrices.
  int* matsubara_node_index;
  Complex **next_liouville;
  int** next_liouville_index;
  Complex **prev_liouville;
  int** prev_liouville_index;
  Complex *same_liouville;
  void CreateSameLiouvilleOperator(
      const Complex* liouville_hamiltonian,
      const int* diag_num,
      const Index2D* diagonal_block,
      const int num_entries);
  void CreateNextLiouvilleOperator(const int num_entries);
  void CreatePrevLiouvilleOperator(const int num_entries);

  // Steady-state bicgstab.
  Complex *r;
  Complex *r0;
  Complex *v;
  Complex *p;
  Complex *t;
  Complex *steady_state_density_matrix;
  void BicgstabInit();
  void BicgstablInit(int l);
  void DetachDensityMatrix();
};

#endif  // PHI_HIERARCHY_NODE_H_
