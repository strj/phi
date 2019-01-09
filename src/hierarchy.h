#ifndef PHI_HIERARCHY_H_
#define PHI_HIERARCHY_H_
#include <map>

#include "hierarchy_node.h"
#include "numeric_types.h"
#include "phi_parameters.h"

class Hierarchy {
 public:
  Hierarchy();
  explicit Hierarchy(PhiParameters *p);
  ~Hierarchy();

 protected:
  Complex *identity_matrix_;
  PhiParameters* parameters_;
  int verbose_;

  const phi_big_size_t num_states_;
  phi_big_size_t num_elements_;
  phi_size_t matsubara_truncation_;
  phi_size_t num_bath_coupling_terms_;
  phi_size_t num_correlation_fn_terms_;

  phi_big_size_t matrix_count_;
  phi_size_t *hierarchy_level_;
  phi_big_size_t *matrix_count_by_level_;

  HierarchyNode *nodes_, *nodes_tmp_;
  Complex **bath_operators_;
  Complex **full_bath_operators_;

  map<int, MultiIndex> multi_index_map_;
  map<MultiIndex, int, CompareMultiIndex> multi_index_map_lookup_;

  Float thermal_energy_;

  Complex *correlation_fn_constants_;
  Float *abs_correlation_fn_constants_;

  bool should_assign_density_matrix_memory_;

  void ConstructI();
  void FillDiagonalBathOperators(Complex *,
                                 Complex *,
                                 Complex *) const;
  void FillFullBathOperators(Complex *,
                             Complex *,
                             Complex *) const;

  void FillBathOperators(Complex *,
                         Complex *,
                         Complex *) const;

  void FillPropagator(const Float, const Complex *, Complex *, Complex *) const;

  void AddFullBathOperator(const Complex *,
                           const Complex *,
                           Complex *) const;
  void AddDiagonalBathOperator(const Complex *,
                               const Complex *,
                               Complex *) const;
  void AddIndependentBathOperator(const Complex *,
                                  const Complex *,
                                  Complex *) const;

  void UpdateTimeLocalTruncationMatrices(const Float,
                                         const int,
                                         const Complex *,
                                         const Complex *,
                                         Complex **,
                                         Complex *,
                                         Complex *) const;
};
#endif  // PHI_HIERARCHY_H_
