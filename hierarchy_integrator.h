#ifndef PHI_HIERARCHY_INTEGRATOR_H_
#define PHI_HIERARCHY_INTEGRATOR_H_

#include <pthread.h>
#include <sys/time.h>

#include <fstream>
#include <string>

#include "barrier.h"
#include "complex_matrix.h"
#include "cubic_spline.h"
#include "hierarchy.h"
#include "hierarchy_node.h"
#include "hierarchy_updater.h"
#include "hilbert_space_truncator.h"
#include "numeric_types.h"
#include "phi_parameters.h"
#include "restart_helper.h"

#ifndef NDEBUG
#define NDEBUG 1
#endif

class HierarchyIntegrator : public Hierarchy {
 public:
  HierarchyIntegrator(int, char *[]);
  explicit HierarchyIntegrator(PhiParameters *p);
  HierarchyIntegrator(PhiParameters *p, RestartHelper *r);
  void Launch();
  void Run(int id, int num_threads, RunMethod method);
  Complex *density_matrix();
  ~HierarchyIntegrator();

 private:
  int integration_method_;

  // TODO(johanstr): Remove, the code using this is broken.
  Complex *hamiltonian_;

  Complex **full_hamiltonian_th_;
  Complex **hamiltonian_th_;
  Complex **hamiltonian_adjoint_th_;
  Complex **density_matrices_th_;
  Complex **density_matrices_tmp_th_;

  CubicSpline *initial_field_spline_th_;
  CubicSpline *final_field_spline_th_;
  Float *old_initial_field_value_th_;
  Float *old_final_field_value_th_;
  HilbertSpaceTruncator **truncator_th_;

  Complex liouville_prefactor_, negative_liouville_prefactor_;

  Float dt_f_;
  Complex dt_over_2_, dt_over_6_, dt_c_;

  RestartHelper *restart_helper_;
  ofstream outfile_;

  // Variables for time-local truncation.
  Complex **time_local_matrices_, **time_local_adjoint_matrices_;
  Complex **time_local_matrices_store_;
  Complex **hamiltonian_propagator_th_;
  Complex **hamiltonian_propagator_adjoint_th_;
  Complex ***time_local_truncation_terms_th_;
  Complex ***time_local_truncation_store_th_;

  Float *rkf45_diff_th_;

  Float *max_density_matrix_value_th_;

  // BiCGSTAB Variables.
  Complex *omega_btm_, *omega_top_, *alpha_;
  Complex *rho1_;
  Float *bicgstab_diff_;

  // Housekeeping variables for threads.
  pthread_mutex_t io_lock_;
  barrier_t integrator_threads_barrier_, all_threads_barrier_;

  int **node_indices_th_;
  int *node_count_th_;
  phi_big_size_t *matrix_element_count_th_;

  void Initialize(int num_threads, bool assign_memory);
  void PrepareForOutput();
  void PrintMemoryRequirements(bool output_to_file);
  void AllocateMemory(int id, int num_threads, RunMethod method);
  void PartitionHierarchy(int id);
  void PartitionHierarchySimple(int id);
  void CountInterThreadConnections();
  void InitializeDensityMatrix(int id);
  void InitializeBathCouplingOperators(int id);

  void LinkHierarchyNodes(HierarchyNode *nodes,
                          int id,
                          int num_threads,
                          Complex *density_matrices);
  void InitializePrefactors(const int id);

  void PrepForSteadyState(int id, int num_threads, RunMethod run_method);
  void PrepConstructTL();
  void PrintHierarchyConnections(int id);
  void PrepareBathOperators(int id);
  void Rk4Integrate(int id, int num_threads);
  void Rkf45Integrate(int id, int num_threads);
  hierarchy_updater::HierarchyUpdater *GetUpdater(int id,
                                                  const Complex *same,
                                                  const Complex *next,
                                                  const Complex *prev);

  void BicgstabSteadyState(int id, int num_threads);
  void BicgstabLSteadyState(int id, int num_threads);
  void BicgstabSteadyState_unstable(int id, int num_threads);
  void MinimizeStep(const HierarchyNode &node,
                    Complex *drho,
                    const int &time_step,
                    Complex *temp,
                    Complex *next,
                    Complex *prev,
                    Complex *same);
  void OutputTiming(int id,
                    int num_threads,
                    timeval *state_time,
                    int *state,
                    int num_steps);
  bool UpdateTimeDependentHamiltonian(Float time, int id);
  void UpdateTimeLocalTruncation(const int id,
                                 const Float time,
                                 const Float timestep);
  void StoreTimeLocalTruncationMatrices(const int id);
  void RestoreTimeLocalTruncationMatrices(const int id);

  std::string GetInfo();
  void Log(int min_verbosity, std::string s);
  void Log(int min_verbosity, int id, std::string s);
  HierarchyIntegrator();
};

#endif  // PHI_HIERARCHY_INTEGRATOR_H_
