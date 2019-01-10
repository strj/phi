#include "hierarchy_integrator.h"

#ifdef THREADAFFINITY
#include <sched.h>
#endif

#include <cstdio>
#include <ctime>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "blas_wrapper.h"
#include "complex_matrix.h"
#include "hierarchy_node.h"
#include "hierarchy_updater.h"
#include "hilbert_space_truncator.h"
#include "numeric_types.h"
#include "timer.h"

struct thread_data {
  HierarchyIntegrator *integrator;
  int id;
  int num_threads;
  RunMethod method;
};

void *LaunchThread(void *p) {
  struct thread_data *data;
  data = (struct thread_data *)p;
  (static_cast<HierarchyIntegrator *>(data->integrator)->Run)(
      data->id, data->num_threads, data->method);
  return 0;
}

//////////////////////////////////////////
////  Begin HierarchyIntegrator class ////
//////////////////////////////////////////
/*
 * Empty protected constructor for testing.
 */
HierarchyIntegrator::HierarchyIntegrator() : Hierarchy() {}

//  Constructor
//  ------------
//  Read in parameters.
//  Set up some matrices and constants.
HierarchyIntegrator::HierarchyIntegrator(PhiParameters *p) : Hierarchy(p) {
  restart_helper_ = new RestartHelper(parameters_->restart_input_filename,
                                      parameters_->restart_output_filename,
                                      parameters_->restart_backup_filename);

  identity_matrix_ =
      new Complex[parameters_->num_states * parameters_->num_states];
  SetElementsToZero(parameters_->num_states * parameters_->num_states,
                    identity_matrix_);
  for (int i = 0; i < parameters_->num_states; ++i) {
    identity_matrix_[i * (parameters_->num_states + 1)] = kOne;
  }
  Log(0, GetInfo());
}

HierarchyIntegrator::HierarchyIntegrator(PhiParameters *p, RestartHelper *r)
    : Hierarchy(p), restart_helper_(r) {
  identity_matrix_ =
      new Complex[parameters_->num_states * parameters_->num_states];
  SetElementsToZero(parameters_->num_states * parameters_->num_states,
                    identity_matrix_);
  for (int i = 0; i < parameters_->num_states; ++i) {
    identity_matrix_[i * (parameters_->num_states + 1)] = kOne;
  }
  Log(0, GetInfo());
}

std::string HierarchyIntegrator::GetInfo() {
  ostringstream ss;
  ss << "PHI Parallel Hierarchy Integrator by Johan Strumpfer, 2009-2012.\n"
     << "Integration of density matrix using hierarchy equations of motion.\n"
     << "\nPlease cite Strumpfer and Schulten, J. Chem. Theor. Comp. (2012)\n"
     << "in all publications reporting results obtained with PHI\n\n"
#ifdef NOBLAS
     << "Using internal matrix functions\n"
#else
     << "Using BLAS for matrix functions\n"
#endif
#ifdef SINGLEPRECISION
     << "Using single precision floating point operations\n"
#else
     << "Using double precision floating point operations\n"
#endif
     << "Version 1.1";
  return ss.str();
}

void HierarchyIntegrator::Log(int min_verbosity, std::string s) {
  if (verbose_ > min_verbosity) {
    std::cout << s << "\n";
  }
}

void HierarchyIntegrator::Log(int min_verbosity, int thread_id, std::string s) {
  if (verbose_ > min_verbosity) {
    pthread_mutex_lock(&io_lock_);
    std::cout << "[" << thread_id << "]: " << s << "\n";
    if (verbose_ > 2) {
      std::cout << std::flush;
    }
    pthread_mutex_unlock(&io_lock_);
  }
}

void HierarchyIntegrator::Launch() {
  // Threading
  pthread_t *tg;
  size_t stacksize;

  const int num_threads = parameters_->num_threads;
  const RunMethod run_method = parameters_->run_method;
  /* Now launch threads */
  switch (run_method) {
    case RK4:
    case RKF45:
    case RK4SPECTRUM:
    case RKF45SPECTRUM:
    case BICGSTABL:
    case BICGSTABU:
    case BICGSTAB: {
      Initialize(num_threads, true);
      Log(0, "\nLaunching Threads.");
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      // Set Max Stacksize =  8MB on os x.
      stacksize = 8388608;
      int err = pthread_attr_setstacksize(&attr, stacksize);
      if (err) std::cout << "\tSet Stacksize Error=" << err << std::endl;
      // Ensure threads are joinable/
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
      thread_data *t_data;
      // Total number of threads = integrator threads + 1 writer.
      tg = new pthread_t[num_threads + 1];
      t_data = new thread_data[num_threads + 1];
      // Launch each thread.
      for (int i = 0; i <= num_threads; ++i) {
        t_data[i].integrator = this;
        t_data[i].id = i;
        t_data[i].num_threads = num_threads;
        t_data[i].method = run_method;
        pthread_create(&tg[i], &attr, LaunchThread, &t_data[i]);
      }
      // Wait for all threads to finish.
      void *status;
      for (int i = 0; i <= num_threads; i++) {
        pthread_join(tg[i], &status);
      }
    } break;
    case PRINTHIERARCHY:
      Initialize(num_threads, false);
      PrintHierarchyConnections(num_threads);
      break;
    case PRINTMEMORY: {
      matrix_count_ = parameters_->matrix_count;
      num_elements_ = parameters_->num_states * parameters_->num_states;
      std::cout << "Total number of matrices: " << matrix_count_ << "\n";
      std::cout << "Number of elements per matrix: " << num_elements_ << "\n";
      PrintMemoryRequirements(false);
    } break;
    case BROKENINPUT:
    case NONE:
      break;
  }
  if (parameters_->verbose > -2) {
    std::cout << "Done.\n";
  }
}

Complex *HierarchyIntegrator::density_matrix() {
  Complex *density_matrix;
  if (!parameters_->is_truncated_system) {
    density_matrix = new Complex[num_states_ * num_states_];
    for (phi_size_t i = 0; i < num_states_ * num_states_; ++i) {
      density_matrix[i] = nodes_[0].density_matrix[i];
    }
  } else {
    const int hilbert_space_num_elements =
        parameters_->hilbert_space_num_elements;
    density_matrix = new Complex[hilbert_space_num_elements];
    truncator_th_[0]->UnTruncate(nodes_[0].density_matrix, density_matrix);
  }
  return density_matrix;
}

void HierarchyIntegrator::PrintMemoryRequirements(bool output_to_file) {
  float memreq = 0;
  float rk4 = 0;
  float rkf45 = 0;
  float spectrum_rk4 = 0;
  float spectrum_rkf45 = 0;
  float steadystate = 0;
  float matrices = 0;

  const int n = parameters_->num_states;
  const int m = parameters_->num_bath_couplings;
  const int k = parameters_->num_matsubara_terms;
  const int num_threads = parameters_->num_threads;
  const phi_big_size_t hilbert_space_num_elements =
      parameters_->hilbert_space_num_elements;
  const phi_size_t hilbert_space_size = parameters_->hilbert_space_size;

  matrices =
      static_cast<float>(sizeof(Complex) * num_elements_ * matrix_count_) /
      1024 / 1024;
  // HierarchyNodes rho and rho_mid
  memreq += sizeof(int) * (m + 4 + m + m + m + m + 5) +
            sizeof(Complex) * (1 + m + m + m + m + num_elements_);
  memreq *= 2 * matrix_count_;
  // Integration Variables
  memreq += sizeof(Complex) * (3 + num_elements_) + sizeof(int) * 6 +
            sizeof(time_t) * 2;
  if (parameters_->use_time_local_truncation) {
    memreq += sizeof(Complex) * num_elements_ * num_threads;
    memreq += sizeof(Complex) * m * k * num_elements_;
  }
  // Make sure Vbath matrices are taken into account: Kt+2 * M matrices, each
  // sized Ns*Ns.
  if (parameters_->is_diagonal_bath_coupling) {
    // Take the system-bath coupling operators into account:
    // (Kt + 2) * M matrices, each sized Ns * Ns.
    memreq += sizeof(Complex) * m * num_elements_ * (k + 2) * num_threads;
    // For bath_operators.
    memreq += sizeof(Complex) * m * hilbert_space_size;
  }
  if (parameters_->is_full_bath_coupling) {
    // Take the system-bath coupling operators into account:
    // 3 * M matrices, each sized Ns * Ns.
    memreq += sizeof(Complex) * m * num_elements_ * 3 * num_threads;
    // For bath_operators.
    memreq += sizeof(Complex) * m * hilbert_space_num_elements;
  }

  // full_hamiltonian_th
  memreq += sizeof(Complex) * hilbert_space_num_elements * num_threads;

  if (parameters_->is_truncated_system) {
    // Memory for truncators
    // eigenvalues
    memreq += sizeof(Complex) * hilbert_space_size * num_threads;
    // eigenvectors
    memreq += sizeof(Complex) * hilbert_space_num_elements * num_threads;
    // Size variables in truncator.
    memreq += sizeof(phi_size_t) * 2 * num_threads;
    // TODO(johanstr) Add ZheevrDecomposition parameters
    // H_th
    memreq += sizeof(Complex) * num_elements_ * num_threads;
  }

  rk4 = static_cast<float>(sizeof(Complex) * num_elements_ * matrix_count_ * 4 +
                           memreq) /
        1024 / 1024;

  rkf45 += sizeof(Complex) * num_elements_ * matrix_count_ * 6;
  rkf45 += sizeof(Complex) * 31;
  rkf45 += sizeof(Float) * 7 + sizeof(int) * 2;
  rkf45 = static_cast<float>(memreq + rkf45) / 1024 / 1024;

  // Spectrum
  // HierarchyNode rho and rho_mid
  memreq = 0;
  memreq += sizeof(int) * (m + 4 + m + m + m + m + 5) +
            sizeof(Complex) * (1 + m + m + m + m + n);
  memreq *= 2 * matrix_count_;
  // Integration Variables
  memreq += sizeof(Complex) * (3 + n) + sizeof(int) * 6 + sizeof(time_t) * 2;

  spectrum_rk4 += sizeof(Complex) * (n + 1) * matrix_count_ * 4;
  spectrum_rk4 = static_cast<float>(spectrum_rk4 + memreq) / 1024 / 1024;

  spectrum_rkf45 += sizeof(Complex) * 3 * n * matrix_count_ * 6;
  spectrum_rkf45 += sizeof(Complex) * 31;
  spectrum_rkf45 += sizeof(Float) * 7 + sizeof(int) * 2;
  spectrum_rkf45 = static_cast<float>(spectrum_rkf45 + memreq) / 1024 / 1024;

  // SteadyState
  steadystate = 0;
  // Hierarchy
  steadystate += sizeof(int) * (m + 4 + m + m + m + m + 5) +
                 sizeof(Complex) * (1 + m + m + m + m);
  // BiCGSTAB vectors & SteadyStateRho
  steadystate += 6 * (sizeof(Complex) * (n * n));
  steadystate *= matrix_count_;
  steadystate += 9 * sizeof(Complex) + 3 * sizeof(int) + 2 * sizeof(Float);
  steadystate = steadystate / 1024 / 1024;

  if (!output_to_file) {
    std::cout << "Memory to store matrices for calculations:\n";
    std::cout << "\thierarchy: " << matrices << " Mb.\n";
    std::cout << "\trk4: " << rk4 << " Mb.\n";
    std::cout << "\trkf45: " << rkf45 << " Mb.\n";
    std::cout << "\trk4spectrum: " << spectrum_rk4 << " Mb.\n";
    std::cout << "\trkf45spectrum: " << spectrum_rkf45 << " Mb.\n";
    std::cout << "\tsteadystate: " << steadystate << " Mb.\n";
  } else {
    outfile_ << "#Memory to store matrices for calculations:\n";
    outfile_ << "#hierarchy: " << matrices << " Mb.\n";
    outfile_ << "#rk4: " << rk4 << " Mb.\n";
    outfile_ << "#rkf45: " << rkf45 << " Mb.\n";
    outfile_ << "#rk4spectrum: " << spectrum_rk4 << " Mb.\n";
    outfile_ << "#rkf45spectrum: " << spectrum_rkf45 << " Mb.\n";
    outfile_ << "#steadystate: " << steadystate << " Mb.\n";
  }
}

void HierarchyIntegrator::PrepareForOutput() {
  outfile_.open((parameters_->output_filename).c_str());
  parameters_->Write(&outfile_, matrix_count_);
  outfile_ << "#Total Number of matrices:" << matrix_count_ << std::endl;
  PrintMemoryRequirements(true);
  outfile_ << "#-------------\n";
}

void HierarchyIntegrator::PrintHierarchyConnections(int num_threads) {
  int mc_minus = 0;
  int vb = verbose_;
  verbose_ = -1;
  for (int i = 0; i < parameters_->num_threads; ++i) {
    LinkHierarchyNodes(nodes_, i, 1, NULL);
  }
  verbose_ = vb;
  ostringstream ss;
  ss << "Hierarchy connections:\n";
  for (int i = parameters_->hierarchy_truncation_level - 1; i >= 0; --i) {
    mc_minus += hierarchy_level_[i];
    ss << "Level " << i << ":\n";
    for (int m = hierarchy_level_[i] - 1; m >= 0; --m) {
      ss << "\t" << matrix_count_ - mc_minus + m << " ";
      ss << "->";
      for (int k = 0;
           k < nodes_[matrix_count_ - mc_minus + m].num_prev_hierarchy_nodes;
           ++k) {
        ss << " "
           << nodes_[matrix_count_ - mc_minus + m].prev_hierarchy_nodes[k]->id;
      }
      ss << "\n";
    }
  }
  std::cout << ss.str();
}

void HierarchyIntegrator::PrepConstructTL() {
  Log(0, "Prep time-local truncation matrices (with threads).");
  const int num_bath_couplings = parameters_->num_bath_couplings;
  const int num_threads = parameters_->num_threads;
  time_local_matrices_ = new Complex *[num_bath_couplings];
  time_local_adjoint_matrices_ = new Complex *[num_bath_couplings];
  time_local_matrices_store_ = new Complex *[num_bath_couplings];
  time_local_truncation_terms_th_ = new Complex **[num_bath_couplings];
  time_local_truncation_store_th_ = new Complex **[num_bath_couplings];
  hamiltonian_propagator_th_ = new Complex *[num_threads];
  hamiltonian_propagator_adjoint_th_ = new Complex *[num_threads];
}

/*
 *  Set up the integration with integration type Method:
 * 1 = Runga-Kutta 4 fixed timesteps. Time Local Truncation and Arb. Diagonal
 *     Bath Operators allowed. Shi-Adaptive Truncation allowed for independent
 *     bath operators. Full bath operators allowe.
 * 2 = Runga-Kutta-Fehlberg 4/5 adaptive timestepping. Time-Location truncation
 *     allowed. Arb. Diagonal Bath Operators allowed. Shi-Adaptive
 *     Truncation for independent baths allowed. Full bath operators allowed.
 * 3 = Spectrum calculation using Runga-Kutta 4 fixed timestepping.
 * 4 = Spectrum calculation using Runga-Kutta-Fehlberg 4/5 adaptive
 *     timestepping
 * 5 = BiCGSTAB Steady State Calculation independent bath operators only.
 *     TL Truncation NOT allowed.
 */
void HierarchyIntegrator::Initialize(int num_threads, bool assign_memory) {
  if (!parameters_->are_parameters_valid) {
    Log(-1, "Parameter error.");
    return;
  }
  // Create local copies of the parameters for easier access:
  const Float hbar = parameters_->hbar;
  const Float thermal_energy =
      parameters_->boltzmann_constant * parameters_->temperature;
  const Float nu = 2 * PI * thermal_energy / hbar;
  const Float timestep = parameters_->timestep;
  const Float *gamma = parameters_->gamma;
  const Float *lambda = parameters_->lambda;
  const bool is_diagonal_bath_coupling = parameters_->is_diagonal_bath_coupling;
  const bool use_time_local_truncation = parameters_->use_time_local_truncation;
  const int hierarchy_truncation_level =
      parameters_->hierarchy_truncation_level;
  const RunMethod run_method = parameters_->run_method;
  const bool do_integration = run_method == RK4 || run_method == RKF45 ||
                              run_method == RK4SPECTRUM ||
                              run_method == RKF45SPECTRUM;
  const bool do_steady_state = run_method == BICGSTAB ||
                               run_method == BICGSTABL ||
                               run_method == BICGSTABU;

  dt_over_2_ = Complex(timestep / 2);
  dt_over_6_ = Complex(timestep / 6);
  dt_c_ = Complex(timestep);
  dt_f_ = timestep;

  liouville_prefactor_ = Complex(0, -1. / hbar);
  negative_liouville_prefactor_ = Complex(0, 1. / hbar);

  num_elements_ = num_states_ * num_states_;
  matsubara_truncation_ = parameters_->num_matsubara_terms;
  num_bath_coupling_terms_ = parameters_->num_bath_couplings;
  num_correlation_fn_terms_ = num_bath_coupling_terms_ * matsubara_truncation_;

  nodes_ = NULL;
  nodes_tmp_ = NULL;

  Log(0, "Initializing.");
  if (is_diagonal_bath_coupling) {
    Log(0, "Using diagonal system-bath coupling.");
  }
  if (parameters_->is_full_bath_coupling) {
    Log(0, "Using full system-bath coupling.");
  }
  if (!parameters_->is_rho_normalized) {
    Log(0, "Not using normalized auxiliary density matrices.");
  }
  if (gamma != NULL && lambda != NULL) {
    Log(2, "Initializing correlation function parameters");
    correlation_fn_constants_ = new Complex[num_correlation_fn_terms_];
    SetElementsToZero(num_correlation_fn_terms_, correlation_fn_constants_);
    abs_correlation_fn_constants_ = new Float[num_correlation_fn_terms_];
    if (verbose_ > 2) {
      std::cout << "C:";
    }
    Float min_abs_c;
    min_abs_c = numeric_limits<Float>::max();
    for (int i = 0; i < num_bath_coupling_terms_; ++i) {
      if (gamma[i] > 0) {
        correlation_fn_constants_[i * matsubara_truncation_] =
            Complex(gamma[i] * lambda[i] / hbar) *
            Complex(1 / tan(gamma[i] * hbar / (2 * thermal_energy)), -1);
      }
      abs_correlation_fn_constants_[i * matsubara_truncation_] =
          abs(correlation_fn_constants_[i * matsubara_truncation_]);
      if (abs_correlation_fn_constants_[i * matsubara_truncation_] <
          min_abs_c) {
        min_abs_c = abs_correlation_fn_constants_[i * matsubara_truncation_];
      }
      if (verbose_ > 2) {
        std::cout << correlation_fn_constants_[i * matsubara_truncation_] << ",";
      }

      for (int k = 1; k < matsubara_truncation_; k++) {
        if (gamma[i] > 0) {
          correlation_fn_constants_[i * matsubara_truncation_ + k] = Complex(
              4 * lambda[i] * gamma[i] * thermal_energy / hbar / hbar *
              (nu * k) / (pow(nu * k, 2) - pow(gamma[i], 2)));
        }
        abs_correlation_fn_constants_[i * matsubara_truncation_ + k] =
            abs(correlation_fn_constants_[i * matsubara_truncation_ + k]);
        if (abs_correlation_fn_constants_[i * matsubara_truncation_ + k] <
            min_abs_c) {
          min_abs_c =
              abs_correlation_fn_constants_[i * matsubara_truncation_ + k];
        }
        if (verbose_ > 2) {
          std::cout << correlation_fn_constants_[i * matsubara_truncation_ + k]
               << ",";
        }
      }
    }
    if (verbose_ > 2) std::cout << "\n";
  }

  if (num_states_ > 0 && hierarchy_truncation_level > 0 &&
      num_bath_coupling_terms_ > 0 && matsubara_truncation_ > 0) {
    Log(2, "Initializing matrix hyperindex.");
    int vb = verbose_;
    if (run_method == PRINTHIERARCHY) {
      verbose_ = 3;
    }
    ConstructI();
    if (run_method == PRINTHIERARCHY) {
      verbose_ = vb;
    }
    if ((verbose_ > 0 && run_method != PRINTHIERARCHY) ||
        run_method == PRINTMEMORY) {
      PrintMemoryRequirements(false);
    }
  }

  if (do_integration || do_steady_state) {
    PrepareForOutput();
  }

  if (use_time_local_truncation && do_integration) {
    Log(2, "Initializing time local truncation.");
    PrepConstructTL();
  } else {
    time_local_matrices_ = NULL;
    time_local_adjoint_matrices_ = NULL;
    time_local_matrices_store_ = NULL;
    time_local_truncation_terms_th_ = NULL;
    time_local_truncation_store_th_ = NULL;
    hamiltonian_propagator_th_ = NULL;
    hamiltonian_propagator_adjoint_th_ = NULL;
  }
  node_count_th_ = new int[num_threads + 1];
  node_indices_th_ = new int *[num_threads + 1];
  matrix_element_count_th_ = new phi_big_size_t[num_threads];

  full_hamiltonian_th_ = new Complex *[num_threads];
  hamiltonian_th_ = new Complex *[num_threads];
  hamiltonian_adjoint_th_ = new Complex *[num_threads];

  truncator_th_ = new HilbertSpaceTruncator *[num_threads];

  old_initial_field_value_th_ = new Float[num_threads];
  old_final_field_value_th_ = new Float[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    // Ensure that first Hamiltonian update is performed.
    old_initial_field_value_th_[i] = numeric_limits<Float>::max();
    old_final_field_value_th_[i] = numeric_limits<Float>::max();
  }

  full_bath_operators_ = new Complex *[num_bath_coupling_terms_];
  bath_operators_ = new Complex *[num_bath_coupling_terms_];

  density_matrices_th_ = new Complex *[num_threads];
  density_matrices_tmp_th_ = new Complex *[num_threads];
  for (int t = 0; t < num_threads; ++t) {
    density_matrices_th_[t] = NULL;
    density_matrices_tmp_th_[t] = NULL;
  }

  if (parameters_->is_spectrum_calculation && parameters_->is_restarted) {
    // If we're calculating an emission spectrum.
    liouville_prefactor_ *= Complex(-1);
  }

  Log(2, "Creating hierarchy nodes.");
  nodes_ = new HierarchyNode[matrix_count_];
  if (do_integration) {
    Log(2, "Creating rho_mid nodes.");
    nodes_tmp_ = new HierarchyNode[matrix_count_];
  } else if (do_steady_state) {
    Log(2, "Creating steady state calculation parameters.");
    bicgstab_diff_ = new Float[num_threads + 1];
    alpha_ = new Complex[num_threads];
    omega_top_ = new Complex[num_threads];
    omega_btm_ = new Complex[num_threads];
    rho1_ = new Complex[num_threads];
  }

  max_density_matrix_value_th_ = new Float[matrix_count_];
  for (int i = 0; i < matrix_count_; ++i) {
    max_density_matrix_value_th_[i] = 0;
  }

  rkf45_diff_th_ = new Float[num_threads + 1];

  // Set up integrator and writer barriers.
  Log(2, "Initializing thread barriers.");
  BarrierInit(&integrator_threads_barrier_, num_threads);
  BarrierInit(&all_threads_barrier_, num_threads + 1);

  // Set up mutexes.
  Log(2, "Initializing I/O mutex.");
  pthread_mutex_init(&io_lock_, NULL);

#ifdef THREADAFFINITY
  /* Check cpu affinities */
  const int *cpuaffinity = parameters_->cpuaffinity;
  const int ncores = parameters_->ncores;
  Log(2, "Setting up cpu affinities.");
  int ncpus = 0;
  for (int i = 0; i < ncores; ++i) {
    if (kCpuaffinity[i] > ncpus) ncpus = cpuaffinity[i];
  }
#endif

  for (int i = 0; i < num_threads + 1; ++i) {
    rkf45_diff_th_[i] = 1E99;
  }

  if (!parameters_->stupid_hierarchy_partition) {
    PartitionHierarchy(num_threads);
  } else {
    PartitionHierarchySimple(num_threads);
  }
  if (assign_memory) {
    CountInterThreadConnections();
  }
}

void HierarchyIntegrator::PartitionHierarchySimple(int num_threads) {
  ostringstream ss;
  ss << "Partitioning hierarchy into " << num_threads
     << " sets using cyclic scheme.";
  Log(1, ss.str());
  ss.clear();
  for (int id = 0; id < num_threads + 1; ++id) {
    if (id < num_threads) {
      if (id < matrix_count_ % num_threads) {
        node_count_th_[id] = ceil((Float)matrix_count_ / num_threads);
      } else {
        node_count_th_[id] = floor((Float)matrix_count_ / num_threads);
      }
    } else {
      node_count_th_[id] = matrix_count_;
    }
    node_indices_th_[id] = new int[node_count_th_[id]];
  }
  ss << "Matrices per thread:\n";
  for (int i = 0; i < num_threads; ++i) {
    ss << i << ": " << node_count_th_[i] << "\n";
  }
  Log(2, ss.str());
  ss.clear();
  // Cyclic partitioning.
  int n = 0;
  for (int t = 0; t < num_threads; ++t) {
    n = 0;
    for (int i = t; i < matrix_count_; i += num_threads) {
      node_indices_th_[t][n] = i;
      multi_index_map_[i].thread = t;
      ++n;
    }
  }
  for (int i = 0; i < matrix_count_; ++i) {
    node_indices_th_[num_threads][i] = i;
  }
  Log(3, "Partitioning Done.");
}

void HierarchyIntegrator::PartitionHierarchy(int num_threads) {
  ostringstream ss;
  ss << "Partitioning hierarchy into " << num_threads << " sets.";
  Log(1, ss.str());
  const int truncation_level = parameters_->hierarchy_truncation_level;
  MultiIndex index(num_correlation_fn_terms_);
  map<MultiIndex, int, CompareMultiIndex>::iterator fnd_result;
  for (int id = 0; id < num_threads + 1; ++id) {
    if (id < num_threads) {
      if (id < matrix_count_ % num_threads) {
        node_count_th_[id] = ceil((Float)matrix_count_ / num_threads);
      } else {
        node_count_th_[id] = floor((Float)matrix_count_ / num_threads);
      }
    } else {
      node_count_th_[id] = matrix_count_;
    }
    node_indices_th_[id] = new int[node_count_th_[id]];
  }

  // Writer thread:
  for (int i = 0; i < matrix_count_; ++i) {
    node_indices_th_[num_threads][i] = i;
  }

  if (verbose_ > 2) {
    std::cout << "Matrices per thread:\n";
    for (int i = 0; i < num_threads; ++i) {
      std::cout << i << ": " << node_count_th_[i] << "\n";
    }
  }
  int n = 0;
  int *m_as;
  m_as = new int[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    m_as[i] = 0;
  }
  int nl = 0;

  // Do corners - then update rest by parent thread weight, if equal weight then
  // use least assigned thread, if equal assigned and less than matrix_count_th,
  // then assigned lowest number, else assigned lowest thread weight overall.

  // Assign level 0.
  node_indices_th_[0][0] = 0;
  m_as[0] = 1;
  multi_index_map_[0].thread = 0;
  n = 1;
  if (truncation_level > 1) {
    // First assign Level 1 and corners.
    for (int i = 0; i < hierarchy_level_[1]; ++i) {
      node_indices_th_[i % num_threads][m_as[i % num_threads]] = n + i;
      m_as[i % num_threads] += 1;
      multi_index_map_[n + i].thread = i % num_threads;
    }

    int next_t = 0;
    if (hierarchy_level_[1] < num_threads) {
      next_t = hierarchy_level_[1];
    }

    int max_t = -1;

    n = 1 + hierarchy_level_[1];
    int *prev_weights;
    prev_weights = new int[num_threads];
    Log(3, "Fill in.");
    for (int l = 2; l < truncation_level; ++l) {
      for (int i = 0; i < hierarchy_level_[l]; ++i) {
        if (multi_index_map_[n + i].thread == -1 && next_t == 0) {
          for (int t = 0; t < num_threads; ++t) {
            prev_weights[t] = 0;
          }

          // Determine which thread will call I[n+i].
          for (int j = 0; j < num_correlation_fn_terms_; ++j) {
            index = multi_index_map_[n + i];
            if (index[j] > 0) {
              index.Dec(j + 1);
              fnd_result = multi_index_map_lookup_.find(index);
              nl = fnd_result->second;
              prev_weights[multi_index_map_[nl].thread] += 1;
            }
          }

          // Find the thread that will call it the most. If two or more call it
          // equally then assign least occupied thread.
          max_t = 0;
          for (int t = 0; t < num_threads; ++t) {
            if (prev_weights[t] > prev_weights[max_t]) {
              max_t = t;
            } else if (prev_weights[t] == prev_weights[max_t] &&
                       m_as[t] < m_as[max_t] && prev_weights[t] > 0) {
              max_t = t;
            }
          }

          // If assigned thread is already maximally occupied, assign to least
          // occupied thread.
          if (m_as[max_t] >= ceil(static_cast<Float>(matrix_count_ /
                                                     num_threads))) {
            max_t = 0;
            for (int t = 0; t < num_threads; ++t) {
              if (m_as[max_t] > m_as[t]) {
                max_t = t;
              }
            }
          }
          // Now that we have identified a thread, do the assignment.
          node_indices_th_[max_t][m_as[max_t]] = n + i;
          m_as[max_t] += 1;
          multi_index_map_[n + i].thread = max_t;
        } else if (multi_index_map_[n + i].thread == -1 && next_t != 0) {
          node_indices_th_[next_t][m_as[next_t]] = n + i;
          m_as[next_t] += 1;
          multi_index_map_[n + i].thread = next_t;
          ++next_t;
          next_t = next_t % num_threads;
        }
      }
      n += hierarchy_level_[l];
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    if (m_as[t] != node_count_th_[t]) {
      node_count_th_[t] = m_as[t];
    }
  }
  Log(2, ss.str());
  ss.clear();
  for (int t = 0; t < num_threads; ++t) {
    ss << "[" << t << "]:\t";
    for (int m = 0; m < m_as[t]; m++) {
      ss << node_indices_th_[t][m] << " ";
    }
    ss << "\n";
    Log(2, ss.str());
    ss.clear();
  }
  Log(3, "Partitioning Done.");
}

void HierarchyIntegrator::CountInterThreadConnections() {
  int vertex_id, prev_vertex_id;
  int j;
  int num_inter_thread_edges = 0;
  map<MultiIndex, int, CompareMultiIndex>::iterator fnd_result;
  MultiIndex index(num_correlation_fn_terms_);
  for (vertex_id = 1; vertex_id < matrix_count_; ++vertex_id) {
    for (j = 0; j < num_correlation_fn_terms_; ++j) {
      index = multi_index_map_[vertex_id];
      if (index[j] > 0) {
        index.Dec(j + 1);
        fnd_result = multi_index_map_lookup_.find(index);
        prev_vertex_id = fnd_result->second;
        if (multi_index_map_[prev_vertex_id].thread !=
            multi_index_map_[vertex_id].thread) {
          ++num_inter_thread_edges;
        }
      }
    }
  }
  const int run_method = parameters_->run_method;
  if (run_method == RK4 || run_method == RKF45 || run_method == RK4SPECTRUM ||
      run_method == RKF45SPECTRUM || run_method == BICGSTAB ||
      run_method == BICGSTABL || run_method == BICGSTABU) {
    stringstream ss;
    ss << "Number of inter-thread connections: " << num_inter_thread_edges
       << "\n";
    Log(1, ss.str());
  }
}

/////////////////////////////////////////////////////////////////////////////
///////                                                             /////////
/////// THREADED CODE BELOW HERE --------- THREADED CODE BELOW HERE /////////
///////                                                             /////////
/////////////////////////////////////////////////////////////////////////////

void HierarchyIntegrator::Run(int id, int num_threads, RunMethod method) {
#ifdef THREADAFFINITY
  const int *cpu_affinity = parameters_->cpuaffinity;
  cpu_set_t cpuset;
  pthread_t thread;
  int error_number;

  thread = pthread_self();
  if (id < num_threads && cpuaffinity != NULL) {
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_affinity[id], &cpuset);
    error_number = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
      ostringstream ss;
      ss << "Failed to set cpu affinity (errno=" << error_number << ").";
      Log(-1, id, ss.str());
    } else {
      ostringstream ss;
      ss << "On cpu " << cpu_affinity[id] << ".";
      Log(0, id, ss.str());
    }
  }
#endif
  AllocateMemory(id, num_threads, method);
  BarrierWait(&all_threads_barrier_);
  Log(0, id, "Running.");
  switch (method) {
    case RK4:
      Rk4Integrate(id, num_threads);
      break;
    case RKF45:
      Rkf45Integrate(id, num_threads);
      break;
    case RK4SPECTRUM:
      Rk4Integrate(id, num_threads);
      break;
    case RKF45SPECTRUM:
      Rkf45Integrate(id, num_threads);
      break;
    case BICGSTABL:
      BicgstabLSteadyState(id, num_threads);
      break;
    case BICGSTAB:
      BicgstabSteadyState(id, num_threads);
      break;
    case BICGSTABU:
      BicgstabSteadyState_unstable(id, num_threads);
      break;
    default:
      break;
  }
  pthread_exit(0);
}

void HierarchyIntegrator::AllocateMemory(int id, int num_threads,
                                         RunMethod run_method) {
  Log(0, id, "Allocating memory.");
  const bool is_spectrum_calculation = parameters_->is_spectrum_calculation;
  const bool use_time_local_truncation = parameters_->use_time_local_truncation;
  const int num_states = parameters_->num_states;
  const int num_bath_couplings = parameters_->num_bath_couplings;
  const int hilbert_space_num_elements =
      parameters_->hilbert_space_num_elements;
  const int hilbert_space_size = parameters_->hilbert_space_size;
  const int num_densit_matrix_elements =
      is_spectrum_calculation ? num_states : num_elements_;
  const bool is_integration =
      (run_method == RK4 || run_method == RKF45 || run_method == RK4SPECTRUM ||
       run_method == RKF45SPECTRUM);
  const bool is_truncated_system = parameters_->is_truncated_system;
  BarrierWait(&all_threads_barrier_);
  if (id < num_threads) {
    // TODO(johanstr): Extract to InitializeHamiltonian.
    Log(2, id, "Allocating full Hamiltonian.");
    // Thread-local copy of the full Hamiltonian.
    full_hamiltonian_th_[id] = new Complex[hilbert_space_num_elements];
    SetElementsToZero(hilbert_space_num_elements, full_hamiltonian_th_[id]);
    if (!is_truncated_system) {
      Log(2, id, "Not truncated system: linking integration Hamiltonian.");
      // hilbert_space_num_elements == num_elements.
      // Thread-local copy of the truncated Hamiltonian.
      hamiltonian_th_[id] = full_hamiltonian_th_[id];
    } else {
      Log(2, id, "Creating truncator");
      truncator_th_[id] =
          new HilbertSpaceTruncator(hilbert_space_size, num_states);
      // Thread-local copy of the truncated Hamiltonian.
      hamiltonian_th_[id] = new Complex[num_elements_];
      SetElementsToZero(num_elements_, hamiltonian_th_[id]);
    }
    Log(2, id, "Allocating conjugate transpose integration Hamiltonian");
    hamiltonian_adjoint_th_[id] = new Complex[num_elements_];
    SetElementsToZero(num_elements_, hamiltonian_adjoint_th_[id]);
    if (parameters_->time_indepenent_hamiltonian != NULL) {
      Log(2, id, "Copying static Hamiltonian.");
      blas::Copy(hilbert_space_num_elements,
                 parameters_->time_indepenent_hamiltonian,
                 full_hamiltonian_th_[id]);

      if (is_truncated_system) {
        Log(2, id, "Truncating static Hamiltonian.");
        truncator_th_[id]->UpdateWith(full_hamiltonian_th_[id]);
        truncator_th_[id]->Truncate(full_hamiltonian_th_[id],
                                    hamiltonian_th_[id]);
      } else {
        blas::Copy(num_elements_, parameters_->time_indepenent_hamiltonian,
                   hamiltonian_th_[id]);
      }
      ConjugateTranspose(num_states, hamiltonian_th_[id],
                         hamiltonian_adjoint_th_[id]);
    } else {
      Log(2, id, "Setting annealing splines.");
      // Annealing calculation.
      initial_field_spline_th_ =
          new CubicSpline(parameters_->initial_hamiltonian_field,
                          parameters_->initial_hamiltonian_field_size);
      final_field_spline_th_ =
          new CubicSpline(parameters_->final_hamiltonian_field,
                          parameters_->final_hamiltonian_field_size);
      if (is_truncated_system) {
        Log(2, id, "Truncating time-dependent Hamiltonian.");
        const Complex *h_z = parameters_->final_annealing_hamiltonian;
        const Complex *h_x = parameters_->initial_annealing_hamiltonian;
        Complex h_x_multiplier = initial_field_spline_th_->Get(0);
        Complex h_z_multiplier = final_field_spline_th_->Get(0);
        for (phi_size_t i = 0; i < hilbert_space_num_elements; ++i) {
          full_hamiltonian_th_[id][i] =
              h_x_multiplier * h_x[i] + h_z_multiplier * h_z[i];
        }
        truncator_th_[id]->UpdateWith(full_hamiltonian_th_[id]);
        truncator_th_[id]->Truncate(full_hamiltonian_th_[id],
                                    hamiltonian_th_[id]);
      }
    }

    InitializeBathCouplingOperators(id);

    // TODO(johanstr): Extract to InitializeTimeLocalOperators
    if (use_time_local_truncation) {
      hamiltonian_propagator_th_[id] = new Complex[num_elements_];
      SetElementsToZero(num_elements_, hamiltonian_propagator_th_[id]);
      hamiltonian_propagator_adjoint_th_[id] = new Complex[num_elements_];
      SetElementsToZero(num_elements_, hamiltonian_propagator_adjoint_th_[id]);

      for (int m = id; m < num_bath_couplings; m += num_threads) {
        time_local_matrices_[m] = new Complex[num_elements_];
        SetElementsToZero(num_elements_, time_local_matrices_[m]);
        time_local_adjoint_matrices_[m] = new Complex[num_elements_];
        SetElementsToZero(num_elements_, time_local_adjoint_matrices_[m]);
        time_local_matrices_store_[m] = new Complex[num_elements_];
        SetElementsToZero(num_elements_, time_local_matrices_store_[m]);

        time_local_truncation_terms_th_[m] =
            new Complex *[matsubara_truncation_];
        time_local_truncation_store_th_[m] =
            new Complex *[matsubara_truncation_];
        for (int k = 0; k < matsubara_truncation_; ++k) {
          time_local_truncation_terms_th_[m][k] = new Complex[num_elements_];
          SetElementsToZero(num_elements_,
                            time_local_truncation_terms_th_[m][k]);
          time_local_truncation_store_th_[m][k] = new Complex[num_elements_];
          SetElementsToZero(num_elements_,
                            time_local_truncation_store_th_[m][k]);
        }
      }
    }

    // Hierarchy matrices.
    matrix_element_count_th_[id] =
        node_count_th_[id] * num_densit_matrix_elements;
    density_matrices_th_[id] = new Complex[matrix_element_count_th_[id]];

    phi_size_t nm;
    for (int i = 0; i < node_count_th_[id]; ++i) {
      nm = node_indices_th_[id][i];
      nodes_[nm].density_matrix =
          density_matrices_th_[id] + i * num_densit_matrix_elements;
      for (int n = 0; n < num_densit_matrix_elements; ++n) {
        nodes_[nm].density_matrix[n] = kZero;
      }
    }
    LinkHierarchyNodes(nodes_, id, num_threads, density_matrices_th_[id]);

    if (is_integration) {
      // Hierarchy matrices to store midpoints during Runga-Kutta integration.
      density_matrices_tmp_th_[id] = new Complex[matrix_element_count_th_[id]];

      phi_size_t nm;
      for (int i = 0; i < node_count_th_[id]; ++i) {
        nm = node_indices_th_[id][i];
        nodes_tmp_[nm].density_matrix =
            density_matrices_tmp_th_[id] + i * num_densit_matrix_elements;
        for (int n = 0; n < num_densit_matrix_elements; ++n) {
          nodes_tmp_[nm].density_matrix[n] = kZero;
        }
      }

      int vb = verbose_;
      if (id == 0) {
        verbose_ = -1;
      }
      BarrierWait(&integrator_threads_barrier_);
      LinkHierarchyNodes(nodes_tmp_, id, num_threads,
                         density_matrices_tmp_th_[id]);
      if (id == 0) {
        verbose_ = vb;
      }
    }
    InitializePrefactors(id);
  }
  BarrierWait(&all_threads_barrier_);
  if (is_integration) {
    InitializeDensityMatrix(id);
  } else if ((run_method == BICGSTAB ||
              run_method == BICGSTABU ||
              run_method == BICGSTABL) && id < num_threads) {
    PrepForSteadyState(id, num_threads, run_method);
  }
  Log(2, id, "Finished initialization.");
  BarrierWait(&all_threads_barrier_);
}

void HierarchyIntegrator::InitializeDensityMatrix(int id) {
  Log(2, id, "Preparing initial state.");
  if (id != 0) {
    // Only one thread needs to initialize the density matrix.
    return;
  }
  if (!parameters_->is_spectrum_calculation) {
    if (parameters_->is_restarted) {
      Log(3, id, "\tReading initial state from restart file.");
      pthread_mutex_lock(&io_lock_);
      // Read the restart parameters.
      restart_helper_->ReadRestartFile(&(parameters_->start_time),
                                       &(parameters_->timestep),
                                       nodes_,
                                       matrix_count_,
                                       time_local_truncation_store_th_);
      pthread_mutex_unlock(&io_lock_);
      // Update the timestep.
      dt_f_ = parameters_->timestep;
      dt_over_2_ = Complex(dt_f_ / 2);
      dt_over_6_ = Complex(dt_f_ / 6);
      dt_c_ = Complex(dt_f_);
      if (parameters_->use_time_local_truncation) {
        // Restore the time-local truncation state.
        for (phi_size_t m = 0; m < num_bath_coupling_terms_; ++m) {
          SumMatrices(time_local_truncation_store_th_[m],
                      matsubara_truncation_,
                      num_elements_,
                      time_local_matrices_store_[m]);
          RestoreTimeLocalTruncationMatrices(m);
        }
      }
    } else if (parameters_->initial_density_matrix != NULL) {
      Log(3, id, "\tReading initial state from input parameters.");
      int *m = node_indices_th_[id];
      if (parameters_->is_truncated_system) {
        Log(3, id, "\tTruncating to compute space.");
        truncator_th_[id]->Truncate(parameters_->initial_density_matrix,
                                 nodes_[*m].density_matrix);
      } else {
        Log(3, id, "\tCopying initial state to rho.");
        blas::Copy(num_elements_,
                   parameters_->initial_density_matrix,
                   nodes_[*m].density_matrix);
      }
      Float scale = 1;
      if (parameters_->is_rho_normalized) {
        for (int i = 0; i < num_correlation_fn_terms_; i++) {
          scale *= sqrt(pow(abs_correlation_fn_constants_[i],
                               nodes_[*m].index[i]) *
                           PartialFactorial(1, nodes_[*m].index[i]));
        }
      }
      for (int i = 0; i < num_elements_; ++i) {
        nodes_[*m].density_matrix[i] /= scale;
      }
    } else {
      Log(3, id, "\tPreparing initial state based on Hamiltonian spectrum.");
      HilbertSpaceTruncator t(parameters_->hilbert_space_size, num_elements_);
      t.UpdateWith(full_hamiltonian_th_[id]);
      Complex *tmp_matrix = new Complex[num_elements_];
      t.FillDiagonal(tmp_matrix);
      Float partition_function = 0;
      const Float thermal_energy =
          parameters_->boltzmann_constant * parameters_->temperature;
      for (int i = 0; i < num_elements_; ++i) {
        partition_function +=
            exp(-real(tmp_matrix[i * num_elements_ + i]) / thermal_energy);
      }
      for (int i = 0; i < num_elements_; ++i) {
        tmp_matrix[i * num_elements_ + i] =
            exp(-tmp_matrix[i * num_elements_ + i] / thermal_energy) /
            partition_function;
      }
      if (parameters_->is_truncated_system) {
        blas::Copy(num_elements_, tmp_matrix, nodes_[0].density_matrix);
      } else {
        t.UnTruncate(tmp_matrix, nodes_[0].density_matrix);
      }
    }
  } else {
    // Is a spectrum calculation.
    if (parameters_->is_restarted) {
      // This can be used for calculation of emission spectra that start from an
      // equilibruim density matrix.
      // Note that emission calculation takes Ns x Ns density matrix as input
      // even though only a single row or column of length Ns is used for
      // spectrum calculations.
      pthread_mutex_lock(&io_lock_);
      restart_helper_->ReadEmissionSpectrumInput(
          nodes_, matrix_count_, parameters_->spectrum_initial_density_matrix);
      pthread_mutex_unlock(&io_lock_);
    } else {
      // Read the initial density vector from the input parameters.
      for (int m = 0; m < matrix_count_; ++m) {
        Float scale = 1;
        if (parameters_->is_rho_normalized) {
          for (int i = 0; i < num_correlation_fn_terms_; i++) {
            scale *=
                sqrt(PartialFactorial(1, nodes_[m].index[i]) *
                     pow(abs_correlation_fn_constants_[i], nodes_[m].index[i]));
          }
        }
        for (int i = 0; i < num_elements_; i++) {
          nodes_[m].density_matrix[i] =
              parameters_->spectrum_initial_density_matrix[i] / scale;
        }
      }
    }
  }
  Log(3, id, "\tWriting initial state to output file.");
  pthread_mutex_lock(&io_lock_);
  outfile_ << parameters_->start_time << " ";
  phi_size_t num_elements =
      parameters_->is_spectrum_calculation ? num_states_ : num_elements_;
  for (int i = 0; i < num_elements; ++i) {
    outfile_ << nodes_[0].density_matrix[i] << " ";
  }
  outfile_ << "\n" << std::flush;
  pthread_mutex_unlock(&io_lock_);
}

void HierarchyIntegrator::InitializeBathCouplingOperators(int id) {
  // TODO(johanstr): Need to promote indep/diag calculation to full.
  Log(1, id, "Initializing bath coupling operators.");
  const int hilbert_space_size = parameters_->hilbert_space_size;
  const int num_hilbert_space_elements =
      parameters_->hilbert_space_num_elements;
  if (parameters_->is_full_bath_coupling) {
    // M System-Bath coupling operators.
    Log(2, id, "\tAssigning full bath coupling operators.");
    for (int m = id; m < num_bath_coupling_terms_;
         m += parameters_->num_threads) {
      full_bath_operators_[m] = new Complex[num_hilbert_space_elements];
      // Point integration bath_operators to full bath operators.
      bath_operators_[m] = full_bath_operators_[m];
      SetElementsToZero(num_hilbert_space_elements, full_bath_operators_[m]);
      Copy(num_hilbert_space_elements,
           &(parameters_->bath_coupling_op[m * num_hilbert_space_elements]),
           full_bath_operators_[m]);
    }
    if (parameters_->is_truncated_system) {
      Log(2, id, "\tAssigning truncated full bath coupling operators.");
      for (int m = id; m < num_bath_coupling_terms_;
           m += parameters_->num_threads) {
        bath_operators_[m] = new Complex[num_elements_];
        truncator_th_[id]->Truncate(full_bath_operators_[m],
                                    bath_operators_[m]);
      }
    }
  } else if (parameters_->is_diagonal_bath_coupling) {
    Log(2, id, "\tAssigning diagonal bath coupling operators.");
    for (int m = id; m < num_bath_coupling_terms_;
         m += parameters_->num_threads) {
      bath_operators_[m] = new Complex[hilbert_space_size];
      SetElementsToZero(hilbert_space_size, bath_operators_[m]);
      Copy(hilbert_space_size,
           &(parameters_->bath_coupling_op[m * hilbert_space_size]),
           bath_operators_[m]);
    }
  }
}

/*
 * Constructs pointer-linked hierarchy.
 */
void HierarchyIntegrator::LinkHierarchyNodes(HierarchyNode *nodes, int id,
                                             int num_threads,
                                             Complex *density_matrices) {
  Log(1, id, "Linking hierarchy nodes.");
  const int num_states = parameters_->num_states;
  const int num_bath_couplings = parameters_->num_bath_couplings;
  const int num_matsubara_terms = parameters_->num_matsubara_terms;
  const bool is_multi_bath = parameters_->is_multiple_independent_baths;
  const int *bath_indices = parameters_->multiple_independent_bath_indices;
  const bool is_time_local = parameters_->use_time_local_truncation;
  const int hierarchy_truncation = parameters_->hierarchy_truncation_level;
  const int num_elements =
      parameters_->is_spectrum_calculation ? num_states : num_elements_;

  MultiIndex multi_index(num_correlation_fn_terms_);
  int prev_count = 0;
  int n_fnd;
  map<MultiIndex, int, CompareMultiIndex>::iterator found_result;
  HierarchyNode **nodes_ptr;
  int *index;

  index = new int[num_correlation_fn_terms_];
  nodes_ptr = new HierarchyNode *[num_correlation_fn_terms_];
  int i = 0;
  for (int node_index = 0; node_index < node_count_th_[id]; ++node_index) {
    i = node_indices_th_[id][node_index];
    Complex *density_matrix =
        density_matrices != NULL ? &(density_matrices[i * num_elements]) : NULL;

    // Initialize each auxiliary operator node.
    nodes[i].CreateVec(num_states, num_bath_couplings, num_matsubara_terms,
                       multi_index_map_[i].index_array, density_matrix);
    nodes[i].id = i;
    nodes[i].num_elements = num_elements_;
    nodes[i].is_timelocal_truncated = false;

    if (is_multi_bath) {
      for (int m = 0; m < num_bath_couplings; ++m) {
        nodes[i].matsubara_node_index[m] = bath_indices[m];
      }
    } else {
      for (int m = 0; m < num_bath_couplings; ++m) {
        nodes[i].matsubara_node_index[m] = m;
      }
    }
  }

  BarrierWait(&integrator_threads_barrier_);
  const int kLinkOutputVerbosity = 4;
  ostringstream ss;
  for (int ii = 0; ii < node_count_th_[id]; ++ii) {
    i = node_indices_th_[id][ii];
    prev_count = 0;
    ss << "[" << id << "] " << i << ": " << nodes[i].GetIndexString() << "\n";
    for (int j = 0; j < num_correlation_fn_terms_; ++j) {
      if (nodes[i].index[j] > 0) {
        ++prev_count;
      }
    }

    // Previous level in the hierarchy.
    if (prev_count > 0) {
      Log(kLinkOutputVerbosity, id, ss.str());
      ss << "[" << id << "]\tLinking Prev Matrices (" << prev_count << ").\n";
      int prev_assigned = 0;
      for (int j = 0; j < num_correlation_fn_terms_; ++j) {
        multi_index = multi_index_map_[i];
        if (multi_index[j] > 0) {
          multi_index.Dec(j + 1);
          found_result = multi_index_map_lookup_.find(multi_index);
          if (found_result != multi_index_map_lookup_.end()) {
            n_fnd = found_result->second;
            nodes_ptr[prev_assigned] = &nodes[n_fnd];
            index[prev_assigned] = j;
            ss << "\t" << j << ": " << nodes[n_fnd].GetIndexString() << " - "
               << nodes_ptr[prev_assigned]->GetIndexString() << "\n";
            ++prev_assigned;
          }
        }
      }
      nodes[i].num_prev_hierarchy_nodes = prev_assigned;
      nodes[i].prev_hierarchy_nodes = new HierarchyNode *[prev_assigned];
      nodes[i].prev_node_index = new int[prev_assigned];
      nodes[i].prev_bath_coupling_op_index = new int[prev_assigned];
      for (int j = 0; j < prev_assigned; ++j) {
        nodes[i].prev_hierarchy_nodes[j] = nodes_ptr[j];
        nodes[i].prev_node_index[j] = index[j];
        nodes[i].prev_bath_coupling_op_index[j] =
            bath_indices[(index[j] - index[j] % num_matsubara_terms) /
                         num_matsubara_terms];
      }
    }

    // Next level in the hierarchy.
    if (nodes[i].hierarchy_truncation_level < hierarchy_truncation - 1) {
      int next_assigned = 0;
      ss << "[" << id << "]\tLinking Next Matrices.\n";
      for (int j = 0; j < num_correlation_fn_terms_; ++j) {
        multi_index = multi_index_map_[i];
        multi_index.Inc(j + 1);
        index[next_assigned] = j;
        found_result = multi_index_map_lookup_.find(multi_index);
        if (found_result != multi_index_map_lookup_.end()) {
          n_fnd = found_result->second;
          nodes_ptr[j] = &nodes[n_fnd];
          ss << "\t" << j << ": " << nodes[n_fnd].GetIndexString() << " - "
             << nodes_ptr[next_assigned]->GetIndexString() << "\n";
          ++next_assigned;
        }
      }
      if (next_assigned > 0) {
        nodes[i].num_next_hierarchy_nodes = next_assigned;
        nodes[i].next_hierarchy_nodes = new HierarchyNode *[next_assigned];
        nodes[i].next_node_index = new int[next_assigned];
        nodes[i].next_coupling_op_index = new int[next_assigned];
        for (int j = 0; j < next_assigned; ++j) {
          nodes[i].next_hierarchy_nodes[j] = nodes_ptr[j];
          nodes[i].next_node_index[j] = index[j];
          nodes[i].next_coupling_op_index[j] =
              bath_indices[(index[j] - index[j] % num_matsubara_terms) /
                           num_matsubara_terms];
        }
        nodes[i].is_timelocal_truncated = false;
      } else {
        nodes[i].is_timelocal_truncated = is_time_local;
      }
    } else if (nodes[i].hierarchy_truncation_level ==
               hierarchy_truncation - 1) {
      nodes[i].is_timelocal_truncated = is_time_local;
    }
    Log(kLinkOutputVerbosity, id, ss.str());
    ss.clear();
  }
  BarrierWait(&integrator_threads_barrier_);
}

/*
 * Initializes the prefactors for the different terms in the HEOM.
 */
void HierarchyIntegrator::InitializePrefactors(const int id) {
  Log(2, id, "Initializing prefactors for HEOM integration.");
  const bool is_spectrum = parameters_->is_spectrum_calculation;
  const bool is_normalized = parameters_->is_rho_normalized;
  const bool is_restarted = parameters_->is_restarted;
  const int num_bath_couplings = parameters_->num_bath_couplings;
  const Float hbar = parameters_->hbar;
  const Float *gamma = parameters_->gamma;
  const Float *lambda = parameters_->lambda;
  const Float thermal_energy =
      parameters_->boltzmann_constant * parameters_->temperature;
  const Float nu = 2 * PI * thermal_energy / hbar;

  int nm = 0;
  int i = 0;
  for (int ii = 0; ii < node_count_th_[id]; ++ii) {
    // Create and zero all prefactors.
    i = node_indices_th_[id][ii];
    nodes_[i].dephasing_prefactor = kZero;
    nodes_[i].matsubara_prefactor = new Complex[num_bath_couplings];
    if (nodes_[i].num_next_hierarchy_nodes > 0) {
      nodes_[i].next_prefactor =
          new Complex[nodes_[i].num_next_hierarchy_nodes];
    }
    for (int j = 0; j < num_bath_couplings; ++j) {
      nodes_[i].matsubara_prefactor[j] = kZero;
    }
    if (nodes_[i].num_prev_hierarchy_nodes > 0) {
      nodes_[i].prev_prefactor_row =
          new Complex[nodes_[i].num_prev_hierarchy_nodes];
      nodes_[i].prev_prefactor_col =
          new Complex[nodes_[i].num_prev_hierarchy_nodes];
    }
    for (int j = 0; j < nodes_[i].num_prev_hierarchy_nodes; ++j) {
      nodes_[i].prev_prefactor_row[j] = kZero;
      nodes_[i].prev_prefactor_col[j] = kZero;
    }

    // Prefactor for dephasing in the same level.
    for (int m = 0; m < num_correlation_fn_terms_; ++m) {
      bool is_matsubara_term = m % matsubara_truncation_ == 0;
      int k = is_matsubara_term ? m / matsubara_truncation_
                                : m % matsubara_truncation_;
      Float nu_ak = is_matsubara_term ? gamma[k] : nu * k;
      nodes_[i].dephasing_prefactor += Complex(nodes_[i].index[m] * nu_ak);
    }

    // Prefactors for Matsubara correction.
    for (int j = 0; j < num_bath_couplings; ++j) {
      if (gamma[j] > 0) {
        nodes_[i].matsubara_prefactor[j] = Complex(
            2 * lambda[j] * thermal_energy / hbar / hbar / gamma[j] -
                lambda[j] / hbar /
                    tan(hbar * gamma[j] / (2 * thermal_energy)),
            0);
        for (int k = 1; k < matsubara_truncation_; ++k) {
          nodes_[i].matsubara_prefactor[j] -=
              Complex(4 * lambda[j] * thermal_energy / hbar / hbar *
                          gamma[j] / (pow(nu * k, 2) - pow(gamma[j], 2)),
                      0);
        }
      }
    }

    // Prefactors for coupling to density operators in the next hierarchy level.
    for (int j = 0; j < nodes_[i].num_next_hierarchy_nodes; j++) {
      // i
      nodes_[i].next_prefactor[j] = kImaginary;
      if (is_normalized) {
        // With normalization, increasing hierarchy levels have smaller
        // auxiliary density matrix elements.
        nm = nodes_[i].next_node_index[j];
        Complex normalization = Complex(sqrt(
            (nodes_[i].index[nm] + 1) * abs_correlation_fn_constants_[nm]));
        // -i sqrt((n_ak + 1) |c_ak|)
        nodes_[i].next_prefactor[j] *= normalization;
      }
      if (is_spectrum && is_restarted) {
        // For emission spectrum calculations.
        nodes_[i].next_prefactor[j] *= kNegativeOne;
      }
    }

    // Prefactors for coupling to density operators in the previous hierarchy
    // level.
    for (int j = 0; j < nodes_[i].num_prev_hierarchy_nodes; ++j) {
      nm = nodes_[i].prev_node_index[j];
      // i n_ak c_ak
      nodes_[i].prev_prefactor_row[j] =
          Complex(0, nodes_[i].index[nm]) * correlation_fn_constants_[nm];
      // -i n_ak c_ak^*
      nodes_[i].prev_prefactor_col[j] = Complex(0, -nodes_[i].index[nm]) *
                                        conj(correlation_fn_constants_[nm]);

      if (is_normalized) {
        Complex normalization =
            kOne / Complex(sqrt(nodes_[i].index[nm] *
                                 abs_correlation_fn_constants_[nm]));
        // i n_ak c_ak sqrt(n_ak / |c_ak|)
        nodes_[i].prev_prefactor_row[j] *= normalization;
        // -i n_ak c_ak^* sqrt(n_ak / |c_ak|)
        nodes_[i].prev_prefactor_col[j] *= normalization;
      }
      // Check for emission spectrum calculation:
      if (is_spectrum && is_restarted) {
        nodes_[i].prev_prefactor_row[j] =
            kNegativeImaginary * nodes_[i].prev_prefactor_col[j];
      }
    }
    // Assign the copy of the Hamiltonian matrix nearest in memory to hierarchy
    // node i.
    nodes_[i].hamiltonian = hamiltonian_th_[id];
    nodes_[i].hamiltonian_adjoint = hamiltonian_adjoint_th_[id];
    if (nodes_tmp_ != NULL) {
      nodes_tmp_[i].dephasing_prefactor = nodes_[i].dephasing_prefactor;
      nodes_tmp_[i].next_prefactor = nodes_[i].next_prefactor;
      nodes_tmp_[i].matsubara_prefactor = nodes_[i].matsubara_prefactor;
      nodes_tmp_[i].prev_prefactor_row = nodes_[i].prev_prefactor_row;
      nodes_tmp_[i].prev_prefactor_col = nodes_[i].prev_prefactor_col;
      nodes_tmp_[i].hamiltonian = nodes_[i].hamiltonian;
      nodes_tmp_[i].hamiltonian_adjoint = nodes_[i].hamiltonian_adjoint;
    }
  }
}

/*
 * Prepare steady-state calculation matrices and parameters.
 */
void HierarchyIntegrator::PrepForSteadyState(int id, int num_threads,
                                             RunMethod run_method) {
  Log(-1, id, " Initializing steady state calculation.");
  const int num_states = parameters_->num_states;
  const bool evecs_present = parameters_->are_e_vecs_present;
  const bool evals_present = parameters_->are_e_vals_present;
  const Complex *eval = parameters_->hamiltonian_eigenvalues;
  const Complex *evec = parameters_->hamiltonian_eigenvectors;
  const int bicgstab_ell = parameters_->bicgstab_ell;
  const int *indep_indices = parameters_->multiple_independent_bath_indices;

  for (int i = id; i < matrix_count_; i += num_threads) {
    nodes_[i].DetachDensityMatrix();
    if (run_method == BICGSTAB || run_method == BICGSTABU) {
      nodes_[i].BicgstabInit();
    } else if (run_method == BICGSTABL) {
      nodes_[i].BicgstablInit(bicgstab_ell);
    }
  }

  BarrierWait(&integrator_threads_barrier_);
  // Create Liouville Space operator
  // for system density matrix Hamiltonian
  if (id == 0) {
    Complex *liouville_hamiltonian;
    liouville_hamiltonian = ToLiouville(hamiltonian_, num_states);
    int n;
    for (int i = 0; i < num_elements_ * num_elements_; ++i) {
      liouville_hamiltonian[i] *= liouville_prefactor_;
    }
    for (int i = 0; i < num_states; ++i) {
      for (int ii = 0; ii < num_states; ++ii) {
        for (int j = 0; j < num_bath_coupling_terms_; ++j) {
          n = indep_indices[j];
          if ((i == n && ii != n) || (i != n && ii == n)) {
            const phi_big_size_t idx =
                (i * num_states + ii) * num_elements_ + i * num_states + ii;
            liouville_hamiltonian[idx] -= nodes_[0].matsubara_prefactor[j];
          }
        }
      }
    }
    nodes_[0].same_liouville = new Complex[num_elements_ * num_elements_];
    for (int i = 0; i < num_elements_ * num_elements_; ++i) {
      nodes_[0].same_liouville[i] = liouville_hamiltonian[i];
    }
  }
  // Create Liouville space operator
  // for operations pointing to system
  // density matrix
  const int num_prev_entries = 2 * num_states - 1;
  int **prev_indices;
  prev_indices = new int *[num_bath_coupling_terms_];
  int n, m;
  for (int nj = 0; nj < num_bath_coupling_terms_; ++nj) {
    n = 0;
    m = indep_indices[nj];
    prev_indices[nj] = new int[num_prev_entries];
    for (int i = 0; i < m; ++i) {
      prev_indices[nj][n] = i * num_states + m;
      ++n;
    }
    for (int i = 0; i < m; ++i) {
      prev_indices[nj][n] = m * num_states + i;
      ++n;
    }
    prev_indices[nj][n] = m * num_states + m;
    ++n;
    for (int i = m + 1; i < num_states; ++i) {
      prev_indices[nj][n] = m * num_states + i;
      ++n;
    }
    for (int i = m + 1; i < num_states; ++i) {
      prev_indices[nj][n] = i * num_states + m;
      ++n;
    }
  }
  for (int i = id; i <= hierarchy_level_[1]; i += num_threads) {
    nodes_[i].CreatePrevLiouvilleOperator(num_prev_entries);
    nodes_[i].next_liouville_index = prev_indices;
  }

  // Poor guess of a steady-state matrix
  if (id == 0) {
    Log(1, id, "\tGuessing steady-state density matrix.");
    if (evals_present && evecs_present) {
      Log(1, id, "\tEigenvalue and vectors present.");
      Complex *populations;
      populations = new Complex[num_states];
      Complex z = 0;
      for (int i = 0; i < num_states; ++i) {
        populations[i] = exp(-eval[i] / thermal_energy_);
        z += populations[i];
      }
      for (int i = 0; i < num_states; ++i) {
        populations[i] /= z;
      }
      Complex *a;
      a = new Complex[num_elements_];
      for (int i = 0; i < num_elements_; ++i) {
        a[i] = 0;
      }
      for (int i = 0; i < num_states; ++i) {
        blas::Her(num_states, real(populations[i]),
                  &(evec[i * num_states]), a);
      }

      int m = 0;
      for (int i = 0; i < num_states; ++i) {
        nodes_[m].steady_state_density_matrix[i * num_states + i] =
            a[i * num_states + i];
        for (int j = i + 1; j < num_states; ++j) {
          nodes_[m].steady_state_density_matrix[j * num_states + i] =
              conj(a[i * num_states + j]);
          nodes_[m].steady_state_density_matrix[i * num_states + j] =
              a[i * num_states + j];
        }
      }
    } else {
      int m = 0;
      for (int i = 0; i < num_states; ++i) {
        nodes_[m].steady_state_density_matrix[i * num_states + i] =
            Complex(1. / num_states);
      }
    }

    // Write out initial guess
    if (verbose_ > 2) {
      pthread_mutex_lock(&io_lock_);
      std::cout << "Initial Guess of Steady State:\n";
      PrintMatrix(nodes_[0].steady_state_density_matrix, num_states);
      pthread_mutex_unlock(&io_lock_);
    }
    pthread_mutex_lock(&io_lock_);
    outfile_ << "#Initial rho:\n#";
    for (int i = 0; i < num_states * num_states; ++i) {
      outfile_ << nodes_[0].steady_state_density_matrix[i] << " ";
    }
    outfile_ << std::endl;
    pthread_mutex_unlock(&io_lock_);
  }
  BarrierWait(&integrator_threads_barrier_);
}

/*
 * Update the Hamiltonian for specified time.
 */
bool HierarchyIntegrator::UpdateTimeDependentHamiltonian(Float time, int id) {
  if (parameters_->time_indepenent_hamiltonian) {
    // Hamiltonian is time independent.
    return false;
  }
  const Float alpha = time / parameters_->integration_time;
  const Complex initial_field = initial_field_spline_th_->Get(alpha);
  const Complex final_field = final_field_spline_th_->Get(alpha);
  const Float initial_field_multiplier_change_frac =
      fabs(real(initial_field) / old_initial_field_value_th_[id] - 1.f);
  const Float final_field_multiplier_change_frac =
      fabs(real(final_field) / old_final_field_value_th_[id] - 1.f);
  const Float minimum_field_change = parameters_->minimum_field_change;
  if (initial_field_multiplier_change_frac < minimum_field_change &&
      final_field_multiplier_change_frac < minimum_field_change) {
    // Don't update the Hamiltonian for small changes in the multipliers.
    return false;
  }
  old_initial_field_value_th_[id] = real(initial_field);
  old_final_field_value_th_[id] = real(final_field);
  if (id == 0) {
    const Complex *final_hamiltonian = parameters_->final_annealing_hamiltonian;
    const Complex *initial_hamiltonian =
        parameters_->initial_annealing_hamiltonian;
    for (phi_size_t i = 0; i < parameters_->hilbert_space_num_elements; ++i) {
      full_hamiltonian_th_[id][i] = initial_field * initial_hamiltonian[i] +
                                    final_field * final_hamiltonian[i];
    }
    if (parameters_->is_truncated_system) {
      truncator_th_[id]->UpdateWith(full_hamiltonian_th_[id]);
      truncator_th_[id]->Truncate(full_hamiltonian_th_[id],
                                  hamiltonian_th_[id]);
    }
  }
  BarrierWait(&integrator_threads_barrier_);
  if (id != 0) {
    // Spread the knowledge to other threads.
    if (parameters_->is_truncated_system) {
      truncator_th_[id]->UpdateFrom(truncator_th_[0]);
    }
    blas::Copy(
        num_states_ * num_states_, hamiltonian_th_[0], hamiltonian_th_[id]);
  }
  if (parameters_->is_truncated_system &&
      (parameters_->hierarchy_truncation_level > 1 ||
       parameters_->use_time_local_truncation)) {
    // Update the bath coupling operators for open systems.
    for (phi_size_t m = id; m < parameters_->num_bath_couplings;
         m += parameters_->num_threads) {
      truncator_th_[id]->Rotate(bath_operators_[m]);
    }
  }
  ConjugateTranspose(num_states_, hamiltonian_th_[id],
                     hamiltonian_adjoint_th_[id]);
  return true;
}

void HierarchyIntegrator::UpdateTimeLocalTruncation(const int id,
                                                    const Float time,
                                                    const Float timestep) {
  if (!parameters_->use_time_local_truncation || timestep < 1e-9) {
    return;
  }

  FillPropagator(timestep, hamiltonian_th_[id], hamiltonian_propagator_th_[id],
                 hamiltonian_propagator_adjoint_th_[id]);
  // Iterate over system-bath coupling terms
  for (int idx = id; idx < num_bath_coupling_terms_;
       idx += parameters_->num_threads) {
    UpdateTimeLocalTruncationMatrices(
        timestep, idx, hamiltonian_propagator_th_[id],
        hamiltonian_propagator_adjoint_th_[id],
        time_local_truncation_terms_th_[idx], time_local_matrices_[idx],
        time_local_adjoint_matrices_[idx]);
  }
}

void HierarchyIntegrator::StoreTimeLocalTruncationMatrices(const int id) {
  if (!parameters_->use_time_local_truncation) {
    return;
  }
  const int num_threads = parameters_->num_threads;
  // Iterate over system-bath coupling terms
  for (int m = id; m < num_bath_coupling_terms_; m += num_threads) {
    for (int i = 0; i < num_elements_; ++i) {
      time_local_matrices_store_[m][i] = time_local_matrices_[m][i];
    }
    for (int k = 0; k < matsubara_truncation_; ++k) {
      for (int i = 0; i < num_elements_; ++i) {
        time_local_truncation_store_th_[m][k][i] =
            time_local_truncation_terms_th_[m][k][i];
      }
    }
  }
}

void HierarchyIntegrator::RestoreTimeLocalTruncationMatrices(const int id) {
  if (!parameters_->use_time_local_truncation) {
    return;
  }
  const int num_threads = parameters_->num_threads;
  // Iterate over system-bath coupling terms
  for (int m = id; m < num_bath_coupling_terms_; m += num_threads) {
    for (int i = 0; i < num_elements_; ++i) {
      time_local_matrices_[m][i] = time_local_matrices_store_[m][i];
    }
    for (int k = 0; k < matsubara_truncation_; ++k) {
      for (int i = 0; i < num_elements_; ++i) {
        time_local_truncation_terms_th_[m][k][i] =
            time_local_truncation_store_th_[m][k][i];
      }
    }
    ConjugateTranspose(num_states_, time_local_matrices_[m],
                       time_local_adjoint_matrices_[m]);
  }
}

/*
 * Determines which hierarchy updater to use for the integration
 */
hierarchy_updater::HierarchyUpdater *HierarchyIntegrator::GetUpdater(
    int id, const Complex *same, const Complex *next, const Complex *prev) {
  Log(1, id, "Getting hierarchy updater");
  if (parameters_->hierarchy_truncation_level == 1 &&
      !parameters_->use_time_local_truncation) {
    return new hierarchy_updater::Vacuum(parameters_, liouville_prefactor_);
  }
  if (parameters_->hierarchy_truncation_level == 1) {
    if (parameters_->is_full_bath_coupling) {
      return new hierarchy_updater::NoHierarchyFullBath(
          parameters_, liouville_prefactor_, time_local_matrices_,
          time_local_adjoint_matrices_, same, next);
    }
    if (parameters_->is_diagonal_bath_coupling) {
      return new hierarchy_updater::NoHierarchyDiagonalBath(
          parameters_, liouville_prefactor_, time_local_matrices_,
          time_local_adjoint_matrices_, same, next);
    }
    return new hierarchy_updater::NoHierarchyIndependentBath(
        parameters_, liouville_prefactor_, time_local_matrices_,
        time_local_adjoint_matrices_);
  }
  if (parameters_->is_spectrum_calculation) {
    // if (parameters_->is_full_bath_coupling) {
    // TODO(johanstr): Add full bath method for spectrum calculation.
    // }
    if (parameters_->is_diagonal_bath_coupling) {
      return new hierarchy_updater::SpectrumDiagonalBath(
          parameters_, liouville_prefactor_, time_local_matrices_,
          time_local_adjoint_matrices_, same, next, prev);
    }
    return new hierarchy_updater::SpectrumIndependentBath(
        parameters_, liouville_prefactor_, time_local_matrices_,
        time_local_adjoint_matrices_);
  }
  if (parameters_->is_full_bath_coupling) {
    return new hierarchy_updater::FullBath(
        parameters_, liouville_prefactor_, time_local_matrices_,
        time_local_adjoint_matrices_, same, next, prev);
  }
  if (parameters_->is_diagonal_bath_coupling) {
    return new hierarchy_updater::DiagonalBath(
        parameters_, liouville_prefactor_, time_local_matrices_,
        time_local_adjoint_matrices_, same, next, prev);
  }
  if (parameters_->filter_tolerance > 0) {
    return new hierarchy_updater::IndependentBathAdaptiveTruncation(
        parameters_, liouville_prefactor_, time_local_matrices_,
        time_local_adjoint_matrices_);
  }
  return new hierarchy_updater::IndependentBath(
      parameters_, liouville_prefactor_, time_local_matrices_,
      time_local_adjoint_matrices_);
}

/*
 * Integrate the HEOM using the Runga-Kutta-Fehlberg 4/5 adaptive timestep
 * method.
 */
void HierarchyIntegrator::Rkf45Integrate(int id, int num_threads) {
  hierarchy_updater::HierarchyUpdater *updater = NULL;

  const bool restart = parameters_->restart_output_time > 0;
  const bool is_spectrum_calculation = parameters_->is_spectrum_calculation;
  const bool is_diagonal_coupling = parameters_->is_diagonal_bath_coupling;
  const bool is_truncated_system = parameters_->is_truncated_system;
  const bool is_time_dependent_hamiltonian =
      parameters_->time_indepenent_hamiltonian == NULL;
  const Float filter_tolerance = parameters_->filter_tolerance;
  // Time Checkers
  time_t t1, t2, last_restart_write_time, now;
  int *m_start;
  int *m_end;
  int *m;
  // Memory assigned by thread.
  if (id != num_threads) {
    // Pointer to first hierarchy matrix this thread is responsible for.
    m_start = node_indices_th_[id];
    // Pointer to last hierarchy matrix this thread is responsible for.
    m_end = node_indices_th_[id] + node_count_th_[id];
  } else {
    m_start = new int;
    *m_start = 0;
    m_end = new int;
    *m_end = matrix_count_;
  }

  Complex *k0, *k1, *k2, *k3, *k4, *k5;

  Complex *next_bath_coupling = NULL;
  Complex *prev_bath_coupling = NULL;
  Complex *matsubara_coupling = NULL;

  // Adaptive timestepping variables.
  Float mydt = parameters_->timestep;
  Float diff = 0, max_diff = 0;
  Complex y_rkf45;

  // Keep track of integration time.
  const Float end_time = parameters_->integration_time;
  const Float start_time = parameters_->start_time;
  Float time_now = parameters_->start_time;
  Float next_write_time = time_now;
  // Set time resolution in output file
  Float output_min_timestep = parameters_->output_minimum_timestep;

  // Multiplicative factor to increase step size.
  Complex dtup = 1.1;
  // Multiplicative factor to decrease step size.
  Complex dtdn = 0.5;

  rkf45_diff_th_[id] = 0.0;
  Float tolerance = parameters_->rkf45_tolerance;
  Float mindt = parameters_->rkf45_minimum_timestep;

  // If we have a filter tolerance enable adaptive hierarchy truncation.
  const bool use_adaptive_truncation = filter_tolerance > 0;

  BarrierWait(&all_threads_barrier_);

  const int num_matrix_elements =
      is_spectrum_calculation ? num_states_ : num_elements_;
  const int num_entries = num_matrix_elements * node_count_th_[id];
  if (id < num_threads) {
    Log(1, id, "\tAllocating memory for RKF45 updates.");
    // TODO(johanstr): Extract to function.
    k0 = new Complex[num_entries];
    k1 = new Complex[num_entries];
    k2 = new Complex[num_entries];
    k3 = new Complex[num_entries];
    k4 = new Complex[num_entries];
    k5 = new Complex[num_entries];
    for (int i = 0; i < node_count_th_[id]; ++i) {
      nodes_[node_indices_th_[id][i]].k0 = k0 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k1 = k1 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k2 = k2 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k3 = k3 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k4 = k4 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k5 = k5 + i * num_matrix_elements;
      for (int j = 0; j < num_matrix_elements; ++j) {
        k0[i * num_matrix_elements + j] = kZero;
        k1[i * num_matrix_elements + j] = kZero;
        k2[i * num_matrix_elements + j] = kZero;
        k3[i * num_matrix_elements + j] = kZero;
        k4[i * num_matrix_elements + j] = kZero;
        k5[i * num_matrix_elements + j] = kZero;
      }
    }
    if (is_diagonal_coupling) {
      Log(1, id, "\tAllocating memory for correlated bath operators.");
      matsubara_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      next_bath_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      prev_bath_coupling =
          new Complex[num_correlation_fn_terms_ * num_matrix_elements];
    } else if (parameters_->is_full_bath_coupling) {
      Log(1, id, "\tAllocating memory for full bath operators.");
      matsubara_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      next_bath_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      prev_bath_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
    }
    if (is_diagonal_coupling || parameters_->is_full_bath_coupling) {
      Log(2, id, "\tFilling bath operators.");
      FillBathOperators(matsubara_coupling, next_bath_coupling,
                        prev_bath_coupling);
    }
    ostringstream s;
    s << "\tMemory for " << node_count_th_[id] << " density matrices (each with"
      << " " << num_entries << " elements) assigned.";
    Log(0, id, s.str());
    updater = GetUpdater(id, matsubara_coupling, next_bath_coupling,
                         prev_bath_coupling);
    BarrierWait(&integrator_threads_barrier_);
    Log(-1, id, "\tIntegrating using adaptive timestepping RKF45.");
  } else {
    // id == numthreads, the writer.
    Log(1, id, "\tWriter waiting.");
  }

  BarrierWait(&all_threads_barrier_);
  time(&t1);
  int i = -1;
  int num_accepted_steps = 0;
  int inext = 0;
  // RKF45 Butcher Tableau
  // 0    |
  // 2/9  |  2/9
  // 1/3  |  1/12     1/4
  // 3/4  | 69/128 -243/128  135/64
  // 1    |-17/12    27/4    -27/5   16/15
  // 5/6  | 65/432   -5/16    13/16   4/27   5/144
  // ---------------------------------------------------
  // O(4) |  1/9      0        9/20  16/45   1/12
  // O(5) | 47/450    0       12/25  32/225  1/30  6/23
  // Make thread-local copies of the integration constants
  Float a1(2. / 9), a2(1. / 3), a3(3. / 4), a4(1.0), a5(5. / 6);
  Complex b10(2. / 9.);
  Complex b20(1. / 12.), b21(1. / 4.);
  Complex b30(69. / 128.), b31(-243. / 128.), b32(135. / 64.);
  Complex b40(-17. / 12.), b41(27. / 4.), b42(-27. / 5), b43(16. / 15);
  Complex b50(65. / 432.), b51(-5. / 16), b52(13. / 16.), b53(4. / 27.),
      b54(5. / 144.);
  Complex c50(1. / 9), c52(9. / 20), c53(16. / 45), c54(1. / 12);
  Complex c60(47. / 450), c62(12. / 25), c63(32. / 225), c64(1. / 30),
      c65(6. / 25);
  a1 *= mydt;
  a2 *= mydt;
  a3 *= mydt;
  a4 *= mydt;
  a5 *= mydt;
  b10 *= dt_c_;
  b20 *= dt_c_;
  b21 *= dt_c_;
  b30 *= dt_c_;
  b31 *= dt_c_;
  b32 *= dt_c_;
  b40 *= dt_c_;
  b41 *= dt_c_;
  b42 *= dt_c_;
  b43 *= dt_c_;
  b50 *= dt_c_;
  b51 *= dt_c_;
  b52 *= dt_c_;
  b53 *= dt_c_;
  b54 *= dt_c_;
  c50 *= dt_c_;
  c52 *= dt_c_;
  c53 *= dt_c_;
  c54 *= dt_c_;
  c60 *= dt_c_;
  c62 *= dt_c_;
  c63 *= dt_c_;
  c64 *= dt_c_;
  c65 *= dt_c_;
  timer::Timer *timer = new timer::Timer(7);
  timer->Reset();
  // Initialize the restart output time.
  time(&last_restart_write_time);
  while (time_now < end_time) {
    // The integration loop.
    i += 1;

    if (id < num_threads) {
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 0);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 0);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 0);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 0);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 0);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 0);
      }

      /////////////////
      //      k0     //
      /////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 0);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_[*m], nodes_[*m].k0);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 0);
      // No barrier needed since rho_mid is not used in the prior step.
      RECORD_START(timer, timer::Type::kRhoUpdate, 0);
#ifdef BLASUPDATE
      blas::Copy(num_entries, density_matrices_th_[id],
                 density_matrices_tmp_th_[id]);
      blas::Axpy(num_entries, &b10, k0, 1, density_matrices_tmp_th_[id], 1);
#else
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k0[j] * b10;
      }
#endif
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 0);
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 0);
      UpdateTimeLocalTruncation(id, time_now, a1);  // To Q_tl(t + a1).
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 0);
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 1);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + a1, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 1);
        // Rotate to new space.
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 1);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 1);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 1);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 1);
      }
      /////////////////

      /////////////////
      //      k1     //
      /////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 1);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k1);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 1);
      // Wait for k1 updated.
      BarrierWait(&integrator_threads_barrier_);
      RECORD_START(timer, timer::Type::kRhoUpdate, 1);
#ifdef BLASUPDATE
      blas::Copy(num_entries, density_matrices_th_[id],
                 density_matrices_tmp_th_[id]);
      blas::Axpy(num_entries, &b20, k0, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b21, k1, 1, density_matrices_tmp_th_[id], 1);
#else
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k0[j] * b20 + k1[j] * b21;
      }
#endif
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 1);
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 1);
      UpdateTimeLocalTruncation(id, time_now + a1, a2 - a1);  // To Q_tl(t + a1)
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 1);
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 2);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + a2, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 2);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 2);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
            truncator_th_[id]->Rotate(nodes_[*m].k1);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 2);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 2);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 2);
      }
      /////////////////

      /////////////////
      //     k2      //
      /////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 2);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k2);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 2);
      // Wait for k2 updated
      BarrierWait(&integrator_threads_barrier_);
      RECORD_START(timer, timer::Type::kRhoUpdate, 2);
#ifdef BLASUPDATE
      blas::Copy(num_entries, density_matrices_th_[id],
                 density_matrices_tmp_th_[id]);
      blas::Axpy(num_entries, &b30, k0, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b31, k1, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b32, k2, 1, density_matrices_tmp_th_[id], 1);
#else
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] = density_matrices_th_[id][j] +
                                          k0[j] * b30 + k1[j] * b31 +
                                          k2[j] * b32;
      }
#endif
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 2);
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 2);
      UpdateTimeLocalTruncation(id, time_now + a2,
                                a3 - a2);  // To Q_tl(t + a3).
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 2);
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 3);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + a3, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 3);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 3);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
            truncator_th_[id]->Rotate(nodes_[*m].k1);
            truncator_th_[id]->Rotate(nodes_[*m].k2);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 3);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 3);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 3);
      }
      //////////////////

      //////////////////
      //      k3      //
      //////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 3);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k3);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 3);
      // Wait for k3 updated.
      BarrierWait(&integrator_threads_barrier_);
      RECORD_START(timer, timer::Type::kRhoUpdate, 3);
#ifdef BLASUPDATE
      blas::Copy(num_entries, density_matrices_th_[id],
                 density_matrices_tmp_th_[id]);
      blas::Axpy(num_entries, &b40, k0, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b41, k1, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b42, k2, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b43, k3, 1, density_matrices_tmp_th_[id], 1);
#else
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] = density_matrices_th_[id][j] +
                                          k0[j] * b40 + k1[j] * b41 +
                                          k2[j] * b42 + k3[j] * b43;
      }
#endif
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 3);
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 3);
      UpdateTimeLocalTruncation(id, time_now + a3,
                                a4 - a3);  // To Q_tl(t + a3).
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 3);
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 4);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + a4, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 4);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 4);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
            truncator_th_[id]->Rotate(nodes_[*m].k1);
            truncator_th_[id]->Rotate(nodes_[*m].k2);
            truncator_th_[id]->Rotate(nodes_[*m].k3);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 4);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 4);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 4);
      }
      //////////////////

      //////////////////
      //      k4      //
      //////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 4);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k4);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 4);
      // Wait for k4 updated.
      BarrierWait(&integrator_threads_barrier_);
      RECORD_START(timer, timer::Type::kRhoUpdate, 4);
#ifdef BLASUPDATE
      blas::Copy(num_entries, density_matrices_th_[id],
                 density_matrices_tmp_th_[id]);
      blas::Axpy(num_entries, &b50, k0, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b51, k1, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b52, k2, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b53, k3, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &b54, k4, 1, density_matrices_tmp_th_[id], 1);
#else
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k0[j] * b50 + k1[j] * b51 +
            k2[j] * b52 + k3[j] * b53 + k4[j] * b54;
      }
#endif
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 4);
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 4);
      RestoreTimeLocalTruncationMatrices(id);
      UpdateTimeLocalTruncation(id, time_now, a5);  // To Q_tl(t + a5).
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 4);
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 5);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + a5, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 5);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 5);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
            // k1 is not used to calculate any more updates.
            truncator_th_[id]->Rotate(nodes_[*m].k2);
            truncator_th_[id]->Rotate(nodes_[*m].k3);
            truncator_th_[id]->Rotate(nodes_[*m].k4);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 5);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 5);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 5);
      }
      //////////////////

      //////////////////
      //      k5      //
      //////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 5);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k5);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 5);
      // Wait for k5 updated.
      BarrierWait(&integrator_threads_barrier_);
      // Now we have k0,...,k5. Use yRKF4 to
      // calculate RKF_4 update and use rho_mid for RKF_5 update. Then
      // compare MAX|RKF_4-RKF_5| and make sure this is less than TOL
      RECORD_START(timer, timer::Type::kRhoUpdate, 5);
#ifdef BLASUPDATE
      blas::Copy(num_entries, density_matrices_th_[id],
                 density_matrices_tmp_th_[id];
      blas::Axpy(num_entries, &c60, k0, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &c62, k2, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &c63, k3, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &c64, k4, 1, density_matrices_tmp_th_[id], 1);
      blas::Axpy(num_entries, &c65, k5, 1, density_matrices_tmp_th_[id], 1);
      max_diff = 0;
      for (m = m_start; m != m_end; ++m) {
        for (int j = 0; j < num_matrix_elements; ++j) {
          y_rkf45 = nodes_[*m].density_matrix[j] + nodes_[*m].k0[j] * c50 +
                    nodes_[*m].k2[j] * c52 + nodes_[*m].k3[j] * c53 +
                    nodes_[*m].k4[j] * c54;
          diff = abs((y_rkf45 - nodes_tmp_[*m].density_matrix[j]));
          max_diff = (diff > max_diff) ? diff : max_diff;
          if (use_adaptive_truncation &&
              abs(nodes_tmp_[*m].density_matrix[j]) >
                  max_density_matrix_value_th_[*m]) {
            max_density_matrix_value_th_[*m] =
                abs(nodes_tmp_[*m].density_matrix[j]);
          }
        }
      }
      rkf45_diff_th_[id] = max_diff;
#else
      max_diff = 0;
      // Note: num_elementsM_th = Ns*Ns*MatrixCount_th
      for (int j = 0; j < num_entries; ++j) {
        y_rkf45 = density_matrices_th_[id][j] + k0[j] * c50 + k2[j] * c52 +
                  k3[j] * c53 + k4[j] * c54;
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k0[j] * c60 + k2[j] * c62 +
            k3[j] * c63 + k4[j] * c64 + k5[j] * c65;
        diff = abs((y_rkf45 - density_matrices_tmp_th_[id][j]));
        max_diff = (diff > max_diff) ? diff : max_diff;
      }
      rkf45_diff_th_[id] = max_diff;
#endif
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 5);

      BarrierWait(&integrator_threads_barrier_);
      /////////////////////////
    }  // END OF RKF45 STEP

    //////////////////////////
    // Wait for rho updated //
    //////////////////////////
    BarrierWait(&all_threads_barrier_);  // Barrier for all threads
    if (id == num_threads) {
      // Use Writer Thread to calculate and distribute max_diff.
      max_diff = 0;
      for (int n = 0; n  < num_threads; ++n) {
        if (max_diff < rkf45_diff_th_[n]) {
          max_diff = rkf45_diff_th_[n];
        }
      }
      for (int n = 0; n  < num_threads+1; ++n) {
        rkf45_diff_th_[n] = max_diff;
      }
    }
    // Each thread compares MaxDiff with tolerance and updates mydt & dt_c.
    // Integrating threads then update rho and writer thread writes out rho.
    BarrierWait(&all_threads_barrier_);  // Barrier for all threads.
    if (rkf45_diff_th_[id] < tolerance || dt_f_ * real(dtdn) < mindt) {
      // Accept the integration step.
      if (id < num_threads) {
        // Copy rho_mid -> rho and check if node can be truncated.
        for (m = m_start; m != m_end; ++m) {
          RECORD_START(timer, timer::Type::kRhoUpdate, 6);
          if (use_adaptive_truncation) max_density_matrix_value_th_[*m] = 0;
          for (int j = 0; j < num_matrix_elements; ++j) {
            nodes_[*m].density_matrix[j] = nodes_tmp_[*m].density_matrix[j];
            if (use_adaptive_truncation &&
                abs(nodes_[*m].density_matrix[j]) >
                    max_density_matrix_value_th_[*m]) {
              max_density_matrix_value_th_[*m] =
                  abs(nodes_[*m].density_matrix[j]);
            }
          }
          RECORD_STOP(timer, timer::Type::kRhoUpdate, 6);
          RECORD_START(timer, timer::Type::kHierarchyTruncationUpdate, 6);
          if (use_adaptive_truncation) {
            if (max_density_matrix_value_th_[*m] < filter_tolerance &&
                nodes_[*m].is_active) {
              nodes_[*m].is_active = false;
              nodes_tmp_[*m].is_active = false;
            } else if (max_density_matrix_value_th_[*m] > filter_tolerance &&
                       !nodes_[*m].is_active) {
              nodes_[*m].is_active = true;
              nodes_tmp_[*m].is_active = true;
            }
          }
          RECORD_STOP(timer, timer::Type::kHierarchyTruncationUpdate, 6);
        }
        RECORD_START(timer, timer::Type::kTimelocalUpdate, 6);
        // Update time local truncation for next step.
        UpdateTimeLocalTruncation(id, time_now + a5, mydt - a5);
        StoreTimeLocalTruncationMatrices(id);
        RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 6);
      }
      // Update time and timesteps.
      RECORD_START(timer, timer::Type::kTimestepUpdate, 6);
      time_now += mydt;
      num_accepted_steps += 1;
      bool update_dt = rkf45_diff_th_[id] < tolerance / 10;
      if (time_now + mydt > end_time) {
        update_dt = true;
        dtup = Complex((end_time - time_now) / mydt);
      }
      // Increase timestep if possible.
      if (update_dt) {
        mydt *= real(dtup);
        a1 *= real(dtup);
        a2 *= real(dtup);
        a3 *= real(dtup);
        a4 *= real(dtup);
        a5 *= real(dtup);
        b10 *= dtup;
        b20 *= dtup;
        b21 *= dtup;
        b30 *= dtup;
        b31 *= dtup;
        b32 *= dtup;
        b40 *= dtup;
        b41 *= dtup;
        b42 *= dtup;
        b43 *= dtup;
        b50 *= dtup;
        b51 *= dtup;
        b52 *= dtup;
        b53 *= dtup;
        b54 *= dtup;
        c50 *= dtup;
        c52 *= dtup;
        c53 *= dtup;
        c54 *= dtup;
        c60 *= dtup;
        c62 *= dtup;
        c63 *= dtup;
        c64 *= dtup;
        c65 *= dtup;
      }
      if (id == num_threads) {
        dt_f_ = mydt;
        dt_c_ = Complex(mydt);
      }
      RECORD_STOP(timer, timer::Type::kTimestepUpdate, 6);
    } else {
      // Don't accept step and decrease timestep before trying again.
      RECORD_START(timer, timer::Type::kTimestepUpdate, 6);
      mydt *= real(dtdn);
      a1 *= real(dtdn);
      a2 *= real(dtdn);
      a3 *= real(dtdn);
      a4 *= real(dtdn);
      a5 *= real(dtdn);
      b10 *= dtdn;
      b20 *= dtdn;
      b21 *= dtdn;
      b30 *= dtdn;
      b31 *= dtdn;
      b32 *= dtdn;
      b40 *= dtdn;
      b41 *= dtdn;
      b42 *= dtdn;
      b43 *= dtdn;
      b50 *= dtdn;
      b51 *= dtdn;
      b52 *= dtdn;
      b53 *= dtdn;
      b54 *= dtdn;
      c50 *= dtdn;
      c52 *= dtdn;
      c53 *= dtdn;
      c54 *= dtdn;
      c60 *= dtdn;
      c62 *= dtdn;
      c63 *= dtdn;
      c64 *= dtdn;
      c65 *= dtdn;
      RECORD_STOP(timer, timer::Type::kTimestepUpdate, 6);
      if (id == num_threads) {
        dt_f_ = mydt;           // Updates dt_f_ for all threads.
        dt_c_ = Complex(mydt);  // Updates dt_c_ for all threads.
        if (dt_f_ < 1e-10) std::cout << "WARNING!!! dt < 1e-10\n";
      } else {
        RECORD_START(timer, timer::Type::kTimelocalUpdate, 6);
        RestoreTimeLocalTruncationMatrices(id);
        RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 6);
      }
    }
    BarrierWait(&all_threads_barrier_);
    if (id == num_threads && (max_diff < tolerance || dt_f_ < mindt)) {
      RECORD_START(timer, timer::Type::kFileOutput, 6);
      if (i >= inext) {  // Write out some timing info.
        time(&t2);
        pthread_mutex_lock(&io_lock_);
        outfile_ << "#Info: Time for single timestep: "
                 << difftime(t2, t1) / (i + 1) << " s.\n"
                 << "#Info: Estimated time for 1 time unit: "
                 << difftime(t2, t1) / (now - start_time) / 60 / 60 << " hrs.\n"
                 << "#Info: Total running time: "
                 << difftime(t2, t1) * end_time /
                    (now - start_time) / 60 / 60 / 24
                 << " days.\n";
        pthread_mutex_unlock(&io_lock_);
        inext += 100;
      }
      if (now > next_write_time) {
        pthread_mutex_lock(&io_lock_);
        outfile_ << std::setw(20) << std::setprecision(6) << time_now << std::setw(6)
                 << std::setprecision(4);
        for (int j = 0; j < num_matrix_elements; j++) {
          outfile_ << " " << nodes_[0].density_matrix[j];
        }
        outfile_ << std::endl;
        pthread_mutex_unlock(&io_lock_);
        next_write_time = now + output_min_timestep;
      }
      RECORD_STOP(timer, timer::Type::kFileOutput, 6);
      if (restart) {
        RECORD_START(timer, timer::Type::kRestartOutput, 6);
        time(&now);
        double diff = difftime(now, last_restart_write_time);
        if (diff > parameters_->restart_output_time) {
          // Write out restart file.
          pthread_mutex_lock(&io_lock_);
          restart_helper_->WriteRestartFile(time_now, dt_f_, nodes_,
                                            matrix_count_,
                                            time_local_truncation_store_th_);
          pthread_mutex_unlock(&io_lock_);
          time(&last_restart_write_time);
        }
        RECORD_STOP(timer, timer::Type::kRestartOutput, 6);
      }
    }  // End of writer thread block.
  }  // End of integration loop.

  time(&t2);
  int active_count = 0;
  for (int j = 0; j < matrix_count_; j++) {
    if (nodes_[j].is_active) {
      ++active_count;
    }
  }
  if (id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    if (verbose_ > -1) {
      ostringstream ss;
      if (use_adaptive_truncation && verbose_ > 0) {
        ss << "#Total number of active matrices = " << active_count << "/"
           << matrix_count_ << "\n";
      }
      ss << "#Total number of integration steps = " << i + 1 << "\n"
         << "#Total number of accepted integration steps = "
         << num_accepted_steps << "\n"
         << "#Total integration time = " << difftime(t2, t1) / 60 << " mins.\n";
      std::cout << ss.str();
      outfile_ << ss.str();
    }
    // Output the final density matrix.
    outfile_ << std::setw(20) << std::setprecision(6) << time_now << std::setw(6)
             << std::setprecision(4);
    for (int j = 0; j < num_matrix_elements; j++) {
      outfile_ << " " << nodes_[0].density_matrix[j];
    }
    outfile_ << std::endl;
    if (restart) {
      // Write out restart file.
      restart_helper_->WriteRestartFile(time_now, dt_f_, nodes_, matrix_count_,
                                        time_local_truncation_store_th_);
    }
    pthread_mutex_unlock(&io_lock_);
  }
  // Write out any recorded timing stats.
  BarrierWait(&all_threads_barrier_);
  std::string timer_output = "";
  RECORD_SUMMARY(timer, timer_output);
  Log(0, id, timer_output);
  delete updater;
  pthread_exit(0);
}

/*
 * Integrate the HEOM using the Runga-Kutta 4 fixed timestep method.
 */
void HierarchyIntegrator::Rk4Integrate(int id, int num_threads) {
  hierarchy_updater::HierarchyUpdater *updater = NULL;

  // The number of integration steps.
  bool restart = parameters_->restart_output_time > 0;
  const bool is_spectrum_calculation = parameters_->is_spectrum_calculation;
  const bool is_diagonal_coupling = parameters_->is_diagonal_bath_coupling;
  const Float filter_tolerance = parameters_->filter_tolerance;
  const bool is_truncated_system = parameters_->is_truncated_system;
  const bool is_time_dependent_hamiltonian =
      parameters_->time_indepenent_hamiltonian == NULL;
  const bool use_adaptive_truncation = filter_tolerance > 0;

  time_t t1, t2, last_restart_write_time, now;

  int *m;
  int *m_start;
  int *m_end;
  if (id != num_threads) {
    m_start = node_indices_th_[id];
    m_end = node_indices_th_[id] + node_count_th_[id];
  } else {
    m_start = new int;
    *m_start = 0;
    m_end = new int;
    *m_end = matrix_count_;
  }
  const int num_matrix_elements =
      is_spectrum_calculation ? num_states_ : num_elements_;
  const int num_entries = num_matrix_elements * node_count_th_[id];
  Complex *k0, *k1, *k2, *k3;

  Complex *next_bath_coupling = NULL;
  Complex *prev_bath_coupling = NULL;
  Complex *matsubara_coupling = NULL;

  const Float start_time = parameters_->start_time;
  const Float end_time = parameters_->integration_time;
  Float time_now = start_time;
  const int output_factor =
      static_cast<int>(ceil(parameters_->output_minimum_timestep / dt_f_));
  const int num_steps = (time_now - start_time) / dt_f_;

  BarrierWait(&all_threads_barrier_);
  if (id < num_threads) {
    Log(1, id, "\tAllocating memory for RK4 updates.");
    k0 = new Complex[num_entries];
    k1 = new Complex[num_entries];
    k2 = new Complex[num_entries];
    k3 = new Complex[num_entries];

    for (int i = 0; i < node_count_th_[id]; ++i) {
      nodes_[node_indices_th_[id][i]].k0 = k0 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k1 = k1 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k2 = k2 + i * num_matrix_elements;
      nodes_[node_indices_th_[id][i]].k3 = k3 + i * num_matrix_elements;
      for (int j = 0; j < num_matrix_elements; ++j) {
        k0[i * num_matrix_elements + j] = Complex(0);
        k1[i * num_matrix_elements + j] = Complex(0);
        k2[i * num_matrix_elements + j] = Complex(0);
        k3[i * num_matrix_elements + j] = Complex(0);
      }
    }
    if (is_diagonal_coupling) {
      Log(1, id, "\tAllocating memory for correlated bath operators.");
      matsubara_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      next_bath_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      prev_bath_coupling =
          new Complex[num_correlation_fn_terms_ * num_matrix_elements];
    } else if (parameters_->is_full_bath_coupling) {
      Log(1, id, "\tAllocating memory for full bath operators.");
      matsubara_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      next_bath_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
      prev_bath_coupling =
          new Complex[num_bath_coupling_terms_ * num_matrix_elements];
    }
    if (is_diagonal_coupling || parameters_->is_full_bath_coupling) {
      Log(2, id, "\tFilling bath operators.");
      FillBathOperators(matsubara_coupling,
                        next_bath_coupling,
                        prev_bath_coupling);
    }
    ostringstream s;
    s << "\tMemory for " << node_count_th_[id] << " density matrices (each with"
      << " " << num_entries << " elements) assigned.";
    Log(0, id, s.str());
    updater = GetUpdater(id, matsubara_coupling,
                         next_bath_coupling, prev_bath_coupling);
    BarrierWait(&integrator_threads_barrier_);
    Log(-1, id, "\tIntegrating using RK4.");
  } else {
    Log(1, id, "\tWriter waiting.");
  }
  BarrierWait(&all_threads_barrier_);
  time(&t1);
  time(&last_restart_write_time);
  timer::Timer *timer = new timer::Timer(7);
  timer->Reset();

  int i = -1;
  while (time_now < end_time) {
    i += 1;
    if (id < num_threads) {
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 0);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 0);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 0);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 0);
         }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 0);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 0);
      }
      /////////////////
      //      k0     //
      /////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 0);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_[*m], nodes_[*m].k0);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 0);
      // No barrier needed since rho_mid is not used in the prior step.
      RECORD_START(timer, timer::Type::kRhoUpdate, 0);
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k0[j] * dt_over_2_;
      }
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 0);
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 0);
      UpdateTimeLocalTruncation(id, time_now, dt_f_ / 2);
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 0);
      /////////////////

      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 1);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + dt_f_/ 2, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 1);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 1);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 1);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 1);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 1);
      }
      /////////////////
      //     k1      //
      /////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 1);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k1);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 1);
      // Wait for k1 updated.
      BarrierWait(&integrator_threads_barrier_);
      RECORD_START(timer, timer::Type::kRhoUpdate, 1);
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k1[j] * dt_over_2_;
      }
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 1);
      BarrierWait(&integrator_threads_barrier_);
      //////////////////

      //////////////////
      //      k2      //
      //////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 2);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k2);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 2);
      // Wait for k2 updated.
      BarrierWait(&integrator_threads_barrier_);
      RECORD_START(timer, timer::Type::kRhoUpdate, 2);
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_tmp_th_[id][j] =
            density_matrices_th_[id][j] + k2[j] * dt_c_;
      }
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 2);
      // To Q_tl(t + dt).
      RECORD_START(timer, timer::Type::kTimelocalUpdate, 2);
      UpdateTimeLocalTruncation(id, time_now + dt_f_ / 2, dt_f_ / 2);
      RECORD_STOP(timer, timer::Type::kTimelocalUpdate, 2);
      /////////////////

      // Updated system to t + dt.
      if (is_time_dependent_hamiltonian) {
        RECORD_START(timer, timer::Type::kHamiltonianUpdate, 3);
        const bool is_hamiltonian_updated =
            UpdateTimeDependentHamiltonian(time_now + dt_f_, id);
        RECORD_STOP(timer, timer::Type::kHamiltonianUpdate, 3);
        if (is_hamiltonian_updated && is_truncated_system) {
          RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 3);
          for (m = m_start; m != m_end; ++m) {
            truncator_th_[id]->Rotate(nodes_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_tmp_[*m].density_matrix);
            truncator_th_[id]->Rotate(nodes_[*m].k0);
            truncator_th_[id]->Rotate(nodes_[*m].k1);
            truncator_th_[id]->Rotate(nodes_[*m].k2);
          }
          RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 3);
        }
      }
      BarrierWait(&integrator_threads_barrier_);
      if (is_time_dependent_hamiltonian && is_truncated_system) {
        RECORD_START(timer, timer::Type::kTruncatedSpaceUpdate, 3);
        FillBathOperators(
            matsubara_coupling, next_bath_coupling, prev_bath_coupling);
        RECORD_STOP(timer, timer::Type::kTruncatedSpaceUpdate, 3);
      }
      /////////////////
      //      k3     //
      /////////////////
      RECORD_START(timer, timer::Type::kDrhoUpdate, 3);
      for (m = m_start; m != m_end; ++m) {
        updater->Get(nodes_tmp_[*m], nodes_[*m].k3);
      }
      RECORD_STOP(timer, timer::Type::kDrhoUpdate, 3);
      RECORD_START(timer, timer::Type::kRhoUpdate, 3);
      for (int j = 0; j < num_entries; ++j) {
        density_matrices_th_[id][j] +=
            dt_over_6_ * (k0[j] + kTwo * k1[j] + kTwo * k2[j] + k3[j]);
      }
      RECORD_STOP(timer, timer::Type::kRhoUpdate, 3);

      if (use_adaptive_truncation) {
        RECORD_START(timer, timer::Type::kHierarchyTruncationUpdate, 4);
        // Find the max |rho_{ij}| for each hierarchy node selectively
        // deactivate or reactivate them based on filter_tolerance.
        for (int *m = m_start; m != m_end; ++m) {
          max_density_matrix_value_th_[*m] = 0;
          for (int j = 0; j < num_matrix_elements; ++j) {
            if (abs(nodes_[*m].density_matrix[j]) >
                max_density_matrix_value_th_[*m]) {
              max_density_matrix_value_th_[*m] =
                  abs(nodes_[*m].density_matrix[j]);
            }
          }
          if (max_density_matrix_value_th_[*m] < filter_tolerance &&
              nodes_[*m].is_active) {
            nodes_[*m].is_active = false;
            nodes_tmp_[*m].is_active = false;
          } else if (max_density_matrix_value_th_[*m] > filter_tolerance &&
                     nodes_[*m].is_active == false) {
            nodes_[*m].is_active = true;
            nodes_tmp_[*m].is_active = true;
          }
        }
      }
      RECORD_STOP(timer, timer::Type::kHierarchyTruncationUpdate, 4);
    }
    //////////////////////////
    // Wait for rho updated //
    //////////////////////////
    BarrierWait(&all_threads_barrier_);
    time_now += dt_f_;
    if (id == num_threads) {
      if ((i % output_factor) == 0) {
        RECORD_START(timer, timer::Type::kFileOutput, 4);
        if (i % 100 == 0) {
          time(&t2);
          outfile_ << "#Time for 1st timestep: " << difftime(t2, t1) / (i + 1)
                   << " s.\n"
                   << "#Estimated time for 1 time unit: "
                   << difftime(t2, t1) / dt_f_ / 60 / 60 / (i + 1) << " hrs.\n"
                   << "#Total running time: "
                   << difftime(t2, t1) * num_steps / 60 / 60 / 24 / (i + 1)
                   << " days.\n";
        }

        outfile_ << time_now;
        for (int j = 0; j < num_matrix_elements; j++) {
          // Writing output in column major order.
          outfile_ << " " << nodes_[0].density_matrix[j];
        }
        outfile_ << "\n";
        RECORD_STOP(timer, timer::Type::kFileOutput, 4);
      }
      if (restart) {
        RECORD_START(timer, timer::Type::kRestartOutput, 4);
        time(&now);
        double diff = difftime(now, last_restart_write_time);
        if (diff > parameters_->restart_output_time) {
          // Write out restart file.
          pthread_mutex_lock(&io_lock_);
          restart_helper_->WriteRestartFile(time_now, dt_f_, nodes_,
                                            matrix_count_,
                                            time_local_truncation_store_th_);
          pthread_mutex_unlock(&io_lock_);
          time(&last_restart_write_time);
        }
        RECORD_STOP(timer, timer::Type::kRestartOutput, 4);
      }
    }
  }
  int num_active_nodes = 0;
  for (int j = 0; j < matrix_count_; ++j) {
    if (nodes_[j].is_active) {
      ++num_active_nodes;
    }
  }
  time(&t2);
  if (id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    if (verbose_ > -1) {
      ostringstream ss;
      if (use_adaptive_truncation && verbose_ > 0) {
        ss << "#Total number of active matrices = " << num_active_nodes << "/"
           << matrix_count_ << "\n";
      }
      ss << "#Total number of integration steps = " << i + 1 << "\n"
         << "#Total integration time = " << difftime(t2, t1) / 60 << " mins\n";
      std::cout << ss.str();
      outfile_ << ss.str();
    }
    // Output the final density matrix.
    outfile_ << std::setw(20) << std::setprecision(6) << time_now << std::setw(6)
             << std::setprecision(4);
    for (int j = 0; j < num_matrix_elements; j++) {
      outfile_ << " " << nodes_[0].density_matrix[j];
    }
    outfile_ << "\n";
    if (restart) {
      // Write out restart file.
      restart_helper_->WriteRestartFile(time_now, dt_f_, nodes_, matrix_count_,
                                        time_local_truncation_store_th_);
    }
    pthread_mutex_unlock(&io_lock_);
  }
  // Write out any recorded timing stats.
  BarrierWait(&all_threads_barrier_);
  std::string timer_output = "";
  RECORD_SUMMARY(timer, timer_output);
  Log(0, id, timer_output);
  delete updater;
  pthread_exit(0);
}

//
// Treats rho[0].rho and appropriate drho as having only Ns*Ns-1 entries,
// starting from index 1.
//
void HierarchyIntegrator::MinimizeStep(const HierarchyNode &node, Complex *drho,
                                       const int &time_step, Complex *temp,
                                       Complex *next, Complex *prev,
                                       Complex *same) {
  int nj;
  int m;
  int j;
  Complex *drho_col_elem, *drho_row_elem, *rho_row_elem, *rho_col_elem;
  if (node.id != 0) {
    blas::Hemm(CblasRight, num_states_, &liouville_prefactor_,
               node.density_matrix, node.hamiltonian, &kZero, drho);
    blas::Hemm(CblasLeft, num_states_, &negative_liouville_prefactor_,
               node.density_matrix, node.hamiltonian_adjoint, &kOne, drho);

    rho_col_elem = node.density_matrix;
    drho_col_elem = drho;
    for (nj = 0; nj < num_elements_; nj++, drho_col_elem++, rho_col_elem++) {
      *drho_col_elem -= node.dephasing_prefactor * (*rho_col_elem);
    }

    if (!parameters_->is_multiple_independent_baths) {
      for (m = 0; m < num_states_; m++) {
        drho_row_elem = &(drho[m * num_states_]);
        rho_row_elem = &(node.density_matrix[m * num_states_]);
        for (nj = 0; nj < num_states_; nj++) {
          drho_row_elem[nj] -=
              (node.matsubara_prefactor[nj] + node.matsubara_prefactor[m]) *
              rho_row_elem[nj];
        }
        nj = (num_states_ + 1) * m;
        drho[nj] +=
            kTwo * node.matsubara_prefactor[m] * node.density_matrix[nj];
      }
    } else {
      for (j = 0; j < num_bath_coupling_terms_; j++) {
        nj = parameters_->multiple_independent_bath_indices[j];
        drho_row_elem = &(drho[nj]);
        drho_col_elem = &(drho[num_states_ * nj]);
        rho_row_elem = &(node.density_matrix[nj]);
        rho_col_elem = &(node.density_matrix[num_states_ * nj]);
        for (m = 0; m < num_states_; m++, rho_row_elem += num_states_,
            rho_col_elem++, drho_row_elem += num_states_, drho_col_elem++) {
          if (m != nj) {
            *drho_row_elem -= node.matsubara_prefactor[j] * (*rho_row_elem);
            *drho_col_elem -= node.matsubara_prefactor[j] * (*rho_col_elem);
          }
        }
      }
    }
  } else {
    Complex dot1, dot2, dot3;
    int nmj = num_states_ - 1;
    // j = 0 case: First Ns Rows of SameLiouville
    rho_row_elem = node.same_liouville + num_states_ * num_states_;
    drho_row_elem = drho + 1;
    for (nj = 1; nj < num_states_; ++nj) {
      rho_col_elem = node.density_matrix;
      blas::DotuSub(num_states_, rho_row_elem, 1, rho_col_elem, 1, &dot2);
      rho_row_elem += num_states_;
      rho_col_elem += num_states_;
      blas::DotuSub(nmj, rho_row_elem + nj, num_states_, rho_col_elem + nj,
                   num_states_, &dot3);
      rho_row_elem += nmj * num_states_;
      *drho_row_elem = dot2 + dot3;
      ++drho_row_elem;
    }
    --nmj;

    // j = 1 -> Ns-1 : //Next Ns*(Ns-2) Rows of SameLiouville
    for (j = 1; j < num_states_ - 1; ++j) {
      for (nj = 0; nj < num_states_; ++nj) {
        rho_col_elem = node.density_matrix;
        blas::DotuSub(j, rho_row_elem + nj, num_states_, rho_col_elem + nj,
                     num_states_, &dot1);
        rho_row_elem += j * num_states_;
        rho_col_elem += j * num_states_;
        blas::DotuSub(num_states_, rho_row_elem, 1, rho_col_elem, 1, &dot2);
        rho_row_elem += num_states_;
        rho_col_elem += num_states_;  // Nmj = Ns-1-j
        blas::DotuSub(nmj, rho_row_elem + nj, num_states_, rho_col_elem + nj,
                     num_states_, &dot3);
        rho_row_elem += nmj * num_states_;
        *drho_row_elem = dot1 + dot2 + dot3;
        ++drho_row_elem;
      }
      --nmj;
    }

    // Final Ns Rows of SameLiouville
    j = num_states_ - 1;
    for (nj = 0; nj < num_states_; ++nj) {
      rho_col_elem = node.density_matrix;
      blas::DotuSub(j, rho_row_elem + nj, num_states_, rho_col_elem + nj,
                   num_states_, &dot1);
      rho_row_elem += j * num_states_;
      rho_col_elem += j * num_states_;
      blas::DotuSub(num_states_, rho_row_elem, 1, rho_col_elem, 1, &dot2);
      rho_row_elem += num_states_;
      *drho_row_elem = dot1 + dot2;
      ++drho_row_elem;
    }
  }

  ////// NEXT HIERARCHY MATRICES
  for (j = 0; j < node.num_next_hierarchy_nodes; j++) {
    nj = node.next_coupling_op_index[j];
    drho_col_elem = &(drho[num_states_ * nj]);
    rho_col_elem =
        &(node.next_hierarchy_nodes[j]->density_matrix[num_states_ * nj]);
    for (m = 0; m < num_states_; m++) {
      drho_col_elem[m] -= (node.next_prefactor[j]) * rho_col_elem[m];
    }
  }
  for (m = 0; m < num_states_; m++) {
    drho_row_elem = &(drho[num_states_ * m]);
    for (j = 0; j < node.num_next_hierarchy_nodes; j++) {
      nj = node.next_coupling_op_index[j];
      rho_row_elem =
          &(node.next_hierarchy_nodes[j]->density_matrix[num_states_ * m]);
      drho_row_elem[nj] += node.next_prefactor[j] * rho_row_elem[nj];
    }
  }
  ////// PREV HIERARCHY MATRICES
  for (j = 0; j < node.num_prev_hierarchy_nodes; j++) {
    if (node.prev_hierarchy_nodes[j]->id != 0 ||
        node.next_liouville_index[j] != 0) {
      rho_col_elem = node.prev_hierarchy_nodes[j]->density_matrix;
      m = node.prev_bath_coupling_op_index[j];
      nj = m;
      for (int i = 0; i < m; ++i) {
        drho[nj] -= node.prev_prefactor_col[j] * rho_col_elem[nj];
        nj += num_states_;
      }

      nj -= m;
      for (int i = 0; i < m; ++i) {
        drho[nj] -= node.prev_prefactor_row[j] * rho_col_elem[nj];
        ++nj;
      }
      drho[nj] += (-node.prev_prefactor_col[j] - node.prev_prefactor_row[j]) *
                  rho_col_elem[nj];
      ++nj;
      for (int i = m + 1; i < num_states_; ++i) {
        drho[nj] -= node.prev_prefactor_row[j] * rho_col_elem[nj];
        ++nj;
      }

      nj += m;
      for (int i = m + 1; i < num_states_; ++i) {
        drho[nj] -= node.prev_prefactor_col[j] * rho_col_elem[nj];
        nj += num_states_;
      }
    } else {
      for (nj = 1; nj < 2 * num_states_ - 1; ++nj) {
        m = node.next_liouville_index[j][nj];
        drho[m] += node.prev_liouville[j][nj] *
                   node.prev_hierarchy_nodes[j]->density_matrix[m];
      }
    }
  }
}

/////////////////////////////////////////////
//       STEADY STATE        //
// Bi-Conjugate Gradient Stabilized method //
//      including l-GMRES steps    //
/////////////////////////////////////////////
void HierarchyIntegrator::BicgstabLSteadyState(int id, int num_threads) {
  int m_start = id;
  time_t t1, t2;
  time(&t1);
  const int bicgstab_ell = parameters_->bicgstab_ell;
  const Float bicgstab_tolerance = parameters_->bicgstab_tolerance;

  int l = bicgstab_ell;
  int lp1 = l + 1;
  Complex *b;
  b = new Complex[num_states_ * num_states_];

  if (id == 0) {
    // Calculate target vector:
    b = new Complex[num_states_ * num_states_];
    for (int j = 0; j < num_states_ * num_states_; ++j)
      b[j] = -nodes_[0].same_liouville[j * num_states_ * num_states_];

    // Set top row to zero:
    for (int j = 0; j < num_states_ * num_states_; ++j)
      nodes_[0].same_liouville[j] = 0;
  }

  Complex result;
  Complex rho0, rho1_th, omega, beta, alpha_th;
  Complex *tau;
  Complex sigma;
  Complex *gamma;
  Complex *gammap;
  Complex *gammapp;

  tau = new Complex[lp1 * lp1];
  gamma = new Complex[lp1];
  gammap = new Complex[lp1];
  gammapp = new Complex[lp1];
  for (int i = 0; i < lp1 * lp1; ++i) tau[i] = 0;
  for (int i = 0; i < lp1; ++i) {
    gamma[i] = 0;
    gammap[i] = 0;
    gammapp[i] = 0;
  }

  rho0 = 1;
  alpha_th = 0;
  omega = 1;
  result = 0;

  int max_steps = 999999;
  Float diff_th = numeric_limits<Float>::max();
  Float eps = bicgstab_tolerance;
  int step = 0;

  if (verbose_ > 2 && id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    std::cout << "Getting initial residuals.\n";
    pthread_mutex_unlock(&io_lock_);
  }

  BarrierWait(&all_threads_barrier_);

  // Calculate initial residuals
  if (id < num_threads) {
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      nodes_[m].density_matrix = nodes_[m].steady_state_density_matrix;
      for (int i = 0; i < num_elements_; ++i) nodes_[m].density_matrix[i] = 0;
    }
    if (id == 0) {
      nodes_[0].density_matrix[0] = 0;
    }
    BarrierWait(&integrator_threads_barrier_);
    for (int m = m_start; m < matrix_count_; m += num_threads)
      MinimizeStep(nodes_[m], nodes_[m].r, step, NULL, NULL, NULL, NULL);
    bicgstab_diff_[id] = 0;
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      if (m >= 1 + hierarchy_level_[1]) {
        for (int i = 0; i < num_states_ * num_states_; ++i) {
          nodes_[m].r[i] = -nodes_[m].r[i];
          nodes_[m].r0[i] = nodes_[m].r[i];
        }
      } else if (m > 0) {
        nodes_[m].r[0] = -nodes_[m].r[0];
        if (nodes_[m].prev_node_index[0] == 0)
          nodes_[m].r[0] -= nodes_[m].prev_liouville[0][0];
        nodes_[m].r0[0] = nodes_[m].r[0];
        for (int i = 1; i < num_states_ * num_states_; ++i) {
          nodes_[m].r[i] = -nodes_[m].r[i];
          nodes_[m].r0[i] = nodes_[m].r[i];
        }
      } else {
        for (int i = 0; i < num_states_ * num_states_; ++i) {
          nodes_[m].r[i] = b[i] - nodes_[m].r[i];
          nodes_[m].r0[i] = nodes_[m].r[i];
        }
        nodes_[m].r[0] = 0;
        nodes_[m].r0[0] = 0;
      }
      blas::DotcSub(num_elements_, nodes_[m].r, 1, nodes_[m].r, 1, &result);
      bicgstab_diff_[id] += real(result);
    }
    BarrierWait(&integrator_threads_barrier_);
    diff_th = 0;
    for (int m = 0; m < num_threads; ++m) {
      diff_th += bicgstab_diff_[m];
    }
    diff_th = sqrt(diff_th);
    result = Complex(1. / diff_th);
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      blas::Scal(num_elements_, &result, nodes_[m].r0, 1);
    }
  }
  if (verbose_ > 1) {
    if (id < num_threads) {
      pthread_mutex_lock(&io_lock_);
      std::cout << "[" << id
           << "]  Calculating steady state density matrix using BiCGSTAB(" << l
           << ").\n";
      pthread_mutex_unlock(&io_lock_);
    }
  } else if (verbose_ > 0 && id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    std::cout << "Calculating steady state density matrix using  BiCGSTAB(" << l
         << ").\n";
    pthread_mutex_unlock(&io_lock_);
  }

  BarrierWait(&all_threads_barrier_);
  int n = 0;
  // MaxSteps = 6;
  while (diff_th > eps && step < max_steps) {
    if (id < num_threads) {
      // Take l BiCG steps
      rho0 = -omega * rho0;
      // cout << step << " Doing BiCG Calc:\n";
      for (int j = 0; j < l; ++j) {
        rho1_[id] = 0;
        for (int m = m_start; m < matrix_count_; m += num_threads) {
          blas::DotcSub(num_elements_, nodes_[m].r0, 1,
                       nodes_[m].r + j * num_elements_, 1, &result);
          rho1_[id] += result;
        }

        ///////////////////////////////////////////
        BarrierWait(&integrator_threads_barrier_);
        rho1_th = 0;
        for (int m = 0; m < num_threads; ++m) {
          rho1_th += rho1_[m];
        }
        beta = -alpha_th * rho1_th / rho0;
        //////////////////////////////////////////
        rho0 = rho1_th;

        for (int m = m_start; m < matrix_count_; m += num_threads) {
          blas::Scal(num_elements_ * (j + 1), &beta, nodes_[m].v, 1);
          blas::Axpy(num_elements_ * (j + 1), &kOne, nodes_[m].r, 1,
                     nodes_[m].v, 1);
          nodes_[m].density_matrix = nodes_[m].v + j * num_elements_;
        }

        ///////////////////////////////////////////
        BarrierWait(&integrator_threads_barrier_);
        for (int m = m_start; m < matrix_count_; m += num_threads)
          MinimizeStep(nodes_[m], nodes_[m].v + (j + 1) * num_elements_, step,
                       NULL, NULL, NULL, NULL);
        if (id == 0) nodes_[0].v[(j + 1) * num_elements_] = 0;
        ///////////////////////////////////////////

        alpha_[id] = 0;
        for (int m = m_start; m < matrix_count_; m += num_threads) {
          blas::DotcSub(num_elements_, nodes_[m].r0, 1,
                        nodes_[m].v + (j + 1) * num_elements_, 1, &result);
          alpha_[id] += result;
        }
        /////////////////////////////////////////
        BarrierWait(&integrator_threads_barrier_);
        alpha_th = 0;
        for (int m = 0; m < num_threads; ++m) {
          alpha_th += alpha_[m];
        }
        alpha_th = -rho1_th / alpha_th;
        /////////////////////////////////////////
        for (int m = m_start; m < matrix_count_; m += num_threads) {
          blas::Axpy(num_elements_ * (j + 1), &alpha_th,
                     nodes_[m].v + num_elements_, 1, nodes_[m].r, 1);
          nodes_[m].density_matrix = nodes_[m].r + j * num_elements_;
        }

        alpha_th = -alpha_th;
        BarrierWait(&integrator_threads_barrier_);
        for (int m = m_start; m < matrix_count_; m += num_threads)
          MinimizeStep(nodes_[m], nodes_[m].r + (j + 1) * num_elements_, step,
                       NULL, NULL, NULL, NULL);
        if (id == 0) nodes_[0].r[(j + 1) * num_elements_] = 0;

        for (int m = m_start; m < matrix_count_; m += num_threads)
          blas::Axpy(num_elements_, &alpha_th, nodes_[m].v, 1,
                     nodes_[m].steady_state_density_matrix, 1);
      }
      //// GMRES SECTION
      for (int j = 1; j <= l; ++j) {
        for (int i = 1; i < j; ++i) {
          alpha_[id] = 0;
          for (int m = m_start; m < matrix_count_; m += num_threads) {
            blas::DotcSub(num_elements_, nodes_[m].r + j * num_elements_, 1,
                         nodes_[m].r + i * num_elements_, 1, &result);
            alpha_[id] += result;
          }
          /////////////////////////////////////////
          BarrierWait(&integrator_threads_barrier_);
          n = i * lp1 + j;
          tau[n] = 0;
          for (int m = 0; m < num_threads; ++m) {
            tau[n] += alpha_[m];
          }
          tau[n] /= -gamma[i];
          /////////////////////////////////////////
          for (int m = m_start; m < matrix_count_; m += num_threads)
            blas::Axpy(num_elements_, tau + n, nodes_[m].r + i * num_elements_,
                       1, nodes_[m].r + j * num_elements_, 1);
          tau[n] *= -1;
        }

        omega_top_[id] = 0;
        omega_btm_[id] = 0;
        for (int m = m_start; m < matrix_count_; m += num_threads) {
          blas::DotcSub(num_elements_, nodes_[m].r + j * num_elements_, 1,
                       nodes_[m].r + j * num_elements_, 1, &result);
          omega_btm_[id] += result;
          blas::DotcSub(num_elements_, nodes_[m].r + j * num_elements_, 1,
                       nodes_[m].r, 1, &result);
          omega_top_[id] += result;
        }
        /////////////////////////////////////////
        BarrierWait(&integrator_threads_barrier_);
        gamma[j] = 0;
        gammap[j] = 0;
        for (int m = 0; m < num_threads; ++m) {
          gamma[j] += omega_btm_[m];
          gammap[j] += omega_top_[m];
        }
        /////////////////////////////////////////
        gammap[j] /= gamma[j];
        // cout << step << " " << j << " " << gammap[j] << endl;
      }

      gamma[l] = gammap[l];
      omega = gammap[l];
      for (int j = l - 1; j > 0; --j) {
        sigma = 0;
        for (int i = j + 1; i <= l; ++i) {
          sigma += tau[j * lp1 + i] * gamma[i];
          // cout << step << " " << j << " " << i << " " << tau[j*lp1+i] <<
          // endl;
        }
        gamma[j] = gammap[j] - sigma;
        // cout << step << " " << j << " " << gamma[j] << endl;
      }

      for (int j = 1; j < l; ++j) {
        sigma = 0;
        for (int i = j + 1; i < l; ++i)
          sigma += tau[j * lp1 + i] * gamma[i + 1];
        gammapp[j] = gamma[j + 1] + sigma;
      }

      for (int m = m_start; m < matrix_count_; m += num_threads) {
        blas::Axpy(num_elements_, gamma + 1, nodes_[m].r, 1,
                   nodes_[m].steady_state_density_matrix, 1);
        for (int j = 1; j < l; ++j)
          blas::Axpy(num_elements_, gammapp + j,
                     nodes_[m].r + j * num_elements_, 1,
                     nodes_[m].steady_state_density_matrix, 1);
      }

      for (int j = 0; j <= l; ++j) {
        gamma[j] = -gamma[j];
        gammap[j] = -gammap[j];
      }

      // Update residuals and search directions
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        for (int j = l; j > 0; --j)
          blas::Axpy(num_elements_, gammap + j, nodes_[m].r + j * num_elements_,
                     1, nodes_[m].r, 1);
        for (int j = l; j > 0; --j)
          blas::Axpy(num_elements_, gamma + j, nodes_[m].v + j * num_elements_,
                     1, nodes_[m].v, 1);
      }

      // Recalculate ||r||
      ////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      bicgstab_diff_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        blas::DotcSub(num_elements_, nodes_[m].r, 1, nodes_[m].r, 1, &result);
        bicgstab_diff_[id] += real(result);
      }
      //////////////////////////////////////

    } else {
      bicgstab_diff_[id] = 0;
    }

    /////////////////////////////////////
    BarrierWait(&all_threads_barrier_);
    diff_th = 0;
    for (int m = 0; m < num_threads; ++m) {
      // if (diff_th < diff[m])
      diff_th += bicgstab_diff_[m];
    }
    diff_th = sqrt(diff_th);
    /////////////////////////////////////
    if (id == num_threads) {
      // cout << step << " diff= " << diff_th << endl;
      if (step % 1 == 0) {
        // rho[0].SteadyStateRho[0] = 1;
        result = 1;
        for (int i = 1; i < num_states_; ++i)
          result +=
              nodes_[0].steady_state_density_matrix[(1 + num_states_) * i];
        outfile_ << diff_th << " " << 1. / real(result);
        for (int i = 0; i < num_states_ * num_states_; ++i)
          outfile_ << nodes_[0].steady_state_density_matrix[i] / real(result)
                   << " ";
        outfile_ << std::endl;
      }
    }
    step += l;
  }  // end while loop
  BarrierWait(&all_threads_barrier_);
  time(&t2);
  // Finished, now write out the results
  if (id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    ostringstream ss;
    if (step == max_steps) {
      ss << "#Convergence failed! (steps = " << step << ", diff = " << diff_th
         << ")\n";
    } else {
      ss << "Converged to diff = " << diff_th << " in " << step << " steps.\n";
    }
    std::cout << ss.str();
    outfile_ << ss.str();
    if (verbose_ > 0) std::cout << "Writing final steady-state density matrix.\n";
    outfile_ << "#Info: Total running time: " << difftime(t2, t1) / 60 / 60 / 24
             << " days.\n";
    nodes_[0].steady_state_density_matrix[0] = 1;
    result = 1;
    for (int i = 1; i < num_states_; ++i) {
      result += nodes_[0].steady_state_density_matrix[(1 + num_states_) * i];
    }
    for (int i = 0; i < num_states_ * num_states_; ++i) {
      nodes_[0].steady_state_density_matrix[i] /= result;
    }
    outfile_ << diff_th << " ";
    for (int i = 0; i < num_states_ * num_states_; ++i) {
      outfile_ << nodes_[0].steady_state_density_matrix[i] << " ";
    }
    outfile_ << std::endl;

    if (verbose_ > 1) {
      PrintMatrix(nodes_[0].steady_state_density_matrix, num_states_);
    }
    if (verbose_ > 2) {
      for (int m = 1; m < matrix_count_; ++m) {
        std::cout << m << ":\n";
        PrintMatrix(nodes_[m].steady_state_density_matrix, num_states_);
      }
    }
    std::cout << "Total running time: " << difftime(t2, t1) / 60 / 60 << " hours.\n";
    if (parameters_->restart_output_time > 0) {
      restart_helper_->WriteRestartFile((Float)step, 0, nodes_, matrix_count_,
                                        NULL);
    }
    pthread_mutex_unlock(&io_lock_);
  }
  BarrierWait(&all_threads_barrier_);
  pthread_exit(0);
}

/////////////////////////////////////////////
//       STEADY STATE        //
// Bi-Conjugate Gradient Stabilized method //
/////////////////////////////////////////////
void HierarchyIntegrator::BicgstabSteadyState(int id, int num_threads) {
  int m_start = id;
  time_t t1, t2;
  time(&t1);

  Complex *b;
  b = new Complex[num_states_ * num_states_];

  if (id == 0) {
    // Correction for improved stability
    b = new Complex[num_states_ * num_states_];
    for (int j = 0; j < num_states_ * num_states_; ++j) {
      b[j] = -nodes_[0].same_liouville[j * num_states_ * num_states_];
      nodes_[0].same_liouville[j * num_states_ * num_states_] = 0;
    }
    // Set top row to zero:
    for (int j = 0; j < num_states_ * num_states_; ++j)
      nodes_[0].same_liouville[j] = 0;
  }

  Complex result;
  Complex rho0, rho1_th, omega_btm_th, omega, beta, alpha_th;
  Complex omega_inv;
  rho0 = rho1_th = alpha_th = beta = 1;
  omega = -1;
  result = 0;
  const int kMaxSteps = 9999;
  Float diff_th = numeric_limits<Float>::max();
  Complex diff_th_c = 0;
  const Float bicgstab_tolerance = parameters_->bicgstab_tolerance;
  int step = 0;

  if (verbose_ > 2 && id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    std::cout << "Getting initial residuals.\n";
    pthread_mutex_unlock(&io_lock_);
  }

  BarrierWait(&all_threads_barrier_);
  // Calculate initial residuals
  if (id < num_threads) {
    for (int m = m_start; m < matrix_count_; m += num_threads)
      nodes_[m].density_matrix = nodes_[m].steady_state_density_matrix;
    if (id == 0) {
      nodes_[0].density_matrix[0] = 0;
    }
    BarrierWait(&integrator_threads_barrier_);
    bicgstab_diff_[id] = 0;
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      if (m >= 1 + hierarchy_level_[1]) {
        for (int i = 0; i < num_states_ * num_states_; ++i) {
          nodes_[m].r[i] = -nodes_[m].r[i];
          nodes_[m].r0[i] = nodes_[m].r[i];
        }
      } else if (m > 0) {
        nodes_[m].r[0] = -nodes_[m].r[0];
        if (nodes_[m].prev_node_index[0] == 0)
          nodes_[m].r[0] -= nodes_[m].prev_liouville[0][0];
        nodes_[m].r0[0] = nodes_[m].r[0];
        for (int i = 1; i < num_states_ * num_states_; ++i) {
          nodes_[m].r[i] = -nodes_[m].r[i];
          nodes_[m].r0[i] = nodes_[m].r[i];
        }
      } else {
        for (int i = 0; i < num_states_ * num_states_; ++i) {
          nodes_[m].r[i] = b[i] - nodes_[m].r[i];
          nodes_[m].r0[i] = nodes_[m].r[i];
        }
        nodes_[m].r[0] = 0;
        nodes_[m].r0[0] = 0;
      }
      blas::DotcSub(num_elements_, nodes_[m].r, 1, nodes_[m].r, 1, &result);
      bicgstab_diff_[id] += real(result);
    }

    BarrierWait(&integrator_threads_barrier_);
    diff_th = 0;
    for (int m = 0; m < num_threads; ++m) {
      if (diff_th < bicgstab_diff_[m]) diff_th = bicgstab_diff_[m];
    }
    diff_th_c = Complex(1. / diff_th);
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      blas::Scal(num_elements_, &diff_th_c, nodes_[m].r0, 1);
    }
  }
  if (verbose_ > 1) {
    if (id < num_threads) {
      pthread_mutex_lock(&io_lock_);
      std::cout << "[" << id
           << "]  Calculating steady state density matrix using BiCGSTAB.\n";
      pthread_mutex_unlock(&io_lock_);
    }
  } else if (verbose_ > 0 && id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    std::cout << "Calculating steady state density matrix using BiCGSTAB.\n";
    pthread_mutex_unlock(&io_lock_);
  }

  BarrierWait(&all_threads_barrier_);

  while (diff_th > bicgstab_tolerance && step < kMaxSteps) {
    ++step;
    if (id < num_threads) {
      rho1_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        blas::DotcSub(num_elements_, nodes_[m].r0, 1, nodes_[m].r, 1, &result);
        rho1_[id] += result;
      }

      ///////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      rho1_th = 0;
      for (int m = 0; m < num_threads; ++m) {
        rho1_th += rho1_[m];
      }
      beta = -rho1_th / rho0 * alpha_th;
      rho0 = rho1_th;
      //////////////////////////////////////////

      omega_inv = 1. / real(omega);
      for (int m = m_start; m < matrix_count_; m += num_threads) {
       // v <- v-p/omega
       blas::Axpy(num_elements_, &omega_inv, nodes_[m].p, 1, nodes_[m].v, 1);
       blas::Copy(num_elements_, nodes_[m].r, nodes_[m].p);  // p <- r
       // p <- p-beta*v
       blas::Axpy(num_elements_, &beta, nodes_[m].v, 1, nodes_[m].p, 1);
        nodes_[m].density_matrix = nodes_[m].p;
      }

      ///////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // v <- A*p
        MinimizeStep(nodes_[m], nodes_[m].v, step, NULL, NULL, NULL, NULL);
      }
      if (id == 0) nodes_[0].v[0] = 0;
      ///////////////////////////////////////////

      alpha_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        blas::DotcSub(num_elements_, nodes_[m].r0, 1, nodes_[m].v, 1, &result);
        alpha_[id] += result;
      }

      /////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      alpha_th = 0;
      for (int m = 0; m < num_threads; ++m) {
        alpha_th += alpha_[m];
      }
      alpha_th = -rho1_th / alpha_th;
      /////////////////////////////////////////

      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // r <- r-alpha*v
       blas::Axpy(num_elements_, &alpha_th, nodes_[m].v, 1, nodes_[m].r, 1);
        nodes_[m].density_matrix = nodes_[m].r;  // ready A*r
      }
      alpha_th = -alpha_th;

      ////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // t <- A*r
        MinimizeStep(nodes_[m], nodes_[m].t, step, NULL, NULL, NULL, NULL);
      }
      if (id == 0) nodes_[0].t[0] = 0;
      //////////////////////////////////////

      omega_btm_[id] = 0;
      omega_top_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // dot(t,r)
        blas::DotcSub(num_elements_, nodes_[m].t, 1, nodes_[m].r, 1, &result);
        omega_top_[id] += result;
        // dot(t,t)
        blas::DotcSub(num_elements_, nodes_[m].t, 1, nodes_[m].t, 1, &result);
        omega_btm_[id] += result;
      }

      /////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      omega_btm_th = 0;
      omega = 0;
      for (int m = 0; m < num_threads; ++m) {
        omega += omega_top_[m];
        omega_btm_th += omega_btm_[m];
      }
      omega /= omega_btm_th;
      /////////////////////////////////////////

      // Update rho(\infty)
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // rho <- rho + alpha*p
       blas::Axpy(num_elements_, &alpha_th, nodes_[m].p, 1,
                 nodes_[m].steady_state_density_matrix, 1);
        // rho <- rho + omega*s
       blas::Axpy(num_elements_, &omega, nodes_[m].r, 1,
                 nodes_[m].steady_state_density_matrix, 1);
      }
      omega = -omega;
      // Recalculate residuals
      ////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      bicgstab_diff_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // r <- r-omega*t
       blas::Axpy(num_elements_, &omega, nodes_[m].t, 1, nodes_[m].r, 1);
        blas::DotcSub(num_elements_, nodes_[m].r, 1, nodes_[m].r, 1, &result);
        bicgstab_diff_[id] += real(result);
      }
      //////////////////////////////////////
    } else {
      bicgstab_diff_[id] = 0;
    }

    /////////////////////////////////////
    BarrierWait(&all_threads_barrier_);
    diff_th = 0;
    for (int m = 0; m < num_threads; ++m) {
      if (diff_th < bicgstab_diff_[m]) diff_th = bicgstab_diff_[m];
    }
    /////////////////////////////////////
    if (id == num_threads) {
      if (step % 1 == 0) {
        // rho[0].SteadyStateRho[0] = 1;
        result = 1;
        for (int i = 1; i < num_states_; ++i)
          result +=
              nodes_[0].steady_state_density_matrix[(1 + num_states_) * i];
        outfile_ << diff_th << " " << 1. / real(result) << " ";
        for (int i = 1; i < num_states_ * num_states_; ++i)
          outfile_ << nodes_[0].steady_state_density_matrix[i] / result << " ";
        outfile_ << std::endl;
      }
    }
  }  // end while loop

  //}
  BarrierWait(&all_threads_barrier_);
  time(&t2);
  // Finished, now write out the results
  if (id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    if (step == kMaxSteps) {
      std::cout << "Convergence failed! (steps = " << step << ", diff = " << diff_th
           << ")\n";
      outfile_ << "#Info: Total steps = " << step << ". No convergence.\n";
    } else {
      std::cout << "Converged to diff = " << diff_th << " in " << step
           << " steps.\n";
      outfile_ << "#Info: Converged to diff = " << diff_th << " in " << step
               << " steps.\n";
    }

    if (verbose_ > 0) std::cout << "Writing final steady-state density matrix.\n";

    outfile_ << "#Info: Total running time: " << difftime(t2, t1) / 60 / 60 / 24
             << " days.\n";

    nodes_[0].steady_state_density_matrix[0] = 1;
    result = 1;
    for (int i = 1; i < num_states_; ++i)
      result += nodes_[0].steady_state_density_matrix[(1 + num_states_) * i];
    for (int i = 0; i < num_states_ * num_states_; ++i)
      nodes_[0].steady_state_density_matrix[i] /= result;
    outfile_ << step << " ";
    for (int i = 0; i < num_states_ * num_states_; ++i) {
      outfile_ << nodes_[0].steady_state_density_matrix[i] << " ";
    }
    outfile_ << std::endl;

    if (verbose_ > 1)
      PrintMatrix(nodes_[0].steady_state_density_matrix, num_states_);
    if (verbose_ > 2) {
      for (int m = 1; m < matrix_count_; ++m) {
        std::cout << m << ":\n";
        PrintMatrix(nodes_[m].steady_state_density_matrix, num_states_);
      }
    }
    std::cout << "Total running time: " << difftime(t2, t1) / 60 / 60 << " hours.\n";
    if (parameters_->restart_output_time > 0) {
      restart_helper_->WriteRestartFile(
          static_cast<Float>(step), 0, nodes_, matrix_count_, NULL);
    }
    pthread_mutex_unlock(&io_lock_);
  }
  BarrierWait(&all_threads_barrier_);
}

void HierarchyIntegrator::BicgstabSteadyState_unstable(int id,
                                                       int num_threads) {
  int m_start = id;
  time_t t1, t2;
  time(&t1);

  Complex result;
  Complex rho0, rho1_th, omega_btm_th, omega, beta, alpha_th;
  Complex omega_inv;
  rho0 = rho1_th = alpha_th = beta = 1;
  omega = -1;
  result = 0;
  const int kMaxSteps = 9999;
  Float diff_th = 1;
  Float eps = parameters_->bicgstab_tolerance;
  int step = 0;
  int num_elements = num_states_ * num_states_;
  hierarchy_updater::IndependentBath *updater =
      new hierarchy_updater::IndependentBath(parameters_, liouville_prefactor_,
                                             NULL, NULL);
  if (verbose_ > 1) {
    if (id < num_threads) {
      pthread_mutex_lock(&io_lock_);
      std::cout << "[" << id
           << "]  Calculating steady state density matrix using BiCGSTAB.\n";
      pthread_mutex_unlock(&io_lock_);
    }
  } else if (verbose_ > 0 && id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    std::cout << "Calculating steady state density matrix using BiCGSTAB.\n";
    pthread_mutex_unlock(&io_lock_);
  }

  BarrierWait(&all_threads_barrier_);
  // Calculate initial residuals
  if (id < num_threads) {
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      nodes_[m].density_matrix = nodes_[m].steady_state_density_matrix;
    }

    BarrierWait(&integrator_threads_barrier_);
    for (int m = m_start; m < matrix_count_; m += num_threads) {
      updater->Get(nodes_[m], nodes_[m].r);
    }

    for (int m = m_start; m < matrix_count_; m += num_threads) {
      for (int i = 0; i < num_elements; ++i) {
        nodes_[m].r[i] = -nodes_[m].r[i];
        nodes_[m].r0[i] = nodes_[m].r[i];
      }
    }
  }

  BarrierWait(&all_threads_barrier_);
  while (diff_th > eps && step < kMaxSteps) {
    ++step;
    if (id < num_threads) {
      rho1_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        blas::DotcSub(num_elements, nodes_[m].r0, 1, nodes_[m].r, 1, &result);
        rho1_[id] += result;
      }

      ///////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      rho1_th = 0;
      for (int m = 0; m < num_threads; ++m) {
        rho1_th += rho1_[m];
      }
      beta = -rho1_th / rho0 * alpha_th;
      //////////////////////////////////////////

      omega_inv = 1. / real(omega);
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // v <- v-p/omega
       blas::Axpy(num_elements, &omega_inv, nodes_[m].p, 1, nodes_[m].v, 1);
       blas::Copy(num_elements, nodes_[m].r, nodes_[m].p);  // p <- r
        // p <- p-beta*v
       blas::Axpy(num_elements, &beta, nodes_[m].v, 1, nodes_[m].p, 1);
        nodes_[m].density_matrix = nodes_[m].p;
      }

      ///////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // v <- A*p
        updater->Get(nodes_[m], nodes_[m].v);
      }
      ///////////////////////////////////////////

      alpha_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        blas::DotcSub(num_elements, nodes_[m].r0, 1, nodes_[m].v, 1, &result);
        alpha_[id] += result;
      }

      /////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      alpha_th = 0;
      for (int m = 0; m < num_threads; ++m) {
        alpha_th += alpha_[m];
      }
      alpha_th = -rho1_th / alpha_th;
      /////////////////////////////////////////

      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // r <- r-alpha*v
       blas::Axpy(num_elements, &alpha_th, nodes_[m].v, 1, nodes_[m].r, 1);
        nodes_[m].density_matrix = nodes_[m].r;  // ready A*r
      }
      alpha_th = -alpha_th;

      ////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      for (int m = m_start; m < matrix_count_; m += num_threads)
        updater->Get(nodes_[m], nodes_[m].t);
      //////////////////////////////////////

      omega_btm_[id] = 0;
      omega_top_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // dot(t,r)
        blas::DotcSub(num_elements, nodes_[m].t, 1, nodes_[m].r, 1, &result);
        omega_top_[id] += result;
        // dot(t,t)
        blas::DotcSub(num_elements, nodes_[m].t, 1, nodes_[m].t, 1, &result);
        omega_btm_[id] += result;
      }

      /////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      omega_btm_th = 0;
      omega = 0;
      for (int m = 0; m < num_threads; ++m) {
        omega += omega_top_[m];
        omega_btm_th += omega_btm_[m];
      }
      omega /= omega_btm_th;
      /////////////////////////////////////////

      // Update rho(\infty)
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // rho <- rho + alpha*p
       blas::Axpy(num_elements, &alpha_th, nodes_[m].p, 1,
                 nodes_[m].steady_state_density_matrix, 1);
        // rho <- rho + omega*s
       blas::Axpy(num_elements, &omega, nodes_[m].r, 1,
                 nodes_[m].steady_state_density_matrix, 1);
      }

      omega = -omega;

      // Recalculate residuals
      ////////////////////////////////////////
      BarrierWait(&integrator_threads_barrier_);
      bicgstab_diff_[id] = 0;
      for (int m = m_start; m < matrix_count_; m += num_threads) {
        // r <- r-omega*t
       blas::Axpy(num_elements, &omega, nodes_[m].t, 1, nodes_[m].r, 1);
      }
      if (id == 0) {
        blas::DotcSub(num_elements, nodes_[0].r, 1, nodes_[0].r, 1, &result);
        bicgstab_diff_[id] += real(result);
      }
      //////////////////////////////////////

      rho0 = rho1_th;
    } else {
    }

    /////////////////////////////////////
    BarrierWait(&all_threads_barrier_);
    diff_th = 0;
    for (int m = 0; m < num_threads; ++m) {
      diff_th += bicgstab_diff_[m];
    }
    /////////////////////////////////////
    if (id == num_threads) {
      if (step % 10 == 0) {
        outfile_ << diff_th << " ";
        for (int i = 0; i < num_states_ * num_states_; ++i)
          outfile_ << nodes_[0].steady_state_density_matrix[i] << " ";
        outfile_ << std::endl;
      }
    }
  }  // end while loop

  //}
  BarrierWait(&all_threads_barrier_);
  time(&t2);
  if (id == num_threads) {
    pthread_mutex_lock(&io_lock_);
    if (step == kMaxSteps) {
      std::cout << "Convergence failed! (steps = " << step << ", diff = " << diff_th
           << ")\n";
      outfile_ << "#Info: Total steps = " << step << ". No convergence.\n";
    } else {
      std::cout << "Converged to diff = " << diff_th << " in " << step
           << " steps.\n";
      outfile_ << "#Info: Converged to diff = " << diff_th << " in " << step
               << " steps.\n";
    }

    if (verbose_ > 0) std::cout << "Writing final steady-state density matrix.\n";

    outfile_ << "#Info: Total running time: " << difftime(t2, t1) / 60 / 60 / 24
             << " days.\n";

    outfile_ << diff_th << " ";
    for (int i = 0; i < num_states_ * num_states_; ++i) {
      outfile_ << nodes_[0].steady_state_density_matrix[i] << " ";
    }
    outfile_ << std::endl;

    if (verbose_ > 1)
      PrintMatrix(nodes_[0].steady_state_density_matrix, num_states_);
    if (verbose_ > 2) {
      for (int m = 1; m < matrix_count_; ++m) {
        std::cout << m << ":\n";
        PrintMatrix(nodes_[m].steady_state_density_matrix, num_states_);
      }
    }
    std::cout << "Total running time: " << difftime(t2, t1) / 60 / 60 << " hours.\n";
    if (parameters_->restart_output_time > 0) {
      restart_helper_->WriteRestartFile((Float)step, 0, nodes_, matrix_count_,
                                        NULL);
    }
    pthread_mutex_unlock(&io_lock_);
  }
  BarrierWait(&all_threads_barrier_);
  pthread_exit(0);
}

HierarchyIntegrator::~HierarchyIntegrator() {
  // TODO(johanstr): Fix the clean-up code here and in the integrate functions
  // to not leak memory.
}
