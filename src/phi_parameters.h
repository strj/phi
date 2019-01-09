#ifndef PHI_PHI_PARAMETERS_H_
#define PHI_PHI_PARAMETERS_H_

#include <string.h>   // for strcmp

#include <fstream>
#include <iostream>
#include <string>

#include "numeric_types.h"
#include "parameter_input.h"

// Reduced Planck constant in [cm-1*ns]
#define HBAR 5.3088354
// Boltzmann constant in [cm-1/Kelvin]
#define K_B 0.6950344

enum RunMethod {
  BROKENINPUT,
  NONE,
  RK4,
  RKF45,
  RK4SPECTRUM,
  RKF45SPECTRUM,
  BICGSTABL,
  BICGSTAB,
  BICGSTABU,
  PRINTHIERARCHY,
  PRINTMEMORY
};

struct PhiParameters {
    int verbose;

    Float hbar;
    Float boltzmann_constant;

    phi_big_size_t num_states;
    phi_big_size_t hilbert_space_num_elements;
    phi_big_size_t hilbert_space_size;
    phi_size_t num_bath_couplings;
    phi_size_t num_matsubara_terms;

    phi_size_t hierarchy_truncation_level;
    phi_big_size_t matrix_count;

    Complex *time_indepenent_hamiltonian;
    Complex *final_annealing_hamiltonian;
    Complex *initial_annealing_hamiltonian;
    phi_size_t initial_hamiltonian_field_size;
    Float *initial_hamiltonian_field;
    phi_size_t final_hamiltonian_field_size;
    Float *final_hamiltonian_field;
    Float minimum_field_change;

    Float temperature;

    Float *kappa;
    Float *gamma;
    Float *lambda;
    Float *bath_coupling_op;
    phi_size_t *multiple_independent_bath_indices;

    Complex *initial_density_matrix;
    Complex *spectrum_initial_density_matrix;

    Float integration_time;
    Float timestep;
    Float output_minimum_timestep;
    Float start_time;
    Float restart_output_time;  // Wall clock time between writing restart info.
    Float rkf45_minimum_timestep;
    Float rkf45_tolerance;
    // Tolerance used by steady state solver.
    Float bicgstab_tolerance;
    // Tolerance for Shi matrix truncation scheme.
    Float filter_tolerance;

    std::string output_filename;
    std::string restart_input_filename;
    std::string restart_output_filename;
    std::string restart_backup_filename;

    bool is_truncated_system;

    bool use_time_local_truncation;
    bool is_multiple_independent_baths;

    bool is_spectrum_calculation;
    bool is_diagonal_bath_coupling;
    bool is_full_bath_coupling;

    bool is_rho_normalized;
    bool is_restarted;

    Complex *hamiltonian_eigenvectors, *hamiltonian_eigenvalues;
    bool are_e_vecs_present;
    bool are_e_vals_present;

    int bicgstab_ell;

    int num_threads;
    int num_cores;
    int *cpuaffinity;

    bool stupid_hierarchy_partition;

    bool are_parameters_valid;

    RunMethod run_method;

    void SetDefaults();
    bool FileExists(const char * filename);
    void FixIncompatibleCombinations();

    PhiParameters();
    ~PhiParameters();
    void ReadCommandLine(int num_args, char *argv[]);
    void ReadInputFile(std::string parameter_file);
    void CheckParameters();
    void Write(std::ofstream * out, int num_matrices);
};

phi_big_size_t PartialFactorial(phi_size_t, phi_size_t);

phi_big_size_t GetHierarchyNodeCountEstimate(phi_size_t, phi_size_t);

#endif  // PHI_PHI_PARAMETERS_H_

