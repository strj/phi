#include "phi_parameters.h"

#include "complex_matrix.h"

PhiParameters::PhiParameters() {
  SetDefaults();
}

void PhiParameters::SetDefaults() {
  verbose = 1;

  // Overridable constants so that we can work with arbitrary units.
  hbar = HBAR;
  boltzmann_constant = K_B;

  num_states = 0;  // Number of System states.
  // Size of the Hilbert space. Usually equal to Ns, but could be bigger for
  // truncated system calculations.
  hilbert_space_size = 0;
  hilbert_space_num_elements = 0;
  num_bath_couplings = 0;  // Number of System-Bath coupling terms.
  num_matsubara_terms = 1;  // Number of Matsubara terms + 1; 1 corresponds to
                            // high temperature approximation (no extra terms).
  hierarchy_truncation_level = 1;  // Hierarchy level to truncate at.

  time_indepenent_hamiltonian = NULL;
  initial_annealing_hamiltonian = NULL;
  final_annealing_hamiltonian = NULL;
  final_hamiltonian_field_size = 0;
  final_hamiltonian_field = NULL;
  initial_hamiltonian_field_size = 0;
  initial_hamiltonian_field = NULL;
  minimum_field_change = 0.01;

  temperature = 300;  // Kelvin.
  kappa = NULL;
  gamma = NULL;
  lambda = NULL;
  bath_coupling_op = NULL;
  multiple_independent_bath_indices = NULL;

  initial_density_matrix = NULL;
  spectrum_initial_density_matrix = NULL;

  integration_time = 0;
  timestep = 0.001;  // Integration timestep for RK4 integration.
  output_minimum_timestep = 0.01;  // The minimum stepsize during output.
  start_time = 0;
  restart_output_time = 0;
  rkf45_minimum_timestep = 1e-6;  // Minimum timestep for RKF45 integration.
  rkf45_tolerance = 1e-6;  // Tolerance for RKF45 integration.

  bicgstab_tolerance = 1e-6;  // Tolerance for BiCGSTAB linear system solver.

  filter_tolerance = 0;  // Shi et al 2009 hierarchy truncation.

  use_time_local_truncation = false;
  is_multiple_independent_baths = false;

  is_truncated_system = false;
  is_spectrum_calculation = false;
  is_diagonal_bath_coupling = false;
  is_full_bath_coupling = false;

  is_rho_normalized = true;
  is_restarted = false;

  are_e_vecs_present = false;
  are_e_vals_present = false;

  bicgstab_ell = 2;

  num_threads = 1;
  num_cores = 1;
  cpuaffinity = NULL;

  stupid_hierarchy_partition = false;

  are_parameters_valid = true;

  run_method = NONE;

  restart_input_filename = "";
  restart_output_filename = "";
  restart_backup_filename = "";
}

bool PhiParameters::FileExists(const char *filename) {
  std::ifstream ifile(filename);
  return ifile.is_open();
}

void PhiParameters::ReadCommandLine(int num_args, char *argv[]) {
  std::stringstream stream;
  if (num_args < 2) {
    std::cout << "Usage: \n\n  phi paramfile {integrator [threads],print}\n"
         << "  integrator  - One of rk4, rkf45, rk4spectrum, rkf45spectrum, "
         << "steadystate.\n"
         << "  threads     - Number of threads\n"
         << "  memory      - Prints memory requirements\n"
         << "  print       - Prints hierarchy connections\n";
    are_parameters_valid = false;
    return;
  }
  std::cout << "Reading parameters from " << argv[1] << "\n\n";
  if (!FileExists(argv[1])) {
    are_parameters_valid = false;
    std::cout << "ERROR: No parameter file.\n";
    return;
  }

  if (num_args == 2) {
    std::cout << "No integration method specified.\n";
    return;  // Just checks input file and returns.
  }

  if (num_args == 4) {
    // Check input value ...
    stream.str(argv[3]);
    stream >> num_threads;
  }

  // Integration method.
  if ((strcmp(argv[2], "rk4spectrum")) == 0) {
    run_method = RK4SPECTRUM;
  } else if ((strcmp(argv[2], "rkf45spectrum")) == 0) {
    run_method = RKF45SPECTRUM;
  } else if ((strcmp(argv[2], "rk4")) == 0) {
    run_method = RK4;
  } else if ((strcmp(argv[2], "rkf45")) == 0) {
    run_method = RKF45;
  } else if (strcmp(argv[2], "print") == 0) {
    run_method = PRINTHIERARCHY;
  } else if (strcmp(argv[2], "memory") == 0) {
    run_method = PRINTMEMORY;
  } else if (strcmp(argv[2], "bicgstabl") == 0) {
    run_method = BICGSTABL;
  } else if (strcmp(argv[2], "bicgstab") == 0) {
    run_method = BICGSTABL;
  } else if (strcmp(argv[2], "bicgstabu") == 0) {
    run_method = BICGSTABU;
  }

  // Move to a separate read.
  ReadInputFile(argv[1]);
}

void ReportInputFormatError(const std::string &var_name) {
  std::cerr << "ERROR: incorrect " << var_name << " input format.\n";
}

void PhiParameters::ReadInputFile(std::string parameter_file) {
  std::ifstream input_stream;
  input_stream.open(parameter_file.c_str());

  std::string line, var_name, var;
  std::stringstream stream;

  int matrix_row_index = 0;

  getline(input_stream, line);
  ParameterInput parser;
  while (!input_stream.eof() && are_parameters_valid) {
    while (line.find("#") != std::string::npos) {
      // Skip lines containing comments.
      var = "";
      getline(input_stream, line);
    }

    // Determing if the input line is for a scalar or array parameter.
    if (line.find("=") != std::string::npos) {
      // Scalar parameter input.
      const int separator_position = line.find("=");
      var_name = line.substr(0, separator_position);
      var = line.substr(separator_position + 1, line.length());
    } else if (line.find(":") != std::string::npos) {
      // Array parameter input start.
      const int separator_position = line.find(":");
      var_name = line.substr(0, separator_position);
      matrix_row_index = 0;
      getline(input_stream, var);
    } else if (line.size() > 0) {
      // Array parameter input continuation.
      var = line;
    } else {
      // Blank line.
      var_name = "";
    }

    getline(input_stream, line);
    stream.clear();
    if (var_name == "fout" || var_name == "Output" ||
        var_name == "OutputFile") {
      output_filename = var;
      if (verbose > 0) {
        std::cout << "Output being written to " << var << "\n";
      }
      // Add code to make backup of file if it exists.
    } else if (var_name == "Ns" || var_name == "NumStates") {
      are_parameters_valid = parser.Get(var, var_name, &num_states,
                                        parser.POSITIVE);
      if (!are_parameters_valid) {
        break;
      }
      if (verbose > 0) {
        std::cout << "Number of states = " << num_states << "\n";
      }
      if (hilbert_space_size == 0) {
        hilbert_space_size = num_states;
        hilbert_space_num_elements = num_states * num_states;
      }
      if (num_bath_couplings == 0) {
        num_bath_couplings = num_states;
      }
      multiple_independent_bath_indices = new int[num_bath_couplings];
      for (int i = 0; i < num_bath_couplings; ++i) {
        multiple_independent_bath_indices[i] = i;
      }
    } else if (var_name == "HilbertSpaceSize") {
      if (time_indepenent_hamiltonian != NULL ||
          final_annealing_hamiltonian != NULL ||
          final_annealing_hamiltonian != NULL) {
        std::cerr << "ERROR: HilbertSpaceSize must be specified before the "
             << "Hamiltonian.\n";
        are_parameters_valid = false;
        break;
      } else if (is_diagonal_bath_coupling || is_full_bath_coupling) {
        std::cerr << "ERROR: HilbertSpaceSize must be specified before the "
             << "bath coupling operators.\n";
        are_parameters_valid = false;
        break;
      }
      are_parameters_valid = parser.Get(var, var_name, &hilbert_space_size,
                                        parser.POSITIVE);
      hilbert_space_num_elements = hilbert_space_size * hilbert_space_size;
      if (hilbert_space_size < num_states) {
        std::cerr << "ERROR: HilbertSpaceSize cannot be less than NumStates.\n";
        are_parameters_valid = false;
      }
    } else if (var_name == "M" || var_name == "NumCouplingTerms") {
      are_parameters_valid = parser.Get(var, var_name, &num_bath_couplings,
                                         parser.POSITIVE);
      if (!are_parameters_valid) {
        break;
      }
      if (verbose > 0) {
        std::cout << "Number of system-bath coupling terms = "
             << num_bath_couplings << "\n";
      }
      if (num_states != 0) {
        delete []multiple_independent_bath_indices;
        multiple_independent_bath_indices = new int[num_bath_couplings];
        for (int i = 0; i < num_bath_couplings; ++i) {
          multiple_independent_bath_indices[i] = i;
        }
      }
    } else if (var_name == "Kt" || var_name == "MatsubaraTerms") {
      are_parameters_valid = parser.Get(var, var_name, &num_matsubara_terms,
                                        parser.NON_NEGATIVE);
      if (!are_parameters_valid) {
        break;
      }
      if (verbose > 0) {
        std::cout << "Number of Matsubara terms = " << num_matsubara_terms << "\n";
      }
      num_matsubara_terms += 1;
    } else if (var_name == "Lh" || var_name == "HierarchyTruncation") {
      are_parameters_valid = parser.Get(var, var_name,
                                        &hierarchy_truncation_level,
                                        parser.POSITIVE);
      if (are_parameters_valid) {
        if (verbose > 0)
          std::cout << "Hierarchy truncation = " << hierarchy_truncation_level << "\n";
        if (hierarchy_truncation_level < 1) {
          std::cerr << "ERROR: HierarchyTruncation must be positive.\n";
          are_parameters_valid = false;
          break;
        }
      }
    } else if (var_name == "dt" || var_name == "Timestep") {
      are_parameters_valid = parser.Get(var, var_name, &timestep,
                                        parser.POSITIVE);
      if (are_parameters_valid) {
        if (verbose > 0) {
          std::cout << "Timestep = " << timestep << " ps.\n";
        }
      }
      if (timestep < 1e-10) {
        std::cout << "Warning: timestep less than 1e-10 ps.\n";
      }
    } else if (var_name == "t" || var_name == "Time") {
      are_parameters_valid = parser.Get(var, var_name, &integration_time,
                                        parser.POSITIVE);
      if (are_parameters_valid) {
        if (verbose > 0)
          std::cout << "Runlength = " << integration_time << " ps.\n";
      }
    } else if (var_name == "OutputMinimumTimestep") {
      are_parameters_valid = parser.Get(
          var, var_name, &output_minimum_timestep, parser.POSITIVE);
      if (are_parameters_valid) {
        if (verbose > 0)
          std::cout << "Minimum timestep in output = "
               << output_minimum_timestep << " ps.\n";
      }

    } else if (var_name == "filter" || var_name == "filtered" ||
               var_name == "FilterTolerance") {
      // Shi et al 2009 truncation.
      are_parameters_valid = parser.Get(var, var_name, &filter_tolerance,
                                        parser.POSITIVE);
      if (are_parameters_valid) {
        if (verbose > 0) {
          std::cout << "Setting ADM Filter tolerance to "
               << filter_tolerance << ".\n";
        }
      }
    } else if (var_name == "TL" || var_name == "TimeLocal") {
      are_parameters_valid = parser.Get(var, var_name,
                                        &use_time_local_truncation);
      if (are_parameters_valid) {
        if (verbose > 0 && use_time_local_truncation) {
          std::cout << "Using time local truncation.\n";
        } else if (verbose > 0) {
          std::cout << "Using time non-local truncation.\n";
        }
      }
    } else if (var_name == "rho_normalized" || var_name == "rho_normalised") {
      are_parameters_valid = parser.Get(var, var_name, &is_rho_normalized);
      if (verbose > 0 && are_parameters_valid) {
        std::cout << ((is_rho_normalized) ? "Is " : "Not ")
             << "using rho hierarchy scaling.\n";
      }
    } else if (var_name == "H" || var_name == "Hamiltonian") {
      if (num_states <= 0) {
        std::cerr << "ERROR: NumStates must be specified before Hamiltonian.\n";
        are_parameters_valid = false;
        break;
      }
      if (initial_annealing_hamiltonian != NULL ||
          final_annealing_hamiltonian != NULL ||
          (time_indepenent_hamiltonian != NULL && matrix_row_index == 0)) {
        std::cerr << "ERROR: Hamiltonian already specified\n";
        are_parameters_valid = false;
        break;
      }
      if (matrix_row_index == 0) {
        time_indepenent_hamiltonian = new Complex[hilbert_space_num_elements];
        SetElementsToZero(hilbert_space_num_elements,
                          time_indepenent_hamiltonian);
      } else if (matrix_row_index >= hilbert_space_size) {
        std::cerr << "ERROR: Overflow in Hamiltonian input.\n";
        are_parameters_valid = false;
        break;
      }
      if (!parser.Get(var, time_indepenent_hamiltonian + matrix_row_index,
                      hilbert_space_size, hilbert_space_size)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      ++matrix_row_index;
    } else if (var_name == "H_X" || var_name == "TransverseHamiltonian") {
      if (num_states <= 0) {
        std::cerr << "ERROR: NumStates must be specified before "
             << var_name << "\n";
        are_parameters_valid = false;
        break;
      } else if (matrix_row_index >= hilbert_space_size) {
        std::cerr << "ERROR: Overflow in TransverseHamiltonian input.\n";
        are_parameters_valid = false;
        break;
      } else if (matrix_row_index == 0) {
        initial_annealing_hamiltonian = new Complex[hilbert_space_num_elements];
        SetElementsToZero(hilbert_space_num_elements,
                          initial_annealing_hamiltonian);
      }
      if (!parser.Get(var, initial_annealing_hamiltonian + matrix_row_index,
                      hilbert_space_size, hilbert_space_size)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      ++matrix_row_index;
    } else if (var_name == "TransverseField") {
      int count = parser.GetCharacterOccurence(',', var) + 1;
      if (count <= 0) {
        std::cerr << "ERROR: Bad input for TransverseField: " << count << ".\n";
        are_parameters_valid = false;
        break;
      }
      initial_hamiltonian_field_size = count;
      initial_hamiltonian_field = new Float[count];
      if (!parser.Get(var, initial_hamiltonian_field, count, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
    } else if (var_name == "H_Z" || var_name == "LongitudinalHamiltonian") {
      if (num_states <= 0) {
        std::cerr << "ERROR: Ns must be specified before " << var_name << ".\n";
        are_parameters_valid = false;
        break;
      } else if (matrix_row_index >= hilbert_space_size) {
        std::cerr << "ERROR: Overflow in LongitudinalHamiltonian input.\n";
        are_parameters_valid = false;
        break;
      } else if (matrix_row_index == 0) {
        final_annealing_hamiltonian = new Complex[hilbert_space_num_elements];
        SetElementsToZero(hilbert_space_num_elements,
                          final_annealing_hamiltonian);
      }
      if (!parser.Get(var, final_annealing_hamiltonian + matrix_row_index,
                      hilbert_space_size, hilbert_space_size)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      ++matrix_row_index;
    } else if (var_name == "LongitudinalField") {
      int count = parser.GetCharacterOccurence(',', var) + 1;
      if (count <= 0) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      final_hamiltonian_field_size = count;
      final_hamiltonian_field = new Float[count];
      if (!parser.Get(var, final_hamiltonian_field, count, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
    } else if (var_name == "MinimumFieldChange") {
      are_parameters_valid = parser.Get(
          var, var_name, &minimum_field_change, parser.NON_NEGATIVE);
      if (are_parameters_valid && verbose > 0) {
        std::cout << "Setting minimum Hamiltonian change fraction to "
             << minimum_field_change << "\n";
      }
    } else if (var_name == "rho0" ||
               var_name == "InitialDensityMatrix") {
      if (num_states <= 0) {
        std::cerr << "ERROR: Ns must be specified before InitialDensityMatrix.\n";
        are_parameters_valid = false;
        break;
      }
      if (initial_density_matrix != NULL && matrix_row_index == 0) {
        std::cerr << "ERROR: InitialDensityMatrix already specified\n";
        are_parameters_valid = false;
        break;
      }
      if (matrix_row_index == 0) {
        initial_density_matrix = new Complex[hilbert_space_num_elements];
        SetElementsToZero(hilbert_space_num_elements, initial_density_matrix);
      } else if (matrix_row_index >= hilbert_space_size) {
        std::cerr << "ERROR: Overflow in " << var_name << " input.\n";
        are_parameters_valid = false;
        break;
      }

      if (!parser.Get(var, initial_density_matrix + matrix_row_index,
                      hilbert_space_size, hilbert_space_size)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      ++matrix_row_index;
    } else if (var_name == "spectrum_rho0" &&
               spectrum_initial_density_matrix == NULL) {
      if (num_states <= 0) {
        std::cerr << "ERROR: NumStates must be specified before spectrum_rho0.\n";
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Specified spectrum calculation initial state.\n";
      }
      spectrum_initial_density_matrix = new Complex[hilbert_space_size];
      if (!parser.Get(
              var, spectrum_initial_density_matrix, hilbert_space_size, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
      }
    } else if (var_name == "Multibath" || var_name == "multiBath" ||
               var_name == "DiagonalCouplingIndices") {
      // This allows you to specify multiple, independent baths for each basis
      // state.
      // "DiagonalCouplingIndices=0,0,0,1" => 3 Baths couple to stat 0 and 1
      // couple to state 1.
      if (num_bath_couplings <= 0) {
        std::cerr << "ERROR: NumCouplingTerms must be specified before "
             << var_name << ".\n";
        are_parameters_valid = false;
        break;
      }
      if (!parser.Get(
              var, multiple_independent_bath_indices, num_bath_couplings, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
    } else if (var_name == "Vbath" ||
               var_name == "DiagonalCouplingTerms" ||
               var_name == "CorrelatedCouplingTerms") {
      if (num_bath_couplings <= 0 || num_states <= 0) {
        std::cerr << "ERROR: NumCouplingTerms and NumStates must be specified "
             << "before " << var_name << ".\n";
        are_parameters_valid = false;
        break;
      }
      if (bath_coupling_op != NULL && matrix_row_index == 0) {
        std::cerr << "ERROR: Bath coupling operators already specified\n";
        are_parameters_valid = false;
        break;
      }
      if (matrix_row_index == 0) {
        is_diagonal_bath_coupling = true;
        bath_coupling_op = new Float[num_bath_couplings * hilbert_space_size];
        SetElementsToZero(num_bath_couplings * hilbert_space_size,
                          bath_coupling_op);
      } else if (matrix_row_index >= num_bath_couplings) {
        std::cerr << "ERROR: Overflow in " << var_name << " input.\n";
        are_parameters_valid = false;
        break;
      }
      if (!parser.Get(var,
                      bath_coupling_op + hilbert_space_size * matrix_row_index,
                      hilbert_space_size, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      ++matrix_row_index;
    } else if (var_name == "FullBathTerms") {
      if (num_bath_couplings <= 0 || num_states <= 0) {
        std::cerr << "ERROR: NumCouplingTerms and NumStates must be specified "
             << "before " << var_name << ".\n";
        are_parameters_valid = false;
        break;
      }
      if (bath_coupling_op != NULL && matrix_row_index == 0) {
        std::cerr << "ERROR: Bath coupling operators already specified\n";
        are_parameters_valid = false;
        break;
      }
      if (matrix_row_index == 0) {
        is_full_bath_coupling = true;
        bath_coupling_op =
            new Float[num_bath_couplings * hilbert_space_num_elements];
        SetElementsToZero(num_bath_couplings * hilbert_space_num_elements,
                          bath_coupling_op);
      } else if (matrix_row_index >= num_bath_couplings) {
        std::cerr << "ERROR: Overflow in " << var_name << " input.\n";
        are_parameters_valid = false;
        break;
      }
      phi_big_size_t line = hilbert_space_num_elements * matrix_row_index;
      if (!parser.Get(
          var, bath_coupling_op + line, hilbert_space_num_elements, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      ++matrix_row_index;
    } else if (var_name == "RestartInput") {
      stream.str(var);
      stream >> restart_input_filename;
      is_restarted = true;
      if (verbose > 0) {
        std::cout << "Restarting calculation using "
             << restart_input_filename << ".\n";
      }
    } else if (var_name == "lambda" || var_name == "Lambda") {
      if (num_bath_couplings <= 0) {
        std::cerr << "ERROR: NumCouplingTerms must be specified before "
             << var_name << ".\n";
        are_parameters_valid = false;
        break;
      }
      if (lambda != NULL) {
        std::cerr << "ERROR: " << var_name << " already specified\n";
        are_parameters_valid = false;
        break;
      }
      lambda = new Float[num_bath_couplings];
      SetElementsToZero(num_bath_couplings, lambda);
      if (!parser.Get(var, lambda, num_bath_couplings, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Read coupling strengths lambda.\n";
      }
      if (verbose > 1) {
        for (int i = 0; i < num_bath_couplings; ++i) {
          std::cout << "lambda[" << i << "] = " << lambda[i] << "\n";
        }
      }
    } else if (var_name == "gamma" || var_name == "Gamma") {
      if (num_bath_couplings <= 0) {
        std::cerr << "ERROR: NumCouplingTerms must be specified before "
             << var_name << ".\n";
        are_parameters_valid = false;
        break;
      }
      if (gamma != NULL) {
        std::cerr << "ERROR: " << var_name << " already specified\n";
        are_parameters_valid = false;
        break;
      }
      gamma = new Float[num_bath_couplings];
      SetElementsToZero(num_bath_couplings, gamma);
      if (!parser.Get(var, gamma, num_bath_couplings, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Read response frequencies gamma.\n";
      }
      if (verbose > 1) {
        for (int i = 0; i < num_bath_couplings; ++i) {
          std::cout << "gamma[" << i << "] = " << gamma[i] << "\n";
        }
      }
    } else if (var_name == "kappa") {
      if (num_states <= 0) {
        std::cerr << "ERROR: NumStates must be specified before "
             << var_name << ".\n";
        are_parameters_valid = false;
        break;
      }
      if (kappa != NULL) {
        std::cerr << "ERROR: Kappa already defined.\n";
        are_parameters_valid = false;
        break;
      }
      kappa = new Float[hilbert_space_size];
      SetElementsToZero(hilbert_space_size, kappa);
      if (!parser.Get(var, kappa, hilbert_space_size, 1)) {
        ReportInputFormatError(var_name);
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Read decay rates.\n";
      }
      if (verbose > 1) {
        for (int i = 0; i < hilbert_space_size; ++i) {
          std::cout << "kappa[" << i << "] = " << kappa[i] << "\n";
        }
      }
    } else if (var_name == "T" || var_name == "Temperature") {
      if (!parser.Get(var, var_name, &temperature, parser.NON_NEGATIVE)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Temperature = " << temperature << "\n";
      }
    } else if (var_name == "verbose") {
      stream.str(var);
      stream >> verbose;
    } else if (var_name == "RestartFile") {
      stream.str(var);
      stream >> restart_output_filename;
      if (verbose > 0) {
        std::cout << "Restart Output File = "
             << restart_output_filename << "\n";
      }
    } else if (var_name == "RestartOutputTimeSecs") {
      if (!parser.Get(var, var_name, &restart_output_time,
                      parser.POSITIVE)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Writing restart file every "
             << restart_output_time << " seconds.\n";
      }
    } else if (var_name == "tol_low" || var_name == "RKF45mindt") {
      if (!parser.Get(var, var_name, &rkf45_minimum_timestep,
                      parser.POSITIVE)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "Minimum RKF45 timestep = " << rkf45_minimum_timestep << "\n";
        // Compatibility with steady state calculations.
        if (var_name == "tol_low") {
          bicgstab_tolerance = rkf45_minimum_timestep;
          std::cout << "bicgstab tolerance = " << bicgstab_tolerance << "\n";
        }
      }
    } else if (var_name == "tol_high" || var_name == "RKF45tolerance") {
      if (!parser.Get(var, var_name, &rkf45_tolerance, parser.POSITIVE)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "RKF45 integration tolerance = " << rkf45_tolerance << "\n";
      }
    } else if (var_name == "bicgstab_tolerance") {
      if (!parser.Get(var, var_name, &bicgstab_tolerance,
                      parser.POSITIVE)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        std::cout << "bicgstab tolerance = " << bicgstab_tolerance << "\n";
      }
    } else if (var_name == "bicgstab") {
      if (!parser.Get(var, var_name, &bicgstab_ell,
                      parser.NON_NEGATIVE)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 0) {
        if (bicgstab_ell > 0) {
          std::cout << "BiCGSTAB(" << bicgstab_ell
               << ") will be used for steadystate.\n";
        } else if (bicgstab_ell == 0) {
          std::cout << "BiCGSTAB will be used for steadystate.\n";
        }
      }
    } else if (var_name == "CyclicPartition") {
       if (!parser.Get(var, var_name, &stupid_hierarchy_partition)) {
        are_parameters_valid = false;
        break;
       }
      if (verbose > 0 && stupid_hierarchy_partition) {
        std::cout << "Using simple hierarchy partition (round-robin).\n";
      }
#ifdef THREADAFFINITY
    } else if (var_name == "cpuaffinity" && cpuaffinity == NULL) {
      num_cores = getCharacterOccurent(',', var);
      cpuaffinity = new int[num_cores];
      if (!parser.Get(var, cpuaffinity, num_cores, 1)) {
        are_parameters_valid = false;
        break;
      }
      if (verbose > 1) {
        for (int n = 0; n < num_cores; ++n) {
          cout << "cpuaffinity[" < <n<< "]=" << cpuaffinity[n] << "\n";
        }
      }
#endif
    } else if (var_name == "hbar") {
      if (!parser.Get(var, var_name, &hbar, parser.POSITIVE)) {
        are_parameters_valid = false;
        break;
      }
    } else if (var_name == "kB" || var_name == "k_B") {
      if (!parser.Get(var, var_name, &boltzmann_constant, parser.POSITIVE)) {
        are_parameters_valid = false;
        break;
      }
    }
  }

  input_stream.close();

  if (verbose > 1) {
    std::cout << "Input file read.\n";
  }

  if (!are_parameters_valid) {
    std::cout << "Invalid parameters.\n";
    return;
  }

  // TODO(johanstr): Fix checkParameters.
  // checkParameters();
  if (run_method != PRINTMEMORY) {
    FixIncompatibleCombinations();
  }
  std::cout << "Getting matrix count estimate.\n";
  phi_size_t num_elements = num_bath_couplings * num_matsubara_terms;
  matrix_count = GetHierarchyNodeCountEstimate(hierarchy_truncation_level,
                                               num_elements);
  std::cout << "Done setting parameters.\n";
}

void PhiParameters::FixIncompatibleCombinations() {
  if (verbose > 1) {
    std::cout << "Checking validity of input parameters.\n";
  }
  are_parameters_valid =
      !(time_indepenent_hamiltonian == NULL &&
          (initial_annealing_hamiltonian == NULL ||
           final_annealing_hamiltonian == NULL));

  if (!are_parameters_valid && run_method != NONE) {
     std::cout << "Invalid parameters, no Hamiltonian?.\n";
     run_method = BROKENINPUT;
     return;
  }

  if (initial_annealing_hamiltonian != NULL &&
      initial_hamiltonian_field == NULL) {
    initial_hamiltonian_field_size = 2;
    initial_hamiltonian_field = new Float[initial_hamiltonian_field_size];
    initial_hamiltonian_field[0] = 1.0;
    initial_hamiltonian_field[1] = 0.0;
  }

  if (initial_annealing_hamiltonian != NULL &&
      final_hamiltonian_field == NULL) {
    final_hamiltonian_field_size = 2;
    final_hamiltonian_field = new Float[final_hamiltonian_field_size];
    final_hamiltonian_field[0] = 0.0;
    final_hamiltonian_field[1] = 1.0;
  }

  if ((run_method == RKF45 ||
       run_method == RKF45SPECTRUM ||
       run_method == BICGSTAB ||
       run_method == BICGSTABL ||
       run_method == BICGSTABU) &&
      !is_rho_normalized) {
    // Force using renormalized auxiliary density matrices for adaptive timestep
    // method, or steady state solver.
    if (verbose > 0) {
      std::cout << "Forcing rho hierarchy scaling.\n";
    }
    is_rho_normalized = true;
  }

  if (run_method == BICGSTAB ||
      run_method == BICGSTABL ||
      run_method == BICGSTABU) {
    // Can't do steady state solver with time local truncation.
    use_time_local_truncation = false;
    std::cout << "Forcing time non-local truncation for steady state solver.\n";
  }

  if (run_method == BICGSTABL && bicgstab_ell < 1) {
    run_method = BICGSTAB;
  }

  if (num_cores < num_threads && cpuaffinity != NULL) {
    std::cout << "WARNING: Trying to run " << num_threads << " threads on a "
         << " computer with " << num_cores << " cores.\n";
  }
  if (filter_tolerance > 0 &&
      (is_diagonal_bath_coupling || is_full_bath_coupling)) {
    filter_tolerance = 0;
    std::cout << "WARNING: Setting filter_tolerance to 0. Correlated/Full bath "
         << "coupling not implemented with dynamic hierarchy truncation.\n";
  }
  is_spectrum_calculation = run_method == RK4SPECTRUM ||
                            run_method == RKF45SPECTRUM;
  if (is_spectrum_calculation) {
    if (initial_density_matrix != NULL &&
        spectrum_initial_density_matrix == NULL) {
      // Fill initial state of spectrum calculation with diagonal of the initial
      // density matrix.
      spectrum_initial_density_matrix = new Complex[hilbert_space_size];
      for (int i = 0; i < hilbert_space_size; ++i) {
        spectrum_initial_density_matrix[i] =
            initial_density_matrix[i * hilbert_space_size + i];
      }
    } else if (spectrum_initial_density_matrix == NULL) {
      // Fill initial state of spectrum calculation with ones.
      spectrum_initial_density_matrix = new Complex[hilbert_space_size];
      for (int i = 0; i < hilbert_space_size; ++i) {
        spectrum_initial_density_matrix[i] = Complex(1);
      }
    }
  }
  // Move to ProcessParameters:
  if (multiple_independent_bath_indices == NULL) {
    multiple_independent_bath_indices = new int[num_bath_couplings];
    for (int i = 0; i < num_bath_couplings; ++i) {
      multiple_independent_bath_indices[i] = i;
    }
  }

  if (hilbert_space_size < num_states) {
    hilbert_space_size = num_states;
    hilbert_space_num_elements = num_states * num_states;
  } else if (hilbert_space_size > num_states) {
    is_truncated_system = true;
  }
  // Check bath_coupling_op is Hermitian.
}

void PhiParameters::CheckParameters() {
  const std::string input_error = "INPUT ERROR ";
  if (num_states <= 0) {
    std::cerr << input_error << "NumStates must be > 0\n";
    are_parameters_valid = false;
  }
  if (hierarchy_truncation_level < 1) {
    std::cerr << input_error << "HierarchyTruncation must be > 0\n";
    are_parameters_valid = false;
  }
  if (num_bath_couplings <= 0) {
    std::cerr << input_error << "NumCouplingTerms must be > 0\n";
    are_parameters_valid = false;
  }
  if (num_matsubara_terms < 0) {
    std::cerr << input_error << "NumMatsubaraTerms must be >= 0\n";
    are_parameters_valid = false;
  }
  if (time_indepenent_hamiltonian == NULL ||
      (initial_annealing_hamiltonian == NULL &&
       final_annealing_hamiltonian == NULL)) {
    std::cerr << input_error << "Hamiltonian must be specified\n";
    are_parameters_valid = false;
  }
  if (gamma == NULL) {
    std::cerr << input_error << "Gamma values must be specified\n";
    are_parameters_valid = false;
  }
  if (lambda == NULL) {
    std::cerr << input_error << "Lambda values must be specified\n";
    are_parameters_valid = false;
  }
  if (initial_density_matrix == NULL &&
      spectrum_initial_density_matrix == NULL && is_spectrum_calculation) {
    std::cerr << input_error << "Initial state must be specified\n";
    are_parameters_valid = false;
  }
  if (use_time_local_truncation && is_restarted) {
    std::cerr << input_error
         << "Restarting time-local calculations truncation not implemented.\n";
    are_parameters_valid = false;
  }
}

void PhiParameters::Write(std::ofstream *out, int num_matrices) {
  if (!out->is_open()) {
    return;
  }
  double version = 1.1;
  *out << "#PHI " << version << "\n";
  *out << "#Parameters:\n";
  *out << "#-----------\n";
  if (hbar != HBAR) {
    *out << "#ReducedPlanckConstant=" << hbar << "\n";
  }
  if (boltzmann_constant != K_B) {
    *out << "#BoltzmannConstant=" << boltzmann_constant << "\n";
  }
  *out << "#NumStates=" << num_states << "\n";
  if (hilbert_space_size != num_states) {
    *out << "#HilbertSpaceSize=" << hilbert_space_size << "\n";
  }
  *out << "#MatsubaraTerms=" << num_matsubara_terms-1 << "\n";
  *out << "#CouplingTerms=" << num_bath_couplings << "\n";
  *out << "#Temperature=" << temperature << "\n";
  *out << "#HierarchyTruncation=" << hierarchy_truncation_level << "\n";
  *out << "#TimeLocal=" << use_time_local_truncation << "\n";
  *out << "#Timestep=" << timestep << "\n";
  *out << "#Time=" << integration_time << "\n";
  *out << "#filter=" << filter_tolerance << "\n";
  *out << "#RKF45tolerance = " << rkf45_tolerance << "\n";
  *out << "#RKF45mindt = " << rkf45_minimum_timestep << "\n";
  if (gamma) {
    *out << "#gamma:\n";
    *out << "#";
    for (int i = 0; i < num_bath_couplings - 1; ++i) {
      *out << gamma[i] << ",";
    }
    *out << gamma[num_bath_couplings - 1] <<"\n";
  }
  if (lambda) {
    *out << "#lambda:\n";
    *out << "#";
    for (int i = 0; i < num_bath_couplings - 1; ++i) {
      *out << lambda[i] << ",";
    }
    *out << lambda[num_bath_couplings - 1] << "\n";
  }
  if (kappa) {
    *out << "#kappa:\n";
    *out << "#";
    for (int i = 0; i < num_states - 1; ++i) {
      *out << kappa[i] << ",";
    }
    *out << kappa[num_states - 1] << "\n";
  }
  if (initial_hamiltonian_field && initial_hamiltonian_field_size > 0) {
    *out << "#TransverseField:\n"
         << "#" << initial_hamiltonian_field[0];
    for (int i = 1; i < initial_hamiltonian_field_size; ++i) {
      *out << "," << initial_hamiltonian_field[i];
    }
    *out << "\n";
  }
  if (final_hamiltonian_field && final_hamiltonian_field_size > 0) {
    *out << "#LongitudinalField:\n"
         << "#" << final_hamiltonian_field[0];
    for (int i = 1; i < final_hamiltonian_field_size; ++i) {
      *out << "," << final_hamiltonian_field[i];
    }
    *out << "\n";
  }
  if (is_multiple_independent_baths) {
    *out << "#multiBath:\n";
    *out << "#";
    for (int i = 0; i < num_bath_couplings - 1; ++i) {
      *out << multiple_independent_bath_indices[i] << ",";
    }
    *out << multiple_independent_bath_indices[num_bath_couplings - 1] << "\n";
  }
  if (is_diagonal_bath_coupling) {
    *out << "#CorrelatedCouplingTerms:\n";
    for (int m = 0; m < num_bath_couplings; ++m) {
      *out << "#";
      for (int j = 0; j < hilbert_space_size - 1; ++j) {
        *out << bath_coupling_op[m * hilbert_space_size + j] << ",";
      }
      *out << bath_coupling_op[(m + 1) * hilbert_space_size - 1] << "\n";
    }
  }
  if (is_full_bath_coupling) {
    *out << "#FullBathTerms:\n";
    for (int m = 0; m < num_bath_couplings; ++m) {
      *out << "#";
      for (int j = 0; j < hilbert_space_num_elements - 1; ++j) {
        *out << bath_coupling_op[m * hilbert_space_num_elements + j] << ",";
      }
      *out << bath_coupling_op[(m + 1) * hilbert_space_num_elements - 1]
           << "\n";
    }
  }
  if (initial_annealing_hamiltonian) {
    *out << "#TransverseHamiltonian:\n";
    for (int i = 0; i < hilbert_space_size; ++i) {
      *out << "#";
      for (int j = 0; j < hilbert_space_size - 1; ++j) {
        *out << initial_annealing_hamiltonian[i + j * hilbert_space_size]
             << ",";
      }
      *out << initial_annealing_hamiltonian[(i + 1) * hilbert_space_size - 1]
           << "\n";
    }
  }
  if (final_annealing_hamiltonian) {
    *out << "#LongitudinalHamiltonian:\n";
    for (int i = 0; i < hilbert_space_size; ++i) {
      *out << "#";
      for (int j = 0; j < hilbert_space_size - 1; ++j) {
        *out << final_annealing_hamiltonian[i + j * hilbert_space_size] << ",";
      }
      *out << final_annealing_hamiltonian[(i + 1) * hilbert_space_size -1]
           << "\n";
    }
  }
  if (time_indepenent_hamiltonian) {
    *out << "#Hamiltonian:\n";
    for (int i = 0; i < hilbert_space_size; ++i) {
      *out << "#";
      for (int j = 0; j < hilbert_space_size - 1; ++j) {
        *out << time_indepenent_hamiltonian[i + j * hilbert_space_size] << ",";
      }
      *out << time_indepenent_hamiltonian[(i + 1) * hilbert_space_size - 1]
           << "\n";
    }
  }
  if (initial_density_matrix) {
    *out << "#InitialDensityMatrix:\n";
    for (int i = 0; i < hilbert_space_size; ++i) {
      *out << "#";
      for (int j = 0; j < hilbert_space_size - 1; ++j) {
        *out << initial_density_matrix[i + j * hilbert_space_size] << ",";
      }
      *out << initial_density_matrix[(i + 1) * hilbert_space_size - 1] << "\n";
    }
  }
  if (restart_input_filename != "") {
    *out << "#RestartInput=" << restart_input_filename << "\n";
  }
  if (restart_output_time > 0) {
    *out << "#RestartOutputTime=" << restart_output_time << "\n";
    *out << "#RestartFile=" << restart_output_filename << "\n";
  }
  phi_size_t num_elements = num_bath_couplings * num_matsubara_terms;
  *out << "#Estimated Number of Matrices="
       << GetHierarchyNodeCountEstimate(hierarchy_truncation_level, num_elements)
       << "\n";

  *out << "#-------------\n";
  if (run_method == RK4 || run_method == RKF45 ||
      run_method == RK4SPECTRUM || run_method == RKF45SPECTRUM) {
    *out << "#Threads=" << num_threads << "\n";
    *out << "#Integrator=";
    switch (run_method) {
      case RK4:
        *out << "rk4\n";
        break;
      case RKF45:
        *out << "rkf45\n";
        break;
      case RK4SPECTRUM:
        *out << "rk4spectrum\n";
        break;
      case RKF45SPECTRUM:
        *out << "rkf45spectrum\n";
        break;
      default:
        break;
    }
  } else if (run_method == BICGSTAB || run_method == BICGSTABU ||
           run_method == BICGSTABL) {
    *out << "#Threads=" << num_threads << "\n";
    if (run_method == BICGSTABL) {
      *out << "#Calculating steady-state using BiCGSTAB("
           << bicgstab_ell << ")\n";
    } else {
      *out << "#Calculating steady-state using BiCGSTAB\n";
    }
  }
}

phi_big_size_t PartialFactorial(phi_size_t start, phi_size_t end) {
  phi_big_size_t accu = 1;
  phi_size_t i;
  for (i = start; i <= end; i++) {
    accu *= i;
  }
  return accu;
}

Float FloatPartialFactorial(phi_size_t start, phi_size_t end) {
  Float accu = 1;
  phi_size_t i;
  for (i = start; i <= end; i++) {
    accu *= i;
  }
  return accu;
}

phi_big_size_t GetHierarchyNodeCountEstimate(phi_size_t max_level,
                                             phi_size_t matsubara_terms) {
  Float count = 0;
  for (phi_size_t l = 0; l < max_level; ++l) {
    count += FloatPartialFactorial(1, l + matsubara_terms - 1) /
        FloatPartialFactorial(1, l) /
        FloatPartialFactorial(1, matsubara_terms-1);
  }
  return (phi_big_size_t)count;
}

PhiParameters::~PhiParameters() {
  delete []time_indepenent_hamiltonian;
  delete []final_annealing_hamiltonian;
  delete []initial_annealing_hamiltonian;
  delete []initial_hamiltonian_field;
  delete []final_hamiltonian_field;
  delete []kappa;
  delete []gamma;
  delete []lambda;
  delete []bath_coupling_op;
  delete []multiple_independent_bath_indices;
  delete []initial_density_matrix;
  delete []spectrum_initial_density_matrix;
}

