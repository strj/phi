#include "hierarchy_integrator.h"
#include "numeric_types.h"
#include "phi_parameters.h"
#include "testing/base/public/gunit.h"

namespace {

// Set to 3 for all output.
// Set to -2 for no output.
const int kLogOutputVerbosity = -2;

class HierarchyIntegratorTest : public ::testing::Test {
 protected:
  PhiParameters *parameters_;

  HierarchyIntegratorTest() {}
  virtual ~HierarchyIntegratorTest() {}

  void SetUp() {
    parameters_ = GetNewParameters();
  }

  void TearDown() {
     delete parameters_;
  }

  PhiParameters* GetNewParameters() {
    PhiParameters* p = new PhiParameters();
    p->verbose = kLogOutputVerbosity;
    p->hbar = 1;
    p->boltzmann_constant = 1;
    p->temperature = 2.0;
    p->output_filename = "not_used";
    p->integration_time = 1.0;
    return p;
  }

  Complex* GetFinalDensityMatrix(RunMethod r) {
    parameters_->run_method = r;
    parameters_->FixIncompatibleCombinations();
    HierarchyIntegrator integrator(parameters_);
    integrator.Launch();
    return integrator.density_matrix();
  }

  void SetTwoStateHamiltonian() {
    int size = 2;
    parameters_->num_states = size;
    parameters_->time_indepenent_hamiltonian = new Complex[size * size];
    parameters_->time_indepenent_hamiltonian[0] = Complex(0.5);
    parameters_->time_indepenent_hamiltonian[1] = Complex(0.5);
    parameters_->time_indepenent_hamiltonian[2] = Complex(0.5);
    parameters_->time_indepenent_hamiltonian[3] = Complex(-0.5);
  }

  void SetTwoStateAnnealingHamiltonian() {
    int size = 2;
    parameters_->num_states = size;
    parameters_->initial_annealing_hamiltonian = new Complex[size * size];
    parameters_->initial_annealing_hamiltonian[0] = Complex(0.0);
    parameters_->initial_annealing_hamiltonian[1] = Complex(1.0);
    parameters_->initial_annealing_hamiltonian[2] = Complex(1.0);
    parameters_->initial_annealing_hamiltonian[3] = Complex(0.0);
    parameters_->final_annealing_hamiltonian = new Complex[size * size];
    parameters_->final_annealing_hamiltonian[0] = Complex(1.0);
    parameters_->final_annealing_hamiltonian[1] = Complex(0.0);
    parameters_->final_annealing_hamiltonian[2] = Complex(0.0);
    parameters_->final_annealing_hamiltonian[3] = Complex(-1.0);
  }

  void SetTrivialTruncatedTwoStateHamiltonian(const int full_size) {
    int truncated_size = 2;
    parameters_->hilbert_space_size = full_size;
    parameters_->hilbert_space_num_elements = full_size * full_size;
    parameters_->is_truncated_system = true;
    parameters_->num_states = truncated_size;
    Complex hamiltonian[] = {
        Complex(0.5), Complex(0.5),
        Complex(0.5), Complex(-0.5)
    };
    parameters_->time_indepenent_hamiltonian =
        GetEmbeddingInLargerSpace(truncated_size, hamiltonian, full_size);
    parameters_->time_indepenent_hamiltonian[full_size * full_size - 1] =
        Complex(9.9);
  }

  void SetTrivialTruncatedTwoStateAnnealingHamiltonian(const int full_size) {
    int truncated_size = 2;
    parameters_->hilbert_space_size = full_size;
    parameters_->hilbert_space_num_elements = full_size * full_size;
    parameters_->is_truncated_system = true;
    parameters_->num_states = truncated_size;
    Complex h_x[] = {
       Complex(0.0), Complex(1.0),
       Complex(1.0), Complex(0.0)
    };
    Complex h_z[] = {
        Complex(1.0), Complex(0.0),
        Complex(0.0), Complex(-1.0)
    };
    parameters_->initial_annealing_hamiltonian =
        GetEmbeddingInLargerSpace(truncated_size, h_x, full_size);
    parameters_->final_annealing_hamiltonian =
        GetEmbeddingInLargerSpace(truncated_size, h_z, full_size);
    parameters_->initial_annealing_hamiltonian[full_size * full_size - 1] =
        Complex(9.9);
    parameters_->final_annealing_hamiltonian[full_size * full_size - 1] =
        Complex(9.9);
  }

  void SetTruncatedAnnealingHamiltonian(const int full_size,
                                        const int truncated_size) {
    parameters_->hilbert_space_size = full_size;
    parameters_->hilbert_space_num_elements = full_size * full_size;
    parameters_->is_truncated_system = true;
    parameters_->num_states = truncated_size;
    parameters_->initial_annealing_hamiltonian =
        new Complex[full_size * full_size];
    parameters_->final_annealing_hamiltonian =
        new Complex[full_size * full_size];
    SetElementsToZero(full_size * full_size,
                      parameters_->initial_annealing_hamiltonian);
    SetElementsToZero(full_size * full_size,
                      parameters_->final_annealing_hamiltonian);
    for (int i = 0; i < full_size; ++i) {
      parameters_->final_annealing_hamiltonian[i * full_size + i] =
          Complex(1 + i);
      parameters_->initial_annealing_hamiltonian[i * full_size + i] =
          Complex(-1 - i);
      for (int j = 0; j < i; ++j) {
        parameters_->initial_annealing_hamiltonian[i * full_size + j] =
            Complex(1 + i);
        parameters_->initial_annealing_hamiltonian[j * full_size + i] =
            Complex(1 + i);
        parameters_->final_annealing_hamiltonian[i * full_size + j] =
            Complex(-1 - j);
        parameters_->final_annealing_hamiltonian[j * full_size + i] =
            Complex(-1 - j);
      }
    }
  }

  void SetWeakBathParameters(const int num_terms) {
    parameters_->num_bath_couplings = num_terms;
    parameters_->gamma = new Float[num_terms];
    parameters_->lambda = new Float[num_terms];
    for (int i = 0; i < num_terms; ++i) {
      parameters_->gamma[i] = 1.0 + i/10;
      parameters_->lambda[i] = 0.01 + i/1000;
    }
  }

  void SetMediumBathParameters(const int num_terms) {
    parameters_->num_bath_couplings = num_terms;
    parameters_->gamma = new Float[num_terms];
    parameters_->lambda = new Float[num_terms];
    for (int i = 0; i < num_terms; ++i) {
      parameters_->gamma[i] = 0.5 + i/10;
      parameters_->lambda[i] = 0.5 + i/100;
    }
  }

  void SetDiagonalIndependentBathCouplingTerms(const int size) {
    Float* bath_coupling_op = new Float[size * size];
    for (int i = 0; i < size * size; ++i) {
      bath_coupling_op[i] = 0;
    }
    for (int i = 0; i < size; ++i) {
      bath_coupling_op[i * size + i] = 1;
    }
    parameters_->bath_coupling_op = bath_coupling_op;
    parameters_->is_diagonal_bath_coupling = true;
  }

  void SetFullIndependentBathCouplingTerms(const int size) {
    Float* bath_coupling_op = new Float[size * size * size];
    for (int i = 0; i < size * size * size; ++i) {
      bath_coupling_op[i] = 0;
    }
    for (int i = 0; i < size; ++i) {
      bath_coupling_op[i * size * size + i * size + i] = 1;
    }
    parameters_->bath_coupling_op = bath_coupling_op;
    parameters_->is_full_bath_coupling = true;
  }

  void SetDiagonalBathCouplingTerms(const int num_terms, const int size) {
    Float* bath_coupling_op = new Float[num_terms * size];
    for (int i = 0; i < num_terms * size; ++i) {
      bath_coupling_op[i] = 0;
    }
    for (int i = 0; i < num_terms; ++i) {
      for (int j = 0; j < size; ++j) {
        bath_coupling_op[i * size + j] = pow(-1, i + j);
      }
    }
    parameters_->bath_coupling_op = bath_coupling_op;
    parameters_->is_diagonal_bath_coupling = true;
  }

  void SetFullDiagonalBathCouplingTerms(const int num_terms, const int size) {
    Float* bath_coupling_op = new Float[num_terms * size * size];
    for (int i = 0; i < num_terms * size * size; ++i) {
      bath_coupling_op[i] = 0;
    }
    for (int i = 0; i < num_terms; ++i) {
      for (int j = 0; j < size; ++j) {
        bath_coupling_op[i * size * size + j * size + j] = pow(-1, i + j);
      }
    }
    parameters_->bath_coupling_op = bath_coupling_op;
    parameters_->is_full_bath_coupling = true;
  }

  void SetFullBathCouplingTerms(const int num_terms, const int size) {
    Float* bath_coupling_op = new Float[num_terms * size * size];
    for (int i = 0; i < num_terms * size * size; ++i) {
     bath_coupling_op[i] = 0;
    }
    for (int i = 0; i < num_terms; ++i) {
      for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
          bath_coupling_op[i * size * size + j * size + k] = pow(-1, i + j + k);
        }
      }
    }
    parameters_->bath_coupling_op = bath_coupling_op;
    parameters_->is_full_bath_coupling = true;
  }

  void SetInitialState(const int size) {
    parameters_->initial_density_matrix = new Complex[size * size];
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        parameters_->initial_density_matrix[i * size + j] = Complex(0);
      }
    }
    parameters_->initial_density_matrix[0] = Complex(1);
  }

  Complex * GetEmbeddingInLargerSpace(
      const int small_size, const Complex * small, const int large_size) {
    Complex * large = new Complex[large_size * large_size];
    SetElementsToZero(large_size * large_size, large);
    for (int i = 0; i < small_size; ++i) {
      for (int j = 0; j < small_size; ++j) {
        large[i * large_size + j] = small[i * small_size + j];
      }
    }
    return large;
  }

  void assertValidDensityMatrix(const int n, const Complex *actual) {
    stringstream ss;
    ss << "Density Matrix:\n";
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - 1; ++j) {
        ss << actual[i * n + j] << ",";
      }
      ss << actual[(i + 1) * n - 1] << "\n";
    }
    Float trace = 0;
    for (int i = 0; i < n; ++i) {
      trace += std::real(actual[i * n + i]);
      ASSERT_NEAR(0, std::imag(actual[i * n + i]), 1e-9)
          << "Imaginary in diagonal.\n" << ss.str();
      ASSERT_GE(std::real(actual[i * n + i]), 0)
          << "Negative in diagonal.\n" << ss.str();
      ASSERT_LT(std::real(actual[i * n + i]), 1.00001)
          << "Overflow in diagonal.\n" << ss.str();
      for (int j = 0; j < i; ++j) {
          ASSERT_NEAR(std::real(actual[i * n + j]),
                      std::real(actual[j * n + i]), 1e-9)
              << "Adjoint not equal.\n" << ss.str();
          ASSERT_NEAR(-std::imag(actual[i * n + j]),
                      std::imag(actual[j * n + i]), 1e-9)
              << "Adjoint not equal.\n" << ss.str();
      }
    }
    ASSERT_GE(trace, 0) << "Trace must be >= 0.\n" << ss.str();
    ASSERT_LT(trace, 1.00001) << "Trace must be <= 1.\n" << ss.str();
  }

  void assertMatrixNear(const int n,
                        const Complex* expected_matrix,
                        const Complex* actual_matrix,
                        const Float delta) {
    stringstream expected_s;
    stringstream actual_s;
    expected_s << "Expected:\n";
    actual_s << "Actual:\n";
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - 1; ++j) {
        expected_s << expected_matrix[i * n + j] << ",";
        actual_s << actual_matrix[i * n + j] << ",";
      }
      expected_s << expected_matrix[(i + 1) * n - 1] << "\n";
      actual_s << actual_matrix[(i + 1) * n - 1] << "\n";
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        Complex expected = expected_matrix[i*n +j];
        Complex actual = actual_matrix[i*n +j];
        ASSERT_NEAR(std::real(expected), std::real(actual), delta)
            << "Mismatch in real part of element at "
            << "[" << i << "," << j << "]:\n"
            << expected_s.str() << "\n" << actual_s.str();
        ASSERT_NEAR(std::imag(expected), std::imag(actual), delta)
            << "Mismatch in imaginary part of element at "
            << "[" << i << "," << j << "]:\n"
            << expected_s.str() << "\n" << actual_s.str();
      }
    }
  }

  void assertMatrixAlmostEqual(const int n,
                               const Complex* expected,
                               const Complex* actual) {
    assertMatrixNear(n, expected, actual, 1e-6);
  }

  void assertMatrixSomewhatEqual(const int n,
                                 const Complex* expected,
                                 const Complex* actual) {
    assertMatrixNear(n, expected, actual, 1e-3);
  }

  void RunRk4Test(const int n, const Complex* expected) {
    Complex* actual = GetFinalDensityMatrix(RunMethod::RK4);
    assertValidDensityMatrix(n, actual);
    assertMatrixAlmostEqual(n, expected, actual);
    delete []actual;
  }

  void RunRkf45Test(const int n, const Complex* expected) {
    Complex * actual = GetFinalDensityMatrix(RunMethod::RKF45);
    assertValidDensityMatrix(n, actual);
    assertMatrixSomewhatEqual(n, expected, actual);
    delete []actual;
  }

  void RunIntegrationTest(const int n, const Complex *expected) {
    RunRk4Test(n, expected);
    RunRkf45Test(n, expected);
  }
};

// Tests below here.

const Complex kVacuumSoln[] = {
    Complex(0.788986), Complex(0.211014, -0.349228),
    Complex(0.211014, 0.349228), Complex(0.211014)
};

TEST_F(HierarchyIntegratorTest, TestVacuum) {
  SetTwoStateHamiltonian();
  // At least one bath parameter has to be set for a vacuum calculation due to
  // a bug in hierarchy_integrator.cc.
  SetWeakBathParameters(1);
  SetInitialState(2);
  RunIntegrationTest(2, kVacuumSoln);
}

TEST_F(HierarchyIntegratorTest, TestTruncatedVacuum) {
  SetTrivialTruncatedTwoStateHamiltonian(3);
  // At least one bath parameter has to be set for a vacuum calculation due to
  // a bug in hierarchy_integrator.cc.
  SetWeakBathParameters(1);
  SetInitialState(3);
  Complex * solution = GetEmbeddingInLargerSpace(2, kVacuumSoln, 3);
  RunIntegrationTest(3, solution);
  delete []solution;
}

const Complex kWeakIndependentCouplingSoln[] = {
    Complex(0.790012, 0), Complex(0.206755, -0.346119),
    Complex(0.206755, 0.346119), Complex(0.209988, 0)
};

TEST_F(HierarchyIntegratorTest, TestWeakIndependentBath) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 2;
  RunIntegrationTest(2, kWeakIndependentCouplingSoln);
}

TEST_F(HierarchyIntegratorTest, TestWeakIndependentDiagonalBath) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetDiagonalIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 2;
  RunIntegrationTest(2, kWeakIndependentCouplingSoln);
}

TEST_F(HierarchyIntegratorTest, WeakIndependentFullBath) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetFullIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 2;
  RunIntegrationTest(2, kWeakIndependentCouplingSoln);
}

TEST_F(HierarchyIntegratorTest, TestTruncatedWeakIndependentFullBath) {
  SetTrivialTruncatedTwoStateHamiltonian(3);
  SetWeakBathParameters(3);
  SetFullIndependentBathCouplingTerms(3);
  parameters_->hierarchy_truncation_level = 2;
  SetInitialState(3);
  Complex * solution = GetEmbeddingInLargerSpace(
      2, kWeakIndependentCouplingSoln, 3);
  RunRkf45Test(3, solution);
  delete []solution;
}

const Complex kWeakDiagonalCouplingSoln[] = {
    Complex(0.793056, 0), Complex(0.194223, -0.336944),
    Complex(0.194223, 0.336944), Complex(0.206944, 0)
};

TEST_F(HierarchyIntegratorTest, TestWeakDiagonalBath) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetDiagonalBathCouplingTerms(2, 2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 2;
  RunIntegrationTest(2, kWeakDiagonalCouplingSoln);
}

TEST_F(HierarchyIntegratorTest, TestWeakFullDiagonalBath) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetFullDiagonalBathCouplingTerms(2, 2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 2;
  RunIntegrationTest(2, kWeakDiagonalCouplingSoln);
}

const Complex kWeakFullCouplingSoln[] = {
    Complex(0.744586, 0), Complex(0.200681, -0.309559),
    Complex(0.200681, 0.309559), Complex(0.255414, 0)
};

TEST_F(HierarchyIntegratorTest, TestWeakFullBath) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetFullBathCouplingTerms(2, 2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 2;
  RunIntegrationTest(2, kWeakFullCouplingSoln);
}

const Complex kWeakIndependentCouplingMatsubaraSoln[] = {
    Complex(0.789331, 0), Complex(0.208929, -0.348365),
    Complex(0.208929, 0.348365), Complex(0.210669, 0)
};

TEST_F(HierarchyIntegratorTest, TestWeakIndependentBathMatsubara) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetInitialState(2);
  parameters_->temperature = 0.5;
  parameters_->hierarchy_truncation_level = 3;
  parameters_->num_matsubara_terms = 3;
  RunIntegrationTest(2, kWeakIndependentCouplingMatsubaraSoln);
}

TEST_F(HierarchyIntegratorTest, TestTruncatedWeakIndependentFullBathMatsubara) {
  SetTrivialTruncatedTwoStateHamiltonian(3);
  SetWeakBathParameters(3);
  SetFullIndependentBathCouplingTerms(3);
  SetInitialState(3);
  parameters_->temperature = 0.5;
  parameters_->hierarchy_truncation_level = 3;
  parameters_->num_matsubara_terms = 3;
  Complex * solution = GetEmbeddingInLargerSpace(
      2, kWeakIndependentCouplingMatsubaraSoln, 3);
  RunRkf45Test(3, solution);
  delete []solution;
}

TEST_F(HierarchyIntegratorTest, TestWeakDiagonalBathMatsubara) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetDiagonalIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->temperature = 0.5;
  parameters_->hierarchy_truncation_level = 3;
  parameters_->num_matsubara_terms = 3;
  RunIntegrationTest(2, kWeakIndependentCouplingMatsubaraSoln);
}

TEST_F(HierarchyIntegratorTest, TestWeakFullBathMatsubara) {
  SetTwoStateHamiltonian();
  SetWeakBathParameters(2);
  SetFullIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->temperature = 0.5;
  parameters_->hierarchy_truncation_level = 3;
  parameters_->num_matsubara_terms = 3;
  RunIntegrationTest(2, kWeakIndependentCouplingMatsubaraSoln);
}

const Complex kMediumIndependentCouplingSoln[] = {
    Complex(0.83145, 0), Complex(0.0804222, -0.230756),
    Complex(0.0804222, 0.230756), Complex(0.16855, 0)
};

TEST_F(HierarchyIntegratorTest, TestMediumBathTimeLocal) {
  SetTwoStateHamiltonian();
  SetMediumBathParameters(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 4;
  parameters_->use_time_local_truncation = true;
  RunIntegrationTest(2, kMediumIndependentCouplingSoln);
}

TEST_F(HierarchyIntegratorTest, TestMediumDiagonalBathTimeLocal) {
  SetTwoStateHamiltonian();
  SetMediumBathParameters(2);
  SetDiagonalIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 4;
  parameters_->use_time_local_truncation = true;
  RunIntegrationTest(2, kMediumIndependentCouplingSoln);
}

TEST_F(HierarchyIntegratorTest, TestMediumFullBathTimeLocal) {
  SetTwoStateHamiltonian();
  SetMediumBathParameters(2);
  SetFullIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 4;
  parameters_->use_time_local_truncation = true;
  RunIntegrationTest(2, kMediumIndependentCouplingSoln);
}

TEST_F(HierarchyIntegratorTest,
       TestTruncatedMediumIndependentFullBathTimeLocal) {
  SetTrivialTruncatedTwoStateHamiltonian(3);
  SetMediumBathParameters(3);
  SetFullIndependentBathCouplingTerms(3);
  SetInitialState(3);
  parameters_->hierarchy_truncation_level = 4;
  parameters_->use_time_local_truncation = true;
  Complex * solution = GetEmbeddingInLargerSpace(
      2, kMediumIndependentCouplingSoln, 3);
  RunRkf45Test(3, solution);
  delete []solution;
}

const Complex kVacuumAnnealingSoln[] = {
    Complex(0.777864, 0), Complex(0.315941, -0.270135),
    Complex(0.315941, 0.270135), Complex(0.222136, 0)
};

TEST_F(HierarchyIntegratorTest, TestAnnealVacuum) {
  SetTwoStateAnnealingHamiltonian();
  // At least one bath parameter has to be set for a vacuum calculation due to
  // a bug in hierarchy_integrator.cc.
  SetMediumBathParameters(1);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 1;
  RunIntegrationTest(2, kVacuumAnnealingSoln);
}

TEST_F(HierarchyIntegratorTest, TestTruncatedAnnealVacuum) {
  SetTrivialTruncatedTwoStateAnnealingHamiltonian(3);
  // At least one bath parameter has to be set for a vacuum calculation due to
  // a bug in hierarchy_integrator.cc.
  SetMediumBathParameters(1);
  SetInitialState(3);
  parameters_->hierarchy_truncation_level = 1;
  Complex * solution = GetEmbeddingInLargerSpace(
      2, kVacuumAnnealingSoln, 3);
  RunIntegrationTest(3, solution);
  delete []solution;
}

const Complex kMediumIndependentCouplingAnnealingSoln[] = {
  Complex(0.811767, 0), Complex(0.128111, -0.144969),
  Complex(0.128111, 0.144969), Complex(0.188233, 0)
};

TEST_F(HierarchyIntegratorTest, TestAnnealMediumIndependentBath) {
  SetTwoStateAnnealingHamiltonian();
  SetMediumBathParameters(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 3;
  RunIntegrationTest(2, kMediumIndependentCouplingAnnealingSoln);
}

TEST_F(HierarchyIntegratorTest, TestAnnealMediumDiagonalIndependentBath) {
  SetTwoStateAnnealingHamiltonian();
  SetMediumBathParameters(2);
  SetDiagonalIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 3;
  RunIntegrationTest(2, kMediumIndependentCouplingAnnealingSoln);
}

TEST_F(HierarchyIntegratorTest, TestAnnealMediumFullIndependentBath) {
  SetTwoStateAnnealingHamiltonian();
  SetMediumBathParameters(2);
  SetFullIndependentBathCouplingTerms(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 3;
  RunIntegrationTest(2, kMediumIndependentCouplingAnnealingSoln);
}

TEST_F(HierarchyIntegratorTest, TestTruncatedAnnealMediumFullIndependentBath) {
  SetTrivialTruncatedTwoStateAnnealingHamiltonian(3);
  SetMediumBathParameters(3);
  SetFullIndependentBathCouplingTerms(3);
  SetInitialState(3);
  parameters_->hierarchy_truncation_level = 3;
  Complex * solution = GetEmbeddingInLargerSpace(
      2, kMediumIndependentCouplingAnnealingSoln, 3);
  RunIntegrationTest(3, solution);
  delete []solution;
}

const Complex kWeakIndependentCouplingTimeLocalAnnealingSoln[] = {
    Complex(0.467201, -1.0053e-16), Complex(0.255327, 0.207856),
    Complex(0.255327, -0.207856), Complex(0.532799, 1.01096e-16)
};

TEST_F(HierarchyIntegratorTest, TestAnnealWeakIndependentBathTimeLocal) {
  SetTwoStateAnnealingHamiltonian();
  SetWeakBathParameters(2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 3;
  parameters_->integration_time = 10.;
  parameters_->use_time_local_truncation = true;
  // With the longer integration time needed in annealing calculations, only the
  // RKF45 integration is run since it is much faster.
  RunRkf45Test(2, kWeakIndependentCouplingTimeLocalAnnealingSoln);
}

TEST_F(HierarchyIntegratorTest,
       TestTruncatedAnnealWeakFullIndependentBathTimeLocal) {
  SetTrivialTruncatedTwoStateAnnealingHamiltonian(3);
  SetWeakBathParameters(3);
  SetFullIndependentBathCouplingTerms(3);
  SetInitialState(3);
  parameters_->hierarchy_truncation_level = 3;
  parameters_->integration_time = 10.;
  parameters_->use_time_local_truncation = true;
  Complex * solution = GetEmbeddingInLargerSpace(
      2, kWeakIndependentCouplingTimeLocalAnnealingSoln, 3);
  RunRkf45Test(3, solution);
  delete []solution;
}

const Complex kWeakDiagonalCouplingTimeLocalAnnealingSoln[] = {
    Complex(0.423523, 2.67555e-17), Complex(0.0658185, 0.0703083),
    Complex(0.0658185, -0.0703083), Complex(0.576477, -2.67185e-17)
};

TEST_F(HierarchyIntegratorTest, TestAnnealWeakDiagonalBathTimeLocal) {
  SetTwoStateAnnealingHamiltonian();
  SetWeakBathParameters(2);
  SetDiagonalBathCouplingTerms(2, 2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 3;
  parameters_->integration_time = 10.;
  parameters_->use_time_local_truncation = true;
  // With the longer integration time needed in annealing calculations, only the
  // RKF45 integration is run since it is much faster.
  RunRkf45Test(2, kWeakDiagonalCouplingTimeLocalAnnealingSoln);
}

const Complex kWeakFullCouplingTimeLocalAnnealingSoln[] = {
    Complex(0.428181, -6.10088e-18), Complex(0.0237366, 0.0886596),
    Complex(0.0237366, -0.0886596), Complex(0.571819, 4.73958e-18)
};

TEST_F(HierarchyIntegratorTest, TestAnnealWeakFullBathTimeLocal) {
  SetTwoStateAnnealingHamiltonian();
  SetWeakBathParameters(2);
  SetFullBathCouplingTerms(2, 2);
  SetInitialState(2);
  parameters_->hierarchy_truncation_level = 3;
  parameters_->integration_time = 10.;
  parameters_->use_time_local_truncation = true;
  // With the longer integration time needed in annealing calculations, only the
  // RKF45 integration is run since it is much faster.
  RunRkf45Test(2, kWeakFullCouplingTimeLocalAnnealingSoln);
}

const Complex kTruncatedWeakFullCouplingTimeLocalAnnealingSoln[] = {
  Complex(0.267895), Complex(-0.0281567, -0.0246088),
  Complex(-0.00247109, -0.0184745),
  Complex(-0.0281567, 0.0246088), Complex(0.208783),
  Complex(0.154777, 0.00171474),
  Complex(-0.00247109, 0.0184745), Complex(0.154777, -0.00171474),
  Complex(0.116023)
};

TEST_F(HierarchyIntegratorTest, TestTruncatedAnnealWeakFullBathTimeLocal) {
  SetTruncatedAnnealingHamiltonian(3, 2);
  SetWeakBathParameters(2);
  SetFullBathCouplingTerms(2, 3);
  SetInitialState(3);
  parameters_->hierarchy_truncation_level = 3;
  parameters_->integration_time = 10.;
  parameters_->use_time_local_truncation = true;
  // With the longer integration time needed in annealing calculations, only the
  // RKF45 integration is run since it is much faster.
  RunRkf45Test(3, kTruncatedWeakFullCouplingTimeLocalAnnealingSoln);
}

// TODO(johanstr): Add tests for: Shi et al. filtering method, steady-state
// calculations, using different annealing schedules, setting multiple
// independent baths, and restarting calculations.

}  // namespace
