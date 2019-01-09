#include "restart_helper.h"

RestartHelper::RestartHelper(const std::string input_name,
                             const std::string output_name,
                             const std::string backup_name) {
  input_file_name_ = input_name;
  output_file_name_ = output_name;
  backup_file_name_ = backup_name;
}

RestartHelper::~RestartHelper() {};

void RestartHelper::BackupRestartFile() {
  if (backup_file_name_ == "") {
    return;
  }
  // TODO(johantsr): Basic file copy.
}

void RestartHelper::WriteMatrix(const Complex * matrix,
                                const phi_size_t size,
                                ofstream *out) {
  *out << matrix[0];
  for (int i = 1; i < size; ++i) {
    *out << " " << matrix[i];
  }
  *out << "\n";
}

void RestartHelper::WriteRestartFile(const Float time,
                                     const Float timestep,
                                     const HierarchyNode *nodes,
                                     const int num_nodes,
                                     Complex *** time_local_matrices) {
  ofstream out(output_file_name_.c_str());
  out << time << "\n";
  out << timestep << "\n";
  const phi_size_t num_elements = nodes[0].num_elements;
  for (int n = 0; n < num_nodes; ++n) {
    WriteMatrix(nodes[n].density_matrix, num_elements, &out);
  }
  if (time_local_matrices != NULL) {
    const phi_size_t num_bath_couplings = nodes[0].num_bath_couplings;
    const phi_size_t num_temperature_correction_terms =
        nodes[0].num_matsubara_terms;
    for (int m = 0; m < num_bath_couplings; ++m) {
      for (int k = 0; k < num_temperature_correction_terms; ++k) {
        WriteMatrix(time_local_matrices[m][k], num_elements, &out);
      }
    }
  }

  out.close();
  BackupRestartFile();
}

void RestartHelper::ReadRestartFile(Float *time,
                                    Float *timestep,
                                    HierarchyNode *nodes,
                                    const phi_size_t num_nodes,
                                    Complex *** time_local_matrices) {
  ifstream in(input_file_name_.c_str());
  in >> *time;
  in >> *timestep;
  const phi_size_t num_elements = nodes[0].num_elements;
  for (int n = 0; n < num_nodes; ++n) {
    for (int i = 0; i < num_elements; ++i) {
      in >> nodes[n].density_matrix[i];
    }
  }
  if (time_local_matrices != NULL) {
    const phi_size_t num_bath_coupling_terms = nodes[0].num_bath_couplings;
    const phi_size_t num_temperature_correction_terms =
        nodes[0].num_matsubara_terms;
    for (int m = 0; m < num_bath_coupling_terms; ++m) {
      for (int k = 0; k < num_temperature_correction_terms; ++k) {
        for (int n = 0; n < num_elements; ++n) {
          in >> time_local_matrices[m][k][n];
        }
      }
    }
  }
  in.close();
}

void RestartHelper::ReadEmissionSpectrumInput(HierarchyNode *nodes,
                                              const int num_nodes,
                                              const Complex * spectrum) {
  ifstream in(input_file_name_.c_str());
  Float not_used;
  in >> not_used;  // The integration time.
  in >> not_used;  // The intrgration timestep.
  for (int n = 0; n < num_nodes; ++n) {
    for (int i = 0; i < nodes[n].num_states; ++i) {
      nodes[n].density_matrix[i] = 0;
    }
  }
  Complex value;
  for (int n = 0; n < num_nodes; ++n) {
    for (int i = 0; i < nodes[n].num_states; ++i) {
      for (int j = 0; j < nodes[n].num_states; ++j) {
        in >> value;
        nodes[n].density_matrix[j] += value * spectrum[j];
      }
    }
  }
  in.close();
}
