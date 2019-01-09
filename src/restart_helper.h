#ifndef PHI_RESTART_HELPER_H_
#define PHI_RESTART_HELPER_H_

#include <fstream>
#include <string>

#include "complex_matrix.h"
#include "hierarchy_node.h"

class RestartHelper {
 protected:
  std::string input_file_name_;
  std::string output_file_name_;
  std::string backup_file_name_;
  virtual void BackupRestartFile();
  void WriteMatrix(const Complex *, const phi_size_t, ofstream *);
 public:
  RestartHelper(const std::string,
                const std::string,
                const std::string);
  virtual ~RestartHelper();
  void WriteRestartFile(const Float,
                        const Float,
                        const HierarchyNode *,
                        const phi_size_t,
                        Complex ***);

  void ReadRestartFile(Float *,
                       Float *,
                       HierarchyNode *,
                       const phi_size_t,
                       Complex ***);

  void ReadEmissionSpectrumInput(HierarchyNode *,
                                 const int,
                                 const Complex *);
};
#endif  // PHI_RESTART_HELPER_H_
