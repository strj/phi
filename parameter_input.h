#ifndef PHI_PARAMETER_INPUT_H_
#define PHI_PARAMETER_INPUT_H_
#include <string>
#include <iostream>
#include "complex_matrix.h"

class ParameterInput {
 public:
  ParameterInput();
  ~ParameterInput();
  enum ParameterRange { POSITIVE, NON_NEGATIVE, ZERO_OR_ONE, NONE };

  bool Get(const std::string& line,
           const std::string& input_name,
           phi_size_t* int_value,
           ParameterRange range);
  bool Get(const std::string& line,
           const std::string& input_name,
           Float* float_value,
           ParameterRange range);
  bool Get(const std::string& line,
           const std::string& input_name,
           bool* boolean_value);
  bool Get(const std::string& line,
           const std::string& input_name,
           Complex* complex_value);

  // Reads a line of values from a string.
  bool Get(const std::string& line,
           int* int_array,
           const int array_size,
           const int j_skip);
  bool Get(const std::string& line,
           Float* float_array,
           const int array_size,
           const int j_skip);
  bool Get(const std::string& line,
           Complex* complex_array,
           const int array_size,
           const int j_skip);

  int GetCharacterOccurence(const char c, const std::string& line);

 private:
  static const std::string kNumbers;
  static const std::string kArray;
  static const std::string kComplexArray;

  // Reads a value from a string
  template<class T> bool ReadValue(const std::string& line,
                                   const std::string& input_name,
                                   T* var,
                                   ParameterInput::ParameterRange range);

  // vector
  bool IsValidComplexArrayInput(const std::string& s);
  bool IsValidArrayInput(const std::string& s);

  // scalar
  bool IsValidIntegerInput(const std::string& s);
  bool IsValidFloatInput(const std::string& s);
  bool IsValidBoolInput(const std::string& s);
  bool IsValidComplexInput(const std::string& s);
};
#endif  // PHI_PARAMETER_INPUT_H_
