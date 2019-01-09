#include "parameter_input.h"

const std::string ParameterInput::kNumbers = "1234567890";
const std::string ParameterInput::kArray = ".Ee-+,";
const std::string ParameterInput::kComplexArray = ".Ee-+,()";

ParameterInput::ParameterInput() {};
ParameterInput::~ParameterInput() {};

template <class T>
bool ParameterInput::ReadValue(const std::string& line,
                               const std::string& input_name,
                               T* var,
                               ParameterRange range) {
  std::stringstream ss;
  ss.str(line);
  ss >> *var;
  switch (range) {
    case (POSITIVE):
      if (*var <= 0) {
        std::cerr << "get() Error: " << input_name << " must be "
                  << "positive.\n";
        return false;
      }
      break;
    case (NON_NEGATIVE):
      if (*var < 0) {
        std::cerr << "get() Error: " << input_name << " must be "
                  << "non-negative.\n";
        return false;
      }
      break;
    case (ZERO_OR_ONE):
      if (*var != 0 && *var != 1) {
        std::cerr << "get() Error: " << input_name << " must be "
                  << "0 (for 'no' or 'false') or 1 (for 'yes' or 'true').\n";
        return false;
      }
      break;
    default:
      break;
  }
  return true;
}


bool ParameterInput::Get(const std::string& line,
                         const std::string& input_name,
                         phi_size_t* int_value,
                         ParameterRange range) {
  if (!IsValidIntegerInput(line)) {
    std::cerr << "get() Error reading " << input_name << "\n";
    return false;
  }
  if (!ReadValue<phi_size_t>(line, input_name, int_value, range)) {
    return false;
  }
  return true;
}

bool ParameterInput::Get(const std::string& line,
                         const std::string& input_name,
                         Float* float_value,
                         ParameterRange range) {
  if (!IsValidFloatInput(line)) {
    std::cerr << "get() Error reading " << input_name << "\n";
    return false;
  }
  if (!ReadValue<Float>(line, input_name, float_value, range)) {
    return false;
  }
  return true;
}

bool ParameterInput::Get(const std::string& line,
                         const std::string& input_name,
                         bool* boolean_value) {
  if(!IsValidBoolInput(line)) {
    std::cerr << "get() Error reading " << input_name << ".\n"
         << "Boolean parameters must be 0 (for false) or 1 (for true).\n";
    return false;
  }
  int boolean_as_integer;
  if (!ReadValue<int>(line, input_name, &boolean_as_integer, ZERO_OR_ONE)) {
    return false;
  }
  *boolean_value = boolean_as_integer == 1;
  return true;
}

bool ParameterInput::Get(const std::string& line,
                         const std::string& input_name,
                         Complex* complex_value) {
  if (IsValidComplexInput(line)) {
    std::cerr << "get() Error reading " << input_name << ".\n";
    return false;
  }

  int index_left_bracket = line.find("(", 0);
  int index_comma = line.find(",", index_left_bracket + 1);
  int index_right_bracket = line.find(")", index_comma + 1);

  Float real_value_buffer = 0.;
  Float imaginary_value_buffer = 0.;

  // Always returns true.
  ReadValue<Float>(
      line.substr(index_left_bracket + 1, index_comma - index_left_bracket - 1),
      input_name,
      &real_value_buffer,
      NONE);

  // Always returns true.
  ReadValue<Float>(
      line.substr(index_comma + 1, index_right_bracket - index_comma - 1),
      input_name,
      &imaginary_value_buffer,
      NONE);

  *complex_value = Complex(real_value_buffer, imaginary_value_buffer);
  return true;
}


bool ParameterInput::Get(const std::string& line,
                         int * int_array,
                         const int array_size,
                         const int j_skip) {
  if (!IsValidArrayInput(line))
    return false;

  int i = 0;
  int j = 0;
  int n = 0;
  if ((line.compare(line.size()-1,1,",") == 0)) {
      std::cerr << "get() Error: should end with ',' in:"
           << "\n\t" << line << "\n";
      return false;
  }

  int comma_count = GetCharacterOccurence(',', line);
  if (comma_count != array_size-1) {
    std::cerr << "get() Error: incorrect number of ','s in:"
         << "\n\t" << line << "\n";
    return false;
  }

  int iprev = 0;
  std::stringstream input_stream;
  int buf;
  j = 0;
  i = line.find(",",0);
  while (i != std::string::npos) {
      if (i <= iprev) {
        std::cerr << "get() Error: no value to unpack at element " << n
             << " in:\n\t" << line << "\n";
        return false;
      }
      input_stream.str(line.substr(iprev,i-iprev));
      input_stream >> buf;
      int_array[j] = buf;
      input_stream.clear();
      iprev = i+1;
      i = line.find(",",iprev);
      n += 1;
      j += j_skip;
  }
  input_stream.str(line.substr(iprev,line.length()-iprev));
  input_stream >> buf;
  int_array[j] = buf;
  input_stream.clear();
  n += 1;

  if (n != array_size) {
    std::cerr << "get() Error: Expected to read " << array_size
         << " values but got " << n << " instead in:"
         << "\n\t" << line << "\n";
    return false;
  }

  return true;
};


bool ParameterInput::Get(const std::string& line,
                         Float* float_array,
                         const int array_size,
                         const int j_skip) {
  if (!IsValidArrayInput(line)) {
      return false;
  }

  if ((line.compare(line.size()-1, 1, ",") == 0)) {
      std::cerr << "get() Error: should end with ',' in:"
           << "\n\t" << line << "\n";
      return false;
  }

  int comma_count = GetCharacterOccurence(',', line);
  if (comma_count != array_size-1) {
    std::cerr << "get() Error: incorrect number of ','s in:"
         << "\n\t" << line << "\n";
    return false;
  }


  Float buf;
  std::stringstream input_stream;
  
  int iprev = 0;
  int i = line.find(",", 0);
  int j = 0;
  int n = 0;
  while (i != std::string::npos) {
    if (i <= iprev) {
      std::cerr << "get() Error: no value to unpack at element " << n
           << " in:\n\t" << line << "\n";
      return false;
    }
    input_stream.str(line.substr(iprev, i-iprev));
    input_stream >> buf;
    float_array[j] = buf;
    input_stream.clear();
    iprev = i + 1;
    i = line.find(",", iprev);
    ++n;
    j += j_skip;
  }
  input_stream.str(line.substr(iprev, line.length() - iprev));
  input_stream >> buf;
  float_array[j] = buf;
  input_stream.clear();
  ++n;
  if (n != array_size) {
    std::cerr << "get() Error: Expected to read " << array_size
         << " values but got " << n << " instead in line:"
         << "\n\t" << line << "\n";
    return false;
  }
  return true;
};


bool ParameterInput::Get(const std::string& line,
                         Complex * complex_array,
                         const int array_size,
                         const int j_skip) {
  if ((line.compare(line.size()-1, 1, ",") == 0)) {
      std::cerr << "get() Error: should end with ',' in:"
           << "\n\t" << line << "\n";
      return false;
  }

  int comma_count = GetCharacterOccurence(',', line);
  if (comma_count != 2 * array_size - 1 && comma_count != array_size - 1) {
    std::cerr << "get() Error: incorrect number of ','s in:"
         << "\n\t" << line << "\n";
    return false;
  }

  int left_bracket_count = GetCharacterOccurence('(', line);
  if (left_bracket_count != 0 && left_bracket_count != array_size) {
    std::cerr << "get() Error: incorrect number of '('s (" << left_bracket_count
         << ") in:\n\t" << line << "\n";
    return false;
  }

  int right_bracket_count = GetCharacterOccurence('(', line);
  if ((right_bracket_count != 0 && right_bracket_count != array_size) ||
      (right_bracket_count != left_bracket_count)) {
    std::cerr << "get() Error: incorrect number of ')'s in:"
         << "\n\t" << line << "\n";
    return false;
  }

  if (right_bracket_count == 0 && comma_count != array_size -1) {
    std::cerr << "get() Error: incorrect number of ','s for "
         << "apparently non-complex value input in:\n\t" << line << "\n"
         << "(real and complex input should not be mixed the input)\n";
    return false;
  }

  bool is_real_value_input = left_bracket_count == 0;

  int i = 0;
  int j = 0;
  int n = 0;

  if (is_real_value_input) {
    Float * float_array;
    float_array = new Float[array_size];
    if(!Get(line, float_array, array_size, 1)) {
      return false;
    }
    for (j = 0; j < array_size; ++j) {
      complex_array[j * j_skip] = Complex(float_array[j]);
    }
    return true;
  }
  if (!IsValidComplexArrayInput(line)) {
    return false;
  }

  std::stringstream input_stream;
  Float buf_re;
  Float buf_im;
  j = 0;
  int index_left_bracket = line.find("(", 0);
  i = line.find(",", index_left_bracket);
  int index_right_bracket = line.find(")", i);
  while (index_left_bracket != std::string::npos && i != std::string::npos) {
    if (i - index_left_bracket <= 1 && index_right_bracket - i <= 1) {
      std::cerr << "get() Error: no value to unpack at element "
           << n << " in:\n\t" << line << "\n";
      return false;
    }
    input_stream.str(
        line.substr(index_left_bracket + 1, i - index_left_bracket - 1));
    input_stream >> buf_re;
    input_stream.clear();
    input_stream.str(line.substr(i + 1, index_right_bracket - i - 1));
    input_stream >> buf_im;
    input_stream.clear();
    complex_array[j] = Complex(buf_re,buf_im);
    index_left_bracket = line.find("(", index_right_bracket);
    i   = line.find(",", index_left_bracket + 1);
    index_right_bracket = line.find(")", i);
    n += 1;
    j += j_skip;
  }

  if (n != array_size) {
    std::cerr << "get() Error: Expected to read " << array_size
         << " values but got " << n << " instead in:"
         << "\n\t" << line << "\n";
    return false;
  }
  return true;
}

bool ParameterInput::IsValidBoolInput(const std::string& s) {
    bool is_valid = true;
    int i = 0;
    // Skip initial whitespace.
    while (s.at(i) == ' ') {
      ++i;
    }
    if (s.at(i) != '0' && s.at(i) != '1') {
      is_valid = false;
    }
    ++i;
    while (is_valid && i < s.size()) {
      if (s.at(i) != ' ') {
        is_valid = false;
      }
      ++i;
    }
    if (!is_valid) {
      std::stringstream stream;
      for (int j = 0; j < i - 1; ++j) {
        stream << " ";
      }
      stream << "^";
      std::cerr << "isValidBoolInput() Error at position " << i - 1 << " in:"
           << "\n\t" << s << "\n" << stream.str() << "\n";
    }
    return is_valid;
}

bool ParameterInput::IsValidIntegerInput(const std::string& s) {
    bool is_valid = true;
    int i = 0;
    bool is_whitespace = false;
    // Skip initial whitespace.
    while (s.at(i) == ' ') {
      ++i;
    }
    while (is_valid && i < s.size()) {
      if (kNumbers.find(s.at(i)) == std::string::npos ||
          (kNumbers.find(s.at(i)) != std::string::npos && is_whitespace)) {
        is_valid = false;
      }
      ++i;
      while (i < s.size() && s.at(i) == ' ') {
         ++i;
         is_whitespace = true;
      }
    }
    if (!is_valid) {
      std::stringstream stream;
      for (int j = 0; j < i - 1; ++j) {
        stream << " ";
      }
      stream << "^";
      std::cerr << "isValidIntegerInput() Error at position " << i - 1 << " in:\n"
           << "\t" << s << "\n" << stream.str() << "\n";
    }
    return is_valid;
}

bool ParameterInput::IsValidFloatInput(const std::string& s) {
    bool is_valid = true;
    bool is_period = false;
    bool is_e = false;
    bool is_whitespace = false;
    int numi;
    int modi;
    int m = 0;
    int i = 0;
    //skip initial is_whitespace
    while (s.at(i) == ' ') {
      ++i;
    }
    while (is_valid && i < s.size()) {
      numi = kNumbers.find(s.at(i));
      modi = kArray.find(s.at(i));
      if ((numi == std::string::npos && modi == std::string::npos) ||
          (numi != std::string::npos && is_whitespace)) {
         is_valid = false;
      } else if (modi != std::string::npos) {
         switch (modi) {
           case 0: // '.'
             if (is_period || is_e || is_whitespace)
               is_valid = false;
             is_period = true;
             break;
           case 1: // 'E'
           case 2: // 'e'
             if (m == 0 || is_e || is_whitespace)
               is_valid = false;
             is_e = true;
             break;
           case 3: // '-'
           case 4: // '+'
             if ((m != 0 && s.at(i - 1) != 'e' && s.at(i - 1) != 'E') ||
                 is_whitespace)
               is_valid = false;
             break;
           case 5: // ','
           default:
             is_valid = false;
             break;
         }
      }
      ++i;
      while (i < s.size() && s.at(i) == ' ') {
         ++i;
         is_whitespace = true;
      }
      ++m;
    }
    if (!is_valid) {
      std::stringstream stream;
      for (int j = 0; j < i-1; ++j)
        stream << " ";
      stream << "^";
      std::cerr << "IsValidFloatInput() Error at position " << i - 1 << " in:\n"
           << "\t" << s << "\n" << stream.str() << "\n";
    }
    return is_valid;
};


bool ParameterInput::IsValidComplexInput(const std::string& s) {
    bool is_valid = true;
    bool is_period = false;
    bool is_e = false;
    bool is_comma = false;
    bool is_whitespace = false;
    bool is_leftbrace = false;
    int numi;
    int modi;
    int m = -1;
    int i = 0;
    // Skip initial is_whitespace.
    while (s.at(i) == ' ') {
      ++i;
    }
    while (is_valid && i < s.size()) {
      numi = kNumbers.find(s.at(i));
      modi = kComplexArray.find(s.at(i));
      if (numi == std::string::npos && modi == std::string::npos) {
         is_valid = false;
      } else if (numi != std::string::npos) {
        ++m;
        if (is_whitespace || !is_leftbrace) {
          is_valid = false;
        }
        is_whitespace = false;
      } else if (modi != std::string::npos) {
        switch (modi) {
          case 0: // '.'
            if (is_period || is_e || is_whitespace || !is_leftbrace) {
              is_valid = false;
            }
            is_period = true;
            break;
          case 1: // 'E'
          case 2: // 'e'
            if (m == 0 || is_e || is_whitespace || !is_leftbrace) {
              is_valid = false;
            }
            is_e = true;
            m = 0;
            break;
          case 3: // '-'
          case 4: // '+'
            if ((m != 0)
                || is_whitespace || !is_leftbrace) {
              is_valid = false;
            }
            break;
          case 5: // ','
            if (m == 0 || i == s.size()-1 || is_comma) {
              is_valid = false;
            }
            is_period = false;
            is_e = false;
            is_comma = true;
            m = 0;
            break;
          case 6: // '('
            if ((m > 0 && !is_comma) || is_leftbrace) {
              is_valid = false;
            }
            is_leftbrace = true;
            m = 0;
            break;
          case 7: // ')'
            if (!is_leftbrace || !is_comma || m == 0) {
               is_valid = false;
            }
            is_leftbrace = false;
            break;
          default:
            is_valid = false;
            break;
        }
        is_whitespace = false;
      }
      ++i;
      while (is_valid && i < s.size() && s.at(i) == ' ') {
         ++i;
         is_whitespace = (modi != 6 && modi != 7 && modi != 5);
      }
    }
    if (!is_valid) {
      std::stringstream stream;
      for (int j = 0; j < i-1; ++j)
        stream << " ";
      stream << "^";
      std::cerr << "IsValidComplexInput() Error at position " << i - 1 << " in:\n"
           << "\t" << s << "\n" << stream.str() << "\n";
    }
    return is_valid;
}


bool ParameterInput::IsValidArrayInput(const std::string& s) {
    bool isnum = true;
    bool period = false;
    bool e = false;
    bool comma = false;
    bool is_whitespace = false;
    int numi;
    int modi;
    int m = -1;
    int i = 0;
    // Skip initial is_whitespace.
    while (s.at(i) == ' ')
      ++i;
    while (isnum && i < s.size()) {
      ++m;
      numi = kNumbers.find(s.at(i));
      modi = kArray.find(s.at(i));
      if (numi == std::string::npos && modi == std::string::npos)
         isnum = false;
      else if (numi != std::string::npos) {
         comma = false;
         if (is_whitespace)
            isnum = false;
      }
      else if (modi != std::string::npos) {
         if (modi != 5)
           comma = false;
         switch (modi) {
           case 0: // '.'
             if (period || e || is_whitespace)
               isnum = false;
             period = true;
             break;
           case 1: // 'E'
           case 2: // 'e'
             if (m == 0 || e || is_whitespace)
               isnum = false;
             e = true;
             break;
           case 3: // '-'
           case 4: // '+'
             if ((m != 0 && s.at(i-1) != 'e' && s.at(i-1) != 'E') ||
                 is_whitespace)
               isnum = false;
             break;
           case 5: // ','
             if (m == 0 || i == s.size()-1 || comma)
               isnum = false;
             period = false;
             e = false;
             comma = true;
             is_whitespace = false;
             m = -1;
             break;
           default:
             isnum = false;
             break;
         }
      }
      ++i;
      while (i < s.size() && s.at(i) == ' ') {
         ++i;
         if (modi != 5)
           is_whitespace = true;
      }
    }
    if (!isnum) {
      std::stringstream stream;
      for (int j = 0; j < i-1; ++j)
        stream << " ";
      stream << "^";
      std::cerr << "IsValidArrayInput() Error at position " << i - 1 << " in:\n"
           << "\t" << s << "\n" << stream.str() << "\n";
    }
    return isnum;
};


bool ParameterInput::IsValidComplexArrayInput(const std::string& s) {
    bool isnum = true;
    bool period = false;
    bool e = false;
    bool comma = false;
    bool is_whitespace = false;
    bool leftbrace = false;
    int numi;
    int modi;
    int m = -1;
    int i = 0;
    // Skip initial is_whitespace.
    while (s.at(i) == ' ')
      ++i;
    while (isnum && i < s.size()) {
      numi = kNumbers.find(s.at(i));
      modi = kComplexArray.find(s.at(i));
      if (numi == std::string::npos && modi == std::string::npos) {
         isnum = false;
      } else if (numi != std::string::npos) {
         ++m;
         if (is_whitespace || !leftbrace) {
            isnum = false;
        }
         comma = false;
         is_whitespace = false;
      } else if (modi != std::string::npos) {
         switch (modi) {
           case 0: // '.'
             if (period || e || is_whitespace || !leftbrace) {
               isnum = false;
             }
             period = true;
             break;
           case 1: // 'E'
           case 2: // 'e'
             if (m == 0 || e || is_whitespace || !leftbrace) {
               isnum = false;
             }
             e = true;
             m = 0;
             break;
           case 3: // '-'
           case 4: // '+'
             if ((m != 0)
                 || is_whitespace || !leftbrace) {
               isnum = false;
             }
             break;
           case 5: // ','
             if (m == 0 || i == s.size()-1 || comma) {
               isnum = false;
             }
             period = false;
             e = false;
             comma = true;
             m = 0;
             break;
           case 6: // '('
             if ((m > 0 && !comma) || leftbrace) {
               isnum = false;
             }
             leftbrace = true;
             m = 0;
             break;
           case 7: // ')'
             if (!leftbrace || m == 0)
                isnum = false;
             leftbrace = false;
             break;
           default:
             isnum = false;
             break;
         }
         if (modi != 5)
           comma = false;
         is_whitespace = false;
      }
      ++i;
      while (isnum && i < s.size() && s.at(i) == ' ') {
         ++i;
         if (modi != 6 && modi != 7 && modi != 5)
           is_whitespace = true;
      }
    }
    if (!isnum) {
      std::stringstream ss;
      for (int j = 0; j < i-1; ++j)
        ss << " ";
      ss << "^";
      std::cerr << "IsValidComplexArrayInput() Error at position " << i-1
           << " in: \n" << s << "\n" << ss.str() << "\n";
    }
    return isnum;
};

int ParameterInput::GetCharacterOccurence(const char c,
                                          const std::string& line) {
  int count = 0;
  int position = line.find(c, 0);
  while(position != std::string::npos) {
    ++count;
    position = line.find(c, position + 1);
  }
  return count;
};
