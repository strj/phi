"""Reads PHI output files.

PHI is simulation software that outputs the time dependent density matrix of a
quantum system in contact with a thermal environment. Often calculations need 
to be stopped and restarted, resulting in multiple files. This reader makes it
easier to import the dynamics in the files for manipulation in Python.

PHI output is formatted into two sections, first the parameters of the
simulation are output at the top of the file, followed by the density matrix at
each timestep (which can be interspersed with timing updates). Density matrix
lines (referred to as dynamics lines) don't start with '#'. Lines that do are
either parameter lines (if they are at the top of the file) or comment lines.
E.g.,
#NumStates=2                            <- Parameter line
#Timestep=0.1                           <- Parameter line
  0.0 (1,0) (0,0) (0,0) (0,0)           <- 1st dynamics line = end of parameters
#Time remaining=2 hrs                   <- Comments between dynamics lines
  0.1 (0.8,0) (0.1,0) (0.1,0) (0.2,0)   <- Dynamics line
"""

import re
import numpy

_DYNAMICS_LINE_RE = re.compile(r'^ *[0-9].*')
_TIME_RE = re.compile(r'^(?: *)[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]+)? ')
_COMPLEX_VALUES_RE = re.compile(
    r'(?:\A^| +|,)\(([^,]*),([^)]*)\)|(?P<error>.+)')
_PARAMETER_LINE_RE = re.compile(r'([^:=]*)(?:=.+|:)$')


class PhiOutputFileFormatError(Exception):
  """Raised for formatting errors in the output file."""


class PhiOutputError(Exception):
  """Raised while reading phi output."""


class _PhiOutputFile(object):
  """Wrapper around for a PHI output file.

  Since these files can be very large, some analysis is performed on
  initialization so that selected lines can be easily read without needing to
  store the whole file.
  """

  def __init__(self, filename):
    """Initializes the wrapper for the file specified by filename.

    Args:
      filename: The file name (string).
    """
    self.filename = filename
    self._ScanFile()

  def _ScanFile(self):
    """Scans the file to determine the parameter and dynamics line ranges.

    Raises:
      IOError: If a the file could not be read.
      PhiOutputFileFormatError: If the time value from a dynamics line could
          not be read.
    """
    self.num_dynamics_lines = 0
    self._dynamics_start_position = None
    self._dynamics_end_position = None
    self.parameter_lines = []
    self.start_time = None
    self.end_time = None
    with open(self.filename) as input_file:
      pos = input_file.tell()
      line = input_file.readline()
      # Find the end of the parameters.
      while line and line[0] == '#':
        self.parameter_lines.append(line[1::].rstrip())
        pos = input_file.tell()
        line = input_file.readline()
      if not line:
        return
      self.start_time = float(_TIME_RE.match(line).group())
      # Loop through the remaining lines in the file to determine the time range
      # and number of dynamics lines.
      self._dynamics_start_position = pos
      while line:
        # Skip comment lines.
        if line[0] != '#':
          match = _TIME_RE.match(line)
          if not match:
            e = ('Error reading time from dynamics line in file %s:\n%s' % (
                self.filename, line))
            raise PhiOutputFileFormatError(e)
          self.num_dynamics_lines += 1
          self._dynamics_end_position = pos
          try:
            time = float(match.group())
          except (AttributeError, ValueError):
            e = ('Error reading time from dynamics line in file %s:\n%s'
                 % (self.filename, line))
            raise PhiOutputFileFormatError(e)
        pos = input_file.tell()
        line = input_file.readline()
      self.end_time = time
      self._time_range = self.end_time - self.start_time

  def GetDynamicsLinesWithTimestep(
      self, timestep, start_time=None, end_time=None):
    """Reads the dynamics lines spaced apart by at least the given timestep.

    Args:
      timestep: The required minumum simulated time between lines (float)
      start_time: The starting time (float).
      end_time: The ending time (float).

    Returns:
      The dynamics lines (string array).
    """
    if self.num_dynamics_lines == 0 or self._dynamics_start_position is None:
      return []
    if end_time is None or end_time > self.end_time:
      end_time = self.end_time
    if start_time is None or start_time < self.start_time:
      start_time = self.start_time
    if end_time < start_time:
      return []
    time_range = end_time - start_time
    if timestep > time_range:
      with open(self.filename) as input_file:
        input_file.seek(self._dynamics_start_position)
        return [input_file.readline()]

    number = int(numpy.ceil(time_range / timestep))
    line_skip = max(1, self.num_dynamics_lines / number)
    line_count = 0
    last_dynamics_line = None
    lines = []
    with open(self.filename) as input_file:
      input_file.seek(self._dynamics_start_position)
      for line in input_file:
        match = _TIME_RE.match(line)
        if not match:
          # A comment line between dynamics lines.
          continue
        last_dynamics_line = line
        time = float(match.group())
        if time >= start_time and time <= end_time:
          if line_count % line_skip == 0:
            lines.append(line.rstrip())
          line_count += 1
        elif time > end_time:
          break

    if last_dynamics_line:
      last_time = float(_TIME_RE.match(last_dynamics_line).group())
      final_time = float(_TIME_RE.match(lines[-1]).group())
      if final_time < last_time and last_time <= end_time:
        # Make sure we capture the dynamics line closest to the end time.
        lines[-1] = last_dynamics_line.rstrip()
    return lines


def _ReadRealVector(line):
  """Reads a real-valued entries from a line.

  Args:
    line: The line to read, with a format like '1.0,2.0,3.,4.0' (string).

  Returns:
    The vector (float numpy.array).

  Raises:
    PhiOutputFileFormatError: For incorrectly formatted lines.
  """
  try:
    float_array = map(float, line.split(','))
  except ValueError:
    exception = 'Error reading real array from parameter line:\n%s' % line
    raise PhiOutputFileFormatError(exception)
  return numpy.array(float_array)


def _ReadRealMatrix(lines):
  """Reads a real-valued matrix from the input lines.

  Args:
    lines: An iterator for the lines to read (_SingleBackStepIterator).

  Returns:
    The matrix (2D float numpy.array).

  Raises:
    PhiOutputFileFormatError: For incorrectly formatted lines.
  """
  matrix = []
  for line in lines:
    if _PARAMETER_LINE_RE.match(line):
      break
    matrix.append(_ReadRealVector(line))
  lines.StepBack()
  return numpy.array(matrix)


def _ToComplex(real, imag):
  """Returns the complex value from the real and imaginary strings.

  Args:
    real: The real part of the number (string)
    imag: The imaginary part of the number (string)

  Returns:
    Then complex number.

  Raises:
    PhiOutputFileFormatError: If real or imag are not convertable to floats.
  """
  try:
    real, imag = map(float, (real, imag))
  except ValueError:
    exception = 'Incorrectly formatted complex pair (%s,%s)' % (real, imag)
    raise PhiOutputFileFormatError(exception)
  return complex(real, imag)


def _ReadComplex(string):
  """Reads a complex value from a string.

  Args:
    string: A complex value represented by '(real,imaginary)' (string).

  Returns:
    The complex value (complex).

  Raises:
    PhiOutputFileFormatError: For incorrectly formatted strings.
  """
  matches = _COMPLEX_VALUES_RE.findall(string)
  if not matches or len(matches) > 1 or matches[0][2]:
    exception = 'Incorrectly formatted complex pair: %s' % string
    raise PhiOutputFileFormatError(exception)
  return _ToComplex(matches[0][0], matches[0][1])


def _ReadComplexVector(line):
  """Reads a complex-valued vector from a line.

  Args:
    line: The line to read, with a format like '#(1.0,2.0),(3.0,4.0)' (string).

  Returns:
    The vector (complex numpy.array).

  Raises:
    PhiOutputFileFormatError: For incorrectly formatted lines.
  """
  complex_array = []
  for real, imag, error in _COMPLEX_VALUES_RE.findall(line):
    if error:
      exception = 'Garbage %s when reading complex array from line:\n%s' % (
          error, line)
      raise PhiOutputFileFormatError(exception)
    try:
      value = _ToComplex(real, imag)
    except PhiOutputFileFormatError as e:
      exception = '%s on line:\n%s' % (e, line)
      raise PhiOutputFileFormatError(exception)
    complex_array.append(value)
  return numpy.array(complex_array)


def _ReadComplexMatrix(lines):
  """Reads a complex-valued matrix from the input lines.

  Args:
    lines: An iterator for the lines to read (_SingleBackStepIterator).

  Returns:
    The matrix (2D complex numpy.array).

  Raises:
    PhiOutputFileFormatError: For incorrectly formatted lines.
  """
  matrix = []
  for line in lines:
    if _PARAMETER_LINE_RE.match(line):
      break
    matrix.append(_ReadComplexVector(line))
  lines.StepBack()
  return numpy.array(matrix)


class SuccessiveStepBackException(Exception):
  """Raised when taking successive back steps with _SingleBackStepIterator."""


class _SingleBackStepIterator(object):
  """An iterator for a collection that can take a step back."""

  def __init__(self, iterable):
    """Initialized the iterator.

    Args:
      iterable: An interable object.
    """
    self._base_iterator = iter(iterable)
    self._return_prev = False

  def __iter__(self):
    return self

  def next(self):
    """Steps to the next item in the collection and returns the previous."""
    if self._return_prev:
      self._return_prev = False
    else:
      self._prev_item = self._base_iterator.next()
    return self._prev_item

  def StepBack(self):
    """Set the iterator to return the previously returned item.

    Raises:
      SuccessiveStepBackException: For successive calls to StepBack.
    """
    if self._return_prev:
      raise SuccessiveStepBackException(
          'Error: Cannot make consecutive calls to StepBack()')
    self._return_prev = True


class PhiOutput(object):
  """Reads the output of a single, or multiple, PHI output files.

  The parameters for the simulation are taken from the output of the first file.

  At least the following field will be publically accessible after reading the
  files:
    num_states (int)
    hilbert_space_size (int)
    density_matrix (array of complex matrices)
    time - (array of floats)

  Optional fields that may also be populated if their values are present in the
  output file are:
    num_coupling_terms (int)
    num_matsubara_terms (int)
    temperature (float)
    initial_timestep (float)
    hierarchy_truncation (int)
    integration_time (float)
    hbar (float)
    boltzmann_constant (float)
    use_time_local_truncation (bool)
    threads (int)
    integrator (string)
    rkf45_tolerance (float)
    rkf45_min_timestep (float)
    hierarchy_num_nodes (int)
    gamma (float array)
    lambd (float array)
    kappa (float array)
    hamiltonian (complex matrix)
    transverse_hamiltonian (complex matrix)
    longitudinal_hamiltonian (complex matrix)
    initial_density_matrix (complex matrix)
    diagonal_coupling_terms (real matrix)
    full_coupling_terms (real matrix)
  """

  def __init__(self, filenames, max_lines=100):
    """Initializes PhiOutput.

    Args:
      filenames: The list of filenames to read.
      max_lines: The max lines to read over all files (could read up to one
          more) (int).

    Raises:
      PhiOutputError: When unable to read the list of filenames or when a file
          is incomplete.
      PhiOutputFileFormatError: For formatting errors within the output files.
    """
    if not (isinstance(filenames, list) or isinstance(filenames, tuple)):
      raise PhiOutputError(
          'Need to initialize PhiOutput with a list of file names')
    self._filenames = filenames
    self.files = []
    for name in self._filenames:
      output_file = _PhiOutputFile(name)
      if (output_file.start_time is not None and
          output_file.end_time is not None):
        self.files.append(output_file)
    if not self.files:
      raise PhiOutputError('No files in %s had any dynamics output' %
                           str(filenames))
    # Order files by their start time.
    self.files = sorted(self.files, key=lambda entry: entry.start_time)
    self.hilbert_space_size = None
    self.num_state = None
    self._ReadParameters()
    if not self.hilbert_space_size or not self.num_states:
      raise PhiOutputError('NumStates not specified in %s' % self.files[0])
    self.first_time = self.files[0].start_time
    self.last_time = self.files[-1].end_time
    self.min_timestep = (self.last_time - self.first_time) / max_lines
    self.density_matrix = []
    self.populations = []
    self.time = []
    self.num_dynamics_lines = 0
    self._ReadDynamics()

  def _ReadDynamics(self):
    """Reads dynamics from the set of files.

    Raises:
      PhiOutputFileFormatError: If the dynamics from a file could not be read.
    """
    time = self.first_time
    for output_file in self.files:
      dynamics_lines = output_file.GetDynamicsLinesWithTimestep(
          self.min_timestep, start_time=time)
      try:
        self._ProcessDynamics(dynamics_lines)
      except PhiOutputFileFormatError as e:
        exception = 'Error reading file %s:\n%s' % (output_file.filename, e)
        raise PhiOutputFileFormatError(exception)
      self.num_dynamics_lines += len(dynamics_lines)
      if self.time:
        time = self.time[-1] + self.min_timestep

  def _ProcessDynamics(self, dynamics_lines):
    """Proccesses a list of dynamics lines.

    Args:
      dynamics_lines: The dynamics lines to process (string list).

    Raises:
      PhiOutputFileFormatError: If there is an error reading the dynamics lines.
    """
    num_expected_parts = self.num_states * self.num_states + 1
    for line in dynamics_lines:
      parts = line.split()
      if len(parts) != num_expected_parts:
        raise PhiOutputFileFormatError('Incomplete dynamics line\n%s' % line)
      # Earlier processing assures that this will never error out.
      time = float(parts[0])
      density_matrix = numpy.zeros(self.hilbert_space_size, complex)
      populations = numpy.zeros(self.num_states, float)
      for i in xrange(self.num_states):
        for j in xrange(self.num_states):
          entry = parts[1 + i * self.num_states + j]
          try:
            density_matrix[i, j] = _ReadComplex(entry)
          except PhiOutputFileFormatError:
            exception = ('Error reading rho[%d,%d] with entry %s on line:\n%s' %
                         (i, j, entry, line))
            raise PhiOutputFileFormatError(exception)
        populations[i] = density_matrix[i, i].real
      self.density_matrix.append(density_matrix)
      self.populations.append(populations)
      self.time.append(time)

  def _ReadParameters(self):
    """Reads the parameters from the first file.

    Raises:
      PhiOutputFileFormatError: If a non-scalar parameter could not be read.
      TypeError: If a scalar parameter could not be read.
    """
    lines = _SingleBackStepIterator(self.files[0].parameter_lines)
    for line in lines:
      if line.find('=') != -1:
        param_name, param_value = line.split('=', 1)
        if param_name == 'NumStates':
          self.num_states = int(param_value)
          self.hilbert_space_size = (self.num_states, self.num_states)
        elif param_name == 'CouplingTerms':
          self.num_coupling_terms = int(param_value)
        elif param_name == 'MatsubaraTerms':
          self.num_matsubara_terms = int(param_value)
        elif param_name == 'Temperature':
          self.temperature = float(param_value)
        elif param_name == 'Timestep':
          self.initial_timestep = float(param_value)
        elif param_name == 'HierarchyTruncation':
          self.hierarchy_truncation = int(param_value)
        elif param_name == 'Time':
          self.integration_time = float(param_value)
        elif param_name == 'ReducedPlanckConstant':
          self.hbar = float(param_value)
        elif param_name == 'BoltzmannConstant':
          self.boltzmann_constant = float(param_value)
        elif param_name == 'TimeLocal':
          self.use_time_local_truncation = param_value == '1'
        elif param_name == 'Threads':
          self.threads = int(param_value)
        elif param_name == 'Integrator':
          self.integrator = param_value
        elif param_name == 'RKF45tolerance':
          self.rkf45_tolerance = float(param_value)
        elif param_name == 'RKF45mindt':
          self.rkf45_min_timestep = float(param_value)
        elif param_name == 'Estimated Number of Matrices':
          self.hierarchy_num_nodes = int(param_value)
      elif line.endswith(':'):
        param_name = line[:-1]
        if param_name == 'gamma':
          self.gamma = _ReadRealVector(lines.next())
        elif param_name == 'lambda':
          self.lambd = _ReadRealVector(lines.next())
        elif param_name == 'kappa':
          self.kappa = _ReadRealVector(lines.next())
        elif param_name == 'Hamiltonian':
          self.hamiltonian = _ReadComplexMatrix(lines)
        elif param_name == 'TransverseHamiltonian':
          self.transverse_hamiltonian = _ReadComplexMatrix(lines)
        elif param_name == 'LongitudinalHamiltonian':
          self.longitudinal_hamiltonian = _ReadComplexMatrix(lines)
        elif param_name == 'InitialDensityMatrix':
          self.initial_density_matrix = _ReadComplexMatrix(lines)
        elif param_name == 'CorrelatedCouplingTerms':
          self.diagonal_coupling_terms = _ReadRealMatrix(lines)
        elif param_name == 'FullBathTerms':
          self.full_coupling_terms = _ReadRealMatrix(lines)

