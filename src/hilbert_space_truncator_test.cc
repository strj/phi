#include "hilbert_space_truncator.h"

#include <complex>
#include <sstream>

#include "testing/base/public/gunit.h"

#include "complex_matrix.h"
#include "numeric_types.h"

namespace {

const int kLargeSize = 4;
const int kSmallSize = 2;

class HilbertSpaceTruncatorTest : public ::testing::Test {
 protected:
  Complex large_matrix_[kLargeSize * kLargeSize];
  HilbertSpaceTruncator *truncator_;

  HilbertSpaceTruncatorTest() {
    truncator_ = new HilbertSpaceTruncator(kLargeSize, kSmallSize);
  }

  virtual ~HilbertSpaceTruncatorTest() {
    delete truncator_;
  }

  virtual void SetUp() {
    for (int i = 0; i < kLargeSize; ++i) {
      large_matrix_[i * kLargeSize + i] = Complex(i);
      for (int j = 0; j < i; ++j) {
        large_matrix_[i * kLargeSize + j] = Complex(j, -i);
        large_matrix_[j * kLargeSize + i] = Complex(j, i);
      }
    }
    truncator_->UpdateWith(large_matrix_);
  }

  void AssertMatrixAlmostEqual(const int n,
                               const Complex* expected_matrix,
                               const Complex* actual_matrix) {
    std::stringstream expected_s;
    std::stringstream actual_s;
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
        ASSERT_NEAR(std::real(expected), std::real(actual), 1e-6)
            << "Mismatch in real part of element at "
            << "[" << i << ","  << j << "]:\n"
            << expected_s.str() << "\n" << actual_s.str();
        ASSERT_NEAR(std::imag(expected), std::imag(actual), 1e-6)
            << "Mismatch in imaginary part of element at "
            << "[" << i << ","  << j << "]:\n"
            << expected_s.str() << "\n" << actual_s.str();
      }
    }
  }
};

TEST_F(HilbertSpaceTruncatorTest, TestTruncate) {
  // Set up expected matrix
  Complex expected[kSmallSize * kSmallSize];
  SetElementsToZero(kSmallSize * kSmallSize, expected);
  // Two lowest eigenvalues of large_matrix:
  expected[0] = Complex(-3.74174046);
  expected[3] = Complex(-2.12056117e-01);

  Complex actual[kSmallSize * kSmallSize];
  SetElementsToZero(kSmallSize * kSmallSize, actual);
  truncator_->Truncate(large_matrix_, actual);
  AssertMatrixAlmostEqual(kSmallSize, expected, actual);
}

// TODO(johanstr): Add UpdateWith test.

// TODO(johanstr): Add UnTruncate test.

TEST_F(HilbertSpaceTruncatorTest, TestRotate) {
  Complex actual[kSmallSize * kSmallSize];
  SetElementsToZero(kSmallSize * kSmallSize, actual);
  truncator_->Truncate(large_matrix_, actual);
  for (int i = 0; i < kLargeSize; ++i) {
    large_matrix_[i * kLargeSize + i] += Complex(i);
    for (int j = 0; j < i; ++j) {
      large_matrix_[i * kLargeSize + j] += Complex(-j, -i);
      large_matrix_[j * kLargeSize + i] += Complex(-j, i);
    }
  }
  truncator_->UpdateWith(large_matrix_);
  truncator_->Rotate(actual);

  // Set up expected matrix
  Complex expected[] = {
    Complex(-3.6248415263), Complex(0.1194649300, -3.7996962063e-01),
    Complex(0.1194649300, 3.7996962063e-01), Complex(-0.2148191697)
  };
  AssertMatrixAlmostEqual(kSmallSize, expected, actual);
}



}  // namespace
