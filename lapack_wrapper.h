#ifndef PHI_LAPACK_WRAPPER_H_
#define PHI_LAPACK_WRAPPER_H_

#include "numeric_types.h"

#include "third_party/lapack/lapack.h"

#ifdef SINGLEPRECISION
namespace lapack {
  typedef lapack::complex Complex;
  typedef lapack::real Float;
}
#define phi_xheevr(job, range, format, n, data, lda, vi, vu, li, lu, tol, nf, \
                   z, v, ldz, i, w, lw, rw, lrw, iw, liw, info) \
        cheevr_(job, range, format, n, data, lda, vi, vu, lu, lu, tol, nf, \
                z, v, ldz, i, w, lw, rw, lrw, iw, liw, info)
#define phi_real_xpttrf(n, d, e, info) spttrf_(n, d, e, info)
#define phi_real_xpttrs(n, nrhs, d, e, b, ldb, info) \
        spttrs_(n, nrhs, d, e, b, ldb, info)
#define phi_complex_xpttrf(n, d, e, info) cpttrf_(n, d, e, info)
#define phi_complex_xpttrs(uplo, n, nrhs, d, e, b, ldb, info) \
        cpttrs_(uplo, n, nrhs, d, e, b, ldb, info)
#else
namespace lapack {
  typedef lapack::doublecomplex Complex;
  typedef lapack::doublereal Float;
}
#define phi_xheevr(job, range, format, n, data, lda, vi, vu, li, lu, tol, nf, \
                   z, v, ldz, i, w, lw, rw, lrw, iw, liw, info) \
        zheevr_(job, range, format, n, data, lda, vi, vu, li, lu, tol, nf, \
                 z, v, ldz, i, w, lw, rw, lrw, iw, liw, info)
#define phi_real_xpttrf(n, d, e, info) dpttrf_(n, d, e, info)
#define phi_real_xpttrs(n, nrhs, d, e, b, ldb, info) \
        dpttrs_(n, nrhs, d, e, b, ldb, info)
#define phi_complex_xpttrf(n, d, e, info) zpttrf_(n, d, e, info)
#define phi_complex_xpttrs(uplo, n, nrhs, d, e, b, ldb, info) \
        zpttrs_(uplo, n, nrhs, d, e, b, ldb, info)
#endif

struct ZheevrParameters {
  char job[1], range[1], format[1];
  lapack::integer size;
  lapack::integer il, iu;
  lapack::Float tolerance;
  lapack::integer lda;
  lapack::integer ldz;
  lapack::integer number_eigenvalues_returned;
  lapack::integer* eigenvalues_support;
  lapack::Complex* complex_workspace;
  lapack::Float* real_workspace;
  lapack::integer* integer_workspace;
  lapack::integer size_complex_workspace;
  lapack::integer size_real_workspace;
  lapack::integer size_integer_workspace;
  lapack::integer info;
  lapack::Float* eigenvalues_real;
  lapack::Complex* local_matrix;

  // Unused parameters but needed for zheevr_
  lapack::Float vl, vu;

  void Initialize(int rank, int num_vectors) {
    job[0] = 'V';
    bool calculate_all = num_vectors == rank;
    range[0] = calculate_all ? 'A' : 'I';
    format[0] = 'L';

    size = rank;
    lda = rank;
    ldz = rank;

    local_matrix = new lapack::Complex[size * size];

    char safe_tolerance = 'S';
    tolerance = dlamch_(&safe_tolerance);  // 'Safe' tolerance.

    vl = 0;  // Not used.
    vu = 1;  // Not used.
    il = 1;
    iu = num_vectors;

    int M = calculate_all ? size : iu - il + 1;

    eigenvalues_real = new lapack::Float[size];

    lapack::Complex * eigenvectors;
    eigenvectors = new lapack::Complex[size*size];

    eigenvalues_support = new lapack::integer[2 * M];

    size_complex_workspace = -1;
    Complex complex_workspace_size_estimate;
    size_real_workspace = -1;
    Float real_worskspace_size_estimate;
    size_integer_workspace = -1;
    lapack::integer integer_workspace_size_estimate;

    // Optimal size estimate for workspaces.
    phi_xheevr(job,
               range,
               format,
               &size,
               local_matrix,
               &lda,
               &vl,
               &vu,
               &il,
               &iu,
               &tolerance,
               &number_eigenvalues_returned,
               eigenvalues_real,
               eigenvectors,
               &ldz,
               eigenvalues_support,
               &complex_workspace_size_estimate,
               &size_complex_workspace,
               &real_worskspace_size_estimate,
               &size_real_workspace,
               &integer_workspace_size_estimate,
               &size_integer_workspace,
               &info);

    size_complex_workspace =
        (lapack::integer)real(complex_workspace_size_estimate);
    size_real_workspace = (lapack::integer)(real_worskspace_size_estimate);
    size_integer_workspace = integer_workspace_size_estimate;

    complex_workspace = new lapack::Complex[size_complex_workspace];
    real_workspace = new lapack::Float[size_real_workspace];
    integer_workspace = new lapack::integer[size_integer_workspace];
    delete []eigenvectors;
  }

  explicit ZheevrParameters(int size, int num_eigenvectors_to_compute) {
    Initialize(size, num_eigenvectors_to_compute);
  }


  explicit ZheevrParameters(int size) {
    Initialize(size, size);
  }

  ~ZheevrParameters() {
    delete []eigenvalues_support;
    delete []complex_workspace;
    delete []real_workspace;
    delete []integer_workspace;
    delete []eigenvalues_real;
    delete []local_matrix;
  }
};

#endif  // PHI_LAPACK_WRAPPER_H_

