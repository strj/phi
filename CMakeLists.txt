cmake_minimum_required(VERSION 2.8.9)
project (phi)
set(CMAKE_BUILD_TYPE Release)

include_directories(src)

set(SOURCES
	src/phi.cc
	src/barrier.cc
	src/blas_wrapper.cc
	src/complex_matrix.cc
	src/cubic_spline.cc
	src/hierarchy.cc
	src/hierarchy_integrator.cc
	src/hierarchy_node.cc
	src/hierarchy_updater.cc
	src/hilbert_space_truncator.cc
	src/hilbert_space_truncator.h
	src/parameter_input.cc
	src/phi_parameters.cc
	src/restart_helper.cc
	src/timer.cc
	)


find_library(LAPACK_LIBRARY lapack)
find_library(BLAS_LIBRARY blas)
find_library(PTHREAD_LIBRARY pthread)

message("Lapack link library is ${LAPACK_LIBRARY}")
message("Blas link library is ${BLAS_LIBRARY}")

add_executable(phi ${SOURCES})
target_link_libraries(phi ${LAPACK_LIBRARY} ${BLAS_LIBRARY} ${PTHREAD_LIBRARY})


