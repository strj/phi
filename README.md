# phi
Parallel Hierarchy Integrator

[PHI](http://www.ks.uiuc.edu/Research/phi/) is a software package for
integrating the hierarchy equation of motion (HEOM) to compute the noise-
averaged density matrix evolution for a quantum system in contact with a thermal
 environment. PHI is a multi-threaded program to run on shared memory computers.

For a detailed description of the method and its implementation please see the
following paper: ["Open quantum dynamics calculations with the hierarchy
equations of motion on parallel computers." Johan Strümpfer and Klaus Schulten.
Journal of Chemical Theory and Computation, 8:2808-2816, 2012.](
https://pubs.acs.org/doi/pdf/10.1021/ct3003833)

Free text available
[here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3480185/)

## Usage

There is a somewhat out of date users guide available [here](
http://www.ks.uiuc.edu/Research/phi/phi_ug.pdf)

What's missing is the ability to perform quantum annealing computations. See
the example dir for a simple example input.

## Compile on Ubuntu

If you don't have BLAS + Lapack:
```
sudo apt-get install libatlas-base-dev liblapack-dev liblapacke-dev
```
Note that ATLAS is used for the BLAS implementation. There are some API
differences with OpenBLAS (e.g. `double *` used as parameters for complex valued
functions) that require changes in `src/blas_wrapper.h` if you wish to use
OpenBLAS.

Then to build phi:
```
cmake .
make
```

## Apologies

So the code herein is not very clean, not well tested, not well documented nor
even in a good consistent style. It's suffering from a number of years of
neglect so any love it recieves is much appreciated. 

## License

PHI was developed at the University of Illinois at Urbana-Champaign and is
distributed under the University of Illinois open source licence.
See LICENCE.txt
