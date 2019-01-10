#include "hierarchy_integrator.h"
#include "phi_parameters.h"

int main(int num_args,char* argv[]){
    PhiParameters parameters;
    parameters.ReadCommandLine(num_args, argv);
    HierarchyIntegrator integrator(&parameters);
    integrator.Launch();
    return 0;
};

