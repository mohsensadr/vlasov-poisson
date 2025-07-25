#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>  // C++17
#include <cstdio>      // for snprintf
#include "IO.h"
#include "vlasov_poisson_pi.cuh"



int main() {
    run();
    return 0;
}
