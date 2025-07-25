#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>  // C++17
#include <cstdio>      // for snprintf
#include "vlasov_poisson.cuh"



int main() {
    run();
    return 0;
}
