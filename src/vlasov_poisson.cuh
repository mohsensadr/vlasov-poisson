// vlasov_poisson.cuh

#pragma once
#include <string>
#include <vector>

void run();
void run(const std::string& pdf_type, const std::vector<float>& pdf_params);