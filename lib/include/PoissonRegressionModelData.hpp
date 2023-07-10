#pragma once

#include <vector>
#include <string>

#include "Ptr.hpp"
#include "PoissonRegressionData.hpp"

#include "data-types.hpp"
#include "utils.hpp"

class PoissonRegressionModelData {
public:
    PoissonRegressionModelData(std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> data_val, std::vector<std::string> x_col_names_val);
    const std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> data;
    const std::vector<std::string> x_col_names;
    // implement a move constructor
};

bool operator==(const PoissonRegressionModelData& model_data1, const PoissonRegressionModelData& model_data2);