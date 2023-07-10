#include "PoissonRegressionModelData.hpp"

PoissonRegressionModelData::PoissonRegressionModelData(std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> data_val, 
    std::vector<std::string> x_col_names_val) : data{data_val}, x_col_names{x_col_names_val} {}

bool operator==(const PoissonRegressionModelData& model_data1, const PoissonRegressionModelData& model_data2) {
    return ptr_poisson_regression_data_equal(model_data1.data, model_data2.data) && model_data1.x_col_names == model_data2.x_col_names;
}