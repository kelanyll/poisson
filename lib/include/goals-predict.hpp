#pragma once

#include "data-types.hpp"

class DataFramePosRegTransformer {
public:
    virtual ULDataFrame one_hot_encode(ULDataFrame df) = 0;
    virtual std::vector<std::string> get_col_names(ULDataFrame df) = 0;
    virtual std::vector<Ptr<PoissonRegressionData>> convert_to_poisson_regression_data(ULDataFrame df, std::string y_col_name, std::vector<std::string> x_col_names) = 0;
};

class DataFramePosRegTransformerImpl : public DataFramePosRegTransformer {
public:
    ULDataFrame one_hot_encode(ULDataFrame df) override;
    std::vector<std::string> get_col_names(ULDataFrame df) override;
    std::vector<Ptr<PoissonRegressionData>> convert_to_poisson_regression_data(ULDataFrame df, std::string y_col_name, std::vector<std::string> x_col_names) override;
private:
    void one_hot_encode_string(ULDataFrame& df);
    void one_hot_encode_bool(ULDataFrame& df);
};