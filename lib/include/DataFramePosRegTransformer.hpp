#pragma once

#include <vector>
#include <string>

#include "Ptr.hpp"
#include "PoissonRegressionData.hpp"

#include "data-types.hpp"
#include "utils.hpp"

class DataFramePosRegTransformer {
public:
    virtual ULDataFrame one_hot_encode(ULDataFrame& df) = 0;
    virtual std::vector<std::string> get_col_names(const ULDataFrame& df) = 0;
    virtual std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> convert_to_poisson_regression_data(const ULDataFrame& df, const std::string& y_col_name, const std::vector<std::string>& x_col_names) = 0;
    virtual ULDataFrame add_missing_cols(ULDataFrame df, std::vector<std::string> col_names) = 0;
    virtual std::vector<std::vector<unsigned int>> get_row_vectors(ULDataFrame df, std::vector<std::string> col_names) = 0;
};

class DataFramePosRegTransformerImpl : public DataFramePosRegTransformer {
public:
    ULDataFrame one_hot_encode(ULDataFrame& df) override;
    std::vector<std::string> get_col_names(const ULDataFrame& df) override;
    virtual std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> convert_to_poisson_regression_data(const ULDataFrame& df, const std::string& y_col_name, const std::vector<std::string>& x_col_names) override;
    ULDataFrame add_missing_cols(ULDataFrame df, std::vector<std::string> col_names) override;
    std::vector<std::vector<unsigned int>> get_row_vectors(ULDataFrame df, std::vector<std::string> col_names) override;
private:
    void one_hot_encode_string(ULDataFrame& df);
    void one_hot_encode_bool(ULDataFrame& df);
};
