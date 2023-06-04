#include "data-types.hpp"

ULDataFrame transform_to_row_per_goals(ULDataFrame df);

ULDataFrame one_hot_encode(ULDataFrame df);

void one_hot_encode_string(ULDataFrame& df);

void one_hot_encode_bool(ULDataFrame& df);

void add_intercept(ULDataFrame& df);

std::vector<Ptr<PoissonRegressionData>> convert_to_poisson_regression_data(ULDataFrame df, std::string y_col_name);
