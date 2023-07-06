#include <iostream>
#include "data-types.hpp"

template<typename ... Ts>
void print_dataframe(const ULDataFrame &df) { df.write<std::ostream, Ts...>(std::cout); }

int get_num_rows(const ULDataFrame &df);

bool ptr_poisson_regression_data_equal(const std::vector<Ptr<PoissonRegressionData>>& data1, const std::vector<Ptr<PoissonRegressionData>>& data2);

ULDataFrame transform_to_row_per_goals(ULDataFrame df);

ULDataFrame add_intercept(ULDataFrame df);