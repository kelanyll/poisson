#pragma once

#include <iostream>

#include "Ptr.hpp"
#include "PoissonRegressionData.hpp"

#include "data-types.hpp"

template<typename ... Ts>
void print_dataframe(const ULDataFrame &df) { df.write<std::ostream, Ts...>(std::cout); }

int get_num_rows(const ULDataFrame &df);

bool ptr_poisson_regression_data_equal(const std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>>& data1, const std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>>& data2);

ULDataFrame transform_to_row_per_goals(const ULDataFrame& df);

ULDataFrame add_intercept(ULDataFrame&& df);