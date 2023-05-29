#include "data-types.hpp"

ULDataFrame transform_to_row_per_goals(ULDataFrame df);

ULDataFrame one_hot_encode(ULDataFrame df);

void add_intercept(ULDataFrame& df);