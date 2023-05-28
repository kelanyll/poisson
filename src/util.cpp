#include <iostream>
#include "data-types.hpp"

int get_num_rows(const ULDataFrame &df) { return df.shape().first; }
