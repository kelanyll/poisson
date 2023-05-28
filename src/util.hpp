#include <iostream>
#include "data-types.hpp"

template<typename ... Ts>
void print_dataframe(const ULDataFrame &df) { df.write<std::ostream, Ts...>(std::cout); }

int get_num_rows(const ULDataFrame &df);
