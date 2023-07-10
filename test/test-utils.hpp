#include <iostream>

#include "data-types.hpp"
#include "utils.hpp"

template<typename ... Ts>
bool check_df_equal(ULDataFrame actual_df, ULDataFrame expected_df) {
    bool is_equal = actual_df.is_equal<Ts...>(expected_df);
    if (!is_equal) {
        std::cout << "DataFrames are not equal." << std::endl;

        std::cout << "Actual:" << std::endl;
        print_dataframe<Ts...>(actual_df);

        std::cout << "Expected:" << std::endl;
        print_dataframe<Ts...>(expected_df);
    }

    return is_equal;
}