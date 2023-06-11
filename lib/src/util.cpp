#include <iostream>
#include "data-types.hpp"

int get_num_rows(const ULDataFrame &df) { return df.shape().first; }

bool ptr_poisson_regression_data_equal(const std::vector<Ptr<PoissonRegressionData>>& data1, const std::vector<Ptr<PoissonRegressionData>>& data2) {
    bool is_equal = data1.size() == data2.size();
    for (int i = 0; i < data1.size() && is_equal; i++) {
        bool y_equal = data1[i]->y() == data2[i]->y();
        bool x_equal = data1[i]->x() == data2[i]->x();
        if (!(y_equal && x_equal)) {
            is_equal = false;
        }
    }

    return is_equal;
}