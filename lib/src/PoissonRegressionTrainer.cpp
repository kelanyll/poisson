#include <unordered_set>

#include "PoissonRegressionTrainer.hpp"
#include "goals-predict.hpp"

PoissonRegressionTrainer::PoissonRegressionTrainer(ULDataFrame df, std::string y_col_name) : model{0} {
    ULDataFrame transformed_df = one_hot_encode(std::move(df));
    add_intercept(transformed_df);

    auto get_num_cols = [](const ULDataFrame &df) { return df.shape().second; };
    this->model = PoissonRegressionModel{get_num_cols(df)};
    this->model.set_data(convert_to_poisson_regression_data(df, y_col_name));
    this->model.mle();
};

std::vector<double> PoissonRegressionTrainer::predict(ULDataFrame &df) {
    ULDataFrame transformed_df = one_hot_encode(std::move(df));
    add_intercept(transformed_df);
    
    auto get_num_rows = [](const ULDataFrame &df) { return df.shape().first; };
    std::vector<double> result;
    // for (int i = 0; i < get_num_rows(df); i++) {
    //     result.push_back(model.predict(dataframe_row_to_boom_vector(df.get_row(i))));
    // }

    return result;
};
