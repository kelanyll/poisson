#include <unordered_set>

#include "PoissonRegressionTrainer.hpp"

PoissonRegressionTrainer::PoissonRegressionTrainer() : transforms{new DataFramePosRegTransformerImpl{}} {}

PoissonRegressionTrainer::PoissonRegressionTrainer(DataFramePosRegTransformer* transforms_val) : transforms{transforms_val} {}

PoissonRegressionModelData PoissonRegressionTrainer::get_poisson_regression_model_data(ULDataFrame df, std::string y_col_name) {
    ULDataFrame transformed_df = transforms->one_hot_encode(std::move(df));

    std::vector<std::string> col_names{transforms->get_col_names(transformed_df)};
    std::vector<std::string> x_col_names;
    std::copy_if(col_names.begin(), col_names.end(), std::back_inserter(x_col_names), [y_col_name](std::string col_name) {
        return col_name != y_col_name;
    });

    std::vector<Ptr<PoissonRegressionData>> data{transforms->convert_to_poisson_regression_data(transformed_df, y_col_name, x_col_names)};

    return PoissonRegressionModelData{data, x_col_names};
}

std::vector<std::vector<unsigned int>> PoissonRegressionTrainer::generate_x(ULDataFrame df, PoissonRegressionModelData model_data) {
    ULDataFrame transformed_df{transforms->one_hot_encode(std::move(df))};
    transformed_df = transforms->add_missing_cols(transformed_df, model_data.x_col_names);
    
    return transforms->get_row_vectors(transformed_df, model_data.x_col_names);
}
