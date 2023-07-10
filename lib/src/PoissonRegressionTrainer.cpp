#include "PoissonRegressionTrainer.hpp"

PoissonRegressionTrainer::PoissonRegressionTrainer() : transforms{new DataFramePosRegTransformerImpl{}} {}

PoissonRegressionTrainer::PoissonRegressionTrainer(DataFramePosRegTransformer* transforms_val) : transforms{transforms_val} {}

PoissonRegressionModelData PoissonRegressionTrainer::get_poisson_regression_model_data(ULDataFrame&& df, std::string y_col_name) {
    transforms->one_hot_encode(df);

    std::vector<std::string> col_names{transforms->get_col_names(df)};
    std::vector<std::string> x_col_names;
    std::copy_if(col_names.begin(), col_names.end(), std::back_inserter(x_col_names), [y_col_name](std::string col_name) {
        return col_name != y_col_name;
    });

    return PoissonRegressionModelData{
        transforms->convert_to_poisson_regression_data(df, y_col_name, x_col_names), 
        x_col_names
    };
}

std::vector<std::vector<unsigned int>> PoissonRegressionTrainer::generate_x(ULDataFrame df, PoissonRegressionModelData model_data) {
    transforms->one_hot_encode(df);
    ULDataFrame transformed_df = transforms->add_missing_cols(df, model_data.x_col_names);
    
    return transforms->get_row_vectors(transformed_df, model_data.x_col_names);
}
