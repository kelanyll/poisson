#include <unordered_set>

#include "PoissonRegressionTrainer.hpp"
#include "goals-predict.hpp"

PoissonRegressionTrainer::PoissonRegressionTrainer(ULDataFrame df, std::string y_col_name) : model{0} {
    ULDataFrame transformed_df = one_hot_encode(std::move(df));
    add_intercept(transformed_df);

    auto get_num_cols = [](const ULDataFrame &df) { return df.shape().second; };
    this->model = PoissonRegressionModel{get_num_cols(df)};
    //this->model.add_data()
    this->model.mle();
};

std::vector<double> PoissonRegressionTrainer::predict(ULDataFrame &df) {
    ULDataFrame transformed_df = one_hot_encode(std::move(df));
    add_intercept(transformed_df);

    // default cols -> need to store col_names for this
    
    auto get_num_rows = [](const ULDataFrame &df) { return df.shape().first; };
    std::vector<double> result;
    for (int i = 0; i < get_num_rows(df); i++) {
        result.push_back(model.predict(dataframe_row_to_boom_vector(df.get_row(i))));
    }

    return result;
};


BOOM::ConstVectorView PoissonRegressionTrainer::dataframe_row_to_boom_vector(hmdf::HeteroVector row) {
    return BOOM::ConstVectorView{std::vector<double>{0}};
};