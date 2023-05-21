#include <unordered_set>

#include "PoissonRegressionTrainer.hpp"
#include "goals-predict.hpp"

// still needs to store col names (rather than the full data frame) - or col values
// dataframe is transformed first. and then passed (as a reference to trainer)
// lets just store a pointer to it instead
// will we be able to do it in place?
PoissonRegressionTrainer::PoissonRegressionTrainer(ULDataFrame df, std::string y_col_name) : model{0} {
    ULDataFrame transformed_df = add_intercept_column(one_hot_encode(df));
    add_intercept_column(df);

    auto get_num_cols = [](const ULDataFrame &df) { return df.shape().second; };
    this->model = PoissonRegressionModel{get_num_cols(df)};
    //this->model.add_data()
    this->model.mle();
};

std::vector<double> PoissonRegressionTrainer::predict(ULDataFrame &df) {
    transform_columns_in_place(df, this->one_hot_encoded_column_names);

    auto get_num_rows = [](const ULDataFrame &df) { return df.shape().first; };
    std::vector<double> result;
    for (int i = 0; i < get_num_rows(df); i++) {
        result.push_back(model.predict(dataframe_row_to_boom_vector(df.get_row(i))));
    }

    return result;
};

// make this a visitor
std::vector<std::string> PoissonRegressionTrainer::get_one_hot_encoded_column_names(const ULDataFrame &df) {
    auto unique_home_vals = df.get_col_unique_values<std::string>("home");
    auto unique_away_vals = df.get_col_unique_values<std::string>("away");
    std::unordered_set<std::string> teams{unique_home_vals.begin(), unique_home_vals.end()};
    teams.insert(unique_away_vals.begin(), unique_away_vals.end());

    std::vector<std::string> one_hot_encoded_column_names;
    auto generate_team_cols = [](std::string team) {
        return std::vector<std::string>{team + "_team_1", team + "_team_2"};
    };

    std::transform_reduce(teams.begin(), teams.end(), one_hot_encoded_column_names, [](std::vector<std::string> a, std::vector<std::string> b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    }, generate_team_cols);

    return one_hot_encoded_column_names;
};

void PoissonRegressionTrainer::transform_columns_in_place(ULDataFrame &data_frame, const std::vector<std::string> &col_names) {
    return;
};

ULDataFrame PoissonRegressionTrainer::add_intercept_column(ULDataFrame data_frame) {
    return ULDataFrame{};
};

BOOM::ConstVectorView PoissonRegressionTrainer::dataframe_row_to_boom_vector(hmdf::HeteroVector row) {
    return BOOM::ConstVectorView{std::vector<double>{0}};
};