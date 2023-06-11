#include <iostream>
#include <sstream>
#include "PoissonRegressionModel.hpp"
#include <DataFrame/DataFrame.h>
#include "PoissonRegressionTrainer.hpp"
#include "data-types.hpp"
#include "util.hpp"

ULDataFrame get_data() {
    // to be replaced with loading from csv
    ULDataFrame df;
    df.load_data(std::vector<unsigned long>{1,2,3}, 
        std::make_pair("home", std::vector<std::string>{"Wolves", "Sunderland","Chelsea"}),
        std::make_pair("away", std::vector<std::string>{"Sunderland", "Chelsea","Wolves"}),
        std::make_pair("home_goals", std::vector<unsigned int>{0, 2, 2}),
        std::make_pair("away_goals", std::vector<unsigned int>{2, 1, 0})
    );

    return df;
}

/*
home, away, home_goals, away_goals -> team, opponent, home, goals
*/
ULDataFrame transform_to_row_per_goals(ULDataFrame df) {
    ULDataFrame new_df{};

    std::vector<unsigned long> idx{df.get_index()};
    new_df.load_index(idx.begin(), idx.end());

    new_df.load_column<std::string>("team", df.get_column<std::string>("home"));
    new_df.load_column<std::string>("opponent", df.get_column<std::string>("away"));
    new_df.load_column<bool>("home", std::vector<bool>(get_num_rows(df), true));
    new_df.load_column<unsigned int>("goals", df.get_column<unsigned int>("home_goals"));

    ULDataFrame away_rows{};
    
    int num_rows = get_num_rows(df);
    std::vector<unsigned long> away_rows_idx(num_rows);
    std::iota(away_rows_idx.begin(), away_rows_idx.end(), idx.back() + 1);
    away_rows.load_index(away_rows_idx.begin(), away_rows_idx.end());

    away_rows.load_column<std::string>("team", df.get_column<std::string>("away"));
    away_rows.load_column<std::string>("opponent", df.get_column<std::string>("home"));
    away_rows.load_column<bool>("home", std::vector<bool>(get_num_rows(df), false));
    away_rows.load_column<unsigned int>("goals", df.get_column<unsigned int>("away_goals"));

    new_df.self_concat<ULDataFrame, std::string, bool, unsigned int>(away_rows);

    return new_df;
}

ULDataFrame add_intercept(ULDataFrame df) {
    df.load_column<unsigned int>("intercept", std::vector<unsigned int>(get_num_rows(df), 1));

    return df;
}

void print_variables(std::vector<std::string> names, std::vector<double> coefs) {
    std::copy(names.begin(), names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;
    std::copy(coefs.begin(), coefs.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
}

ULDataFrame get_test_data() {
    ULDataFrame df;
    df.load_data(std::vector<unsigned long>{1,2,3}, 
        std::make_pair("team", std::vector<std::string>{"Wolves", "Sunderland"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Wolves"}),
        std::make_pair("home", std::vector<bool>{true, false})
    );

    return df;
}

int main() {
    ULDataFrame train_df{add_intercept(transform_to_row_per_goals(get_data()))};

    PoissonRegressionTrainer trainer{};

    PoissonRegressionModelData model_data{trainer.get_poisson_regression_model_data(std::move(train_df), "goals")};

    PoissonRegressionModel model{model_data.x_col_names.size()};
    model.set_data(model_data.data);
    model.mle();
    print_variables(model_data.x_col_names, model.coef().vectorize());

    // transform_to_row_per_goals has to be able to convert test rows as well (without goals columns)
    ULDataFrame test_df{add_intercept(get_test_data())};
    
    for (Vector x : trainer.generate_x(test_df, model_data)) {
        double lambda = exp(model.predict(x));
        std::cout << "Expected goals: " << lambda << std::endl;
    }

    return 0;
}
