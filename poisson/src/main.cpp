#include <iostream>
#include <sstream>
#include "PoissonRegressionModel.hpp"
#include <DataFrame/DataFrame.h>
#include "PoissonRegressionTrainer.hpp"
#include "data-types.hpp"
#include "util.hpp"
#include <unistd.h>

std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> col_names{};
    std::vector<std::vector<std::string>> data{};

    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return data;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            col_names.push_back(value);
            data.push_back({}); // Each value starts a new column vector
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        size_t col = 0;
        while (std::getline(ss, value, ',')) {
            data[col++].push_back(value);
        }
    }

    file.close();
    return data;
}

ULDataFrame get_data() {
    auto data = readCSV("prem-11-12.csv");

    auto convert_vec_str_to_uint = [](std::vector<std::string>& str_vec) {
        std::vector<unsigned int> uint_vec{};
        std::transform(str_vec.begin(), str_vec.end(), std::back_inserter(uint_vec), [](std::string str) {
            return std::stoul(str);
        });
        return uint_vec;
    };

    ULDataFrame df;
    std::vector<unsigned long> index(data[0].size());
    std::iota(index.begin(), index.end(), 1);
    df.load_data(std::move(index),  
        std::make_pair("HomeTeam", data[2]),
        std::make_pair("AwayTeam", data[3]),
        std::make_pair("FTHG", convert_vec_str_to_uint(data[4])),
        std::make_pair("FTAG", convert_vec_str_to_uint(data[5]))
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

    new_df.load_column<std::string>("team", df.get_column<std::string>("HomeTeam"));
    new_df.load_column<std::string>("opponent", df.get_column<std::string>("AwayTeam"));
    new_df.load_column<bool>("home", std::vector<bool>(get_num_rows(df), true));
    new_df.load_column<unsigned int>("goals", df.get_column<unsigned int>("FTHG"));

    ULDataFrame away_rows{};
    
    int num_rows = get_num_rows(df);
    std::vector<unsigned long> away_rows_idx(num_rows);
    std::iota(away_rows_idx.begin(), away_rows_idx.end(), idx.back() + 1);
    away_rows.load_index(away_rows_idx.begin(), away_rows_idx.end());

    away_rows.load_column<std::string>("team", df.get_column<std::string>("AwayTeam"));
    away_rows.load_column<std::string>("opponent", df.get_column<std::string>("HomeTeam"));
    away_rows.load_column<bool>("home", std::vector<bool>(get_num_rows(df), false));
    away_rows.load_column<unsigned int>("goals", df.get_column<unsigned int>("FTAG"));

    new_df.self_concat<ULDataFrame, std::string, bool, unsigned int>(away_rows);

    return new_df;
}

ULDataFrame add_intercept(ULDataFrame df) {
    df.load_column<unsigned int>("intercept", std::vector<unsigned int>(get_num_rows(df), 1));

    return df;
}

void print_variables(std::vector<std::string> names, std::vector<double> coefs) {
    for (int i = 0; i < names.size(); i++) {
        std::cout << names[i] << ": " << coefs[i] << std::endl;
    }
}

ULDataFrame get_test_data() {
    ULDataFrame df;
    df.load_data(std::vector<unsigned long>{1,2}, 
        std::make_pair("team", std::vector<std::string>{"Aston Villa", "Sunderland"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Aston Villa"}),
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

    ULDataFrame test_df{add_intercept(get_test_data())};
    
    for (Vector x : trainer.generate_x(test_df, model_data)) {
        double lambda = exp(model.predict(x));
        std::cout << "Expected goals: " << lambda << std::endl;
    }

    return 0;
}
