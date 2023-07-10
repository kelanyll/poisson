#include <iostream>
#include <chrono>

#include "PoissonRegressionModel.hpp"

#include "data-types.hpp"
#include "csv.hpp"
#include "utils.hpp"
#include "PoissonRegressionTrainer.hpp"

ULDataFrame get_data() {
    auto data = read_csv("prem-11-12.csv");

    auto convert_vec_str_to_uint = [](std::vector<std::string>& str_vec) {
        std::vector<unsigned int> uint_vec{};
        std::transform(str_vec.begin(), str_vec.end(), std::back_inserter(uint_vec), [](std::string str) {
            return std::stoul(str);
        });
        return uint_vec;
    };

    std::vector<unsigned long> index(data[0].size());
    std::iota(index.begin(), index.end(), 1);

    ULDataFrame df;
    df.load_data(std::move(index),  
        std::make_pair("HomeTeam", data[2]),
        std::make_pair("AwayTeam", data[3]),
        std::make_pair("FTHG", convert_vec_str_to_uint(data[4])),
        std::make_pair("FTAG", convert_vec_str_to_uint(data[5]))
    );

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
    auto start = std::chrono::high_resolution_clock::now();

    ULDataFrame train_df{add_intercept(transform_to_row_per_goals(get_data()))};

    PoissonRegressionTrainer trainer{};

    PoissonRegressionModelData model_data{trainer.get_poisson_regression_model_data(std::move(train_df), "goals")};

    BOOM::PoissonRegressionModel model{static_cast<int>(model_data.x_col_names.size())};
    model.set_data(model_data.data);
    model.mle();
    print_variables(model_data.x_col_names, model.coef().vectorize());

    ULDataFrame test_df{add_intercept(get_test_data())};
    
    for (BOOM::Vector x : trainer.generate_x(test_df, model_data)) {
        double lambda = exp(model.predict(x));
        std::cout << "Expected goals: " << lambda << std::endl;
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Print the duration
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
