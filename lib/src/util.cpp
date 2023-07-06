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

/*
HomeTeam, AwayTeam, FTHG, FTAG -> team, opponent, home, goals
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