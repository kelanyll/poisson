#include "goals-predict.hpp"
#include <cstdio>

#include "data-types.hpp"
#include "util.hpp"

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

ULDataFrame one_hot_encode(ULDataFrame df) {
    one_hot_encode_string(df);
    one_hot_encode_bool(df);

    return df;
}

void one_hot_encode_string(ULDataFrame& df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<std::string>()) {
        auto col_name = std::get<0>(col_info).c_str();
        auto col_data = df.get_column<std::string>(col_name);

        std::unordered_map<std::string, std::vector<unsigned int>> encoded_cols;

        for (auto val : df.get_col_unique_values<std::string>(col_name)) {
            encoded_cols[std::string{col_name} + "_" + val] = std::vector<unsigned int>(col_data.size(), 0);
        }

        for (int i = 0; i < col_data.size(); i++) {
            encoded_cols[std::string{col_name} + "_" + col_data[i]][i] = 1;
        }

        df.remove_column(col_name);

        for (const auto& pair : encoded_cols) {
            df.load_column<unsigned int>(pair.first.c_str(), std::move(pair.second));
        }
    }
}

void one_hot_encode_bool(ULDataFrame& df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<bool>()) {
        auto col_name = std::get<0>(col_info).c_str();
        auto col_data = df.get_column<bool>(col_name);

        std::vector<unsigned int> encoded_col(col_data.size());

        std::transform(col_data.begin(), col_data.end(), encoded_col.begin(),
                    [](bool value) { return value ? 1 : 0; });

        df.remove_column(col_name);
        df.load_column<unsigned int>(col_name, std::move(encoded_col));
    }
}

void add_intercept(ULDataFrame& df) {
    df.load_column<unsigned int>("intercept", std::vector<unsigned int>(get_num_rows(df), 1));
}

std::vector<Ptr<PoissonRegressionData>> convert_to_poisson_regression_data(ULDataFrame df, std::string y_col_name) {
    int y_col_idx = df.col_name_to_idx(y_col_name.c_str());

    std::vector<Ptr<PoissonRegressionData>> data{};
    for (int i = 0; i < get_num_rows(df); i++) {
        std::vector<unsigned int> row{df.get_row<unsigned int>(i).get_vector<unsigned int>()};
        unsigned int y_val = row[y_col_idx];
        row.erase(row.begin() + y_col_idx);

        data.push_back(Ptr{new PoissonRegressionData{y_val, row}});
    }

    return data;
}