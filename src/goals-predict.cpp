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
    new_df.load_column<int>("goals", df.get_column<int>("home_goals"));

    ULDataFrame away_rows{};
    
    int num_rows = get_num_rows(df);
    std::vector<unsigned long> away_rows_idx(num_rows);
    std::iota(away_rows_idx.begin(), away_rows_idx.end(), idx.back() + 1);
    away_rows.load_index(away_rows_idx.begin(), away_rows_idx.end());

    away_rows.load_column<std::string>("team", df.get_column<std::string>("away"));
    away_rows.load_column<std::string>("opponent", df.get_column<std::string>("home"));
    away_rows.load_column<bool>("home", std::vector<bool>(get_num_rows(df), false));
    away_rows.load_column<int>("goals", df.get_column<int>("away_goals"));

    new_df.self_concat<ULDataFrame, std::string, bool, int>(away_rows);

    return new_df;
}

ULDataFrame one_hot_encode(ULDataFrame df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<std::string>()) {
        std::unordered_map<std::string, std::vector<unsigned int>> encoded_cols;

        auto col_name = std::get<0>(col_info).c_str();
        auto col_data = df.get_column<std::string>(col_name);

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

    // same deal but bools

    return df;
}