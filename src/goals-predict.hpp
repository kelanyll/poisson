#include "data-types.hpp"
#include "util.hpp"

/*
home, away, home_goals, away_goals
team, opponent, home, goals (resp) (custom transformation)
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

