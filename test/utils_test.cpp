#include <vector>
#include <string>
#include <utility>

#include "gtest/gtest.h"

#include "data-types.hpp"
#include "utils.hpp"
#include "test-utils.hpp"

TEST(TransformToRowPerGoalsTest, FullTestCase) {
    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3}, 
        std::make_pair("HomeTeam", std::vector<std::string>{"Wolves", "Sunderland","Chelsea"}),
        std::make_pair("AwayTeam", std::vector<std::string>{"Sunderland", "Chelsea","Wolves"}),
        std::make_pair("FTHG", std::vector<unsigned int>{0, 1, 2}),
        std::make_pair("FTAG", std::vector<unsigned int>{2, 1, 0})
    );

    ULDataFrame actual_df = transform_to_row_per_goals(std::move(test_df));

    ULDataFrame expected_df;
    expected_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6}, 
        std::make_pair("team", std::vector<std::string>{"Wolves", "Sunderland","Chelsea", "Sunderland", "Chelsea", "Wolves"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Chelsea", "Wolves", "Wolves", "Sunderland", "Chelsea"}),
        std::make_pair("home", std::vector<bool>{true, true, true, false, false, false}),
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0})
    );

    auto is_equal = check_df_equal<std::string, bool, unsigned int>(actual_df, expected_df);
    EXPECT_TRUE(is_equal);
}

TEST(AddInterceptTest, FullTestCase) {
    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6},
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0, 0, 0, 0, 1}),
        std::make_pair("team_Chelsea", std::vector<unsigned int>{0, 0, 1, 0, 1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1, 0, 1, 0, 0}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 0, 1, 1, 0, 0}),
        std::make_pair("opponent_Chelsea", std::vector<unsigned int>{0, 1, 0, 0, 0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0, 0, 0, 1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 1, 1, 0, 0, 0}),
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0})
    );

    auto actual_df{add_intercept(std::move(test_df))};

    ULDataFrame expected_df;
    expected_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6},
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0, 0, 0, 0, 1}),
        std::make_pair("team_Chelsea", std::vector<unsigned int>{0, 0, 1, 0, 1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1, 0, 1, 0, 0}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 0, 1, 1, 0, 0}),
        std::make_pair("opponent_Chelsea", std::vector<unsigned int>{0, 1, 0, 0, 0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0, 0, 0, 1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 1, 1, 0, 0, 0}),
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0}),
        std::make_pair("intercept", std::vector<unsigned int>{1, 1, 1, 1, 1, 1})
    );

    auto is_equal = check_df_equal<unsigned int>(actual_df, expected_df);
    EXPECT_TRUE(is_equal);
}