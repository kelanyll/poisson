#include "goals-predict.hpp"
#include "gtest/gtest.h"

TEST(TransformToRowPerGoalsTest, FullTestCase) {
    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3}, 
        std::make_pair("home", std::vector<std::string>{"Wolves", "Sunderland","Chelsea"}),
        std::make_pair("away", std::vector<std::string>{"Sunderland", "Chelsea","Wolves"}),
        std::make_pair("home_goals", std::vector<int>{0, 1, 2}),
        std::make_pair("away_goals", std::vector<int>{2, 1, 0})
    );

    ULDataFrame actual_df = transform_to_row_per_goals(std::move(test_df));

    ULDataFrame expected_df;
    expected_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6}, 
        std::make_pair("team", std::vector<std::string>{"Wolves", "Sunderland","Chelsea", "Sunderland", "Chelsea", "Wolves"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Chelsea", "Wolves", "Wolves", "Sunderland", "Chelsea"}),
        std::make_pair("home", std::vector<bool>{true, true, true, false, false, false}),
        std::make_pair("goals", std::vector<int>{0, 1, 2, 2, 1, 0})
    );

    EXPECT_TRUE(actual_df.is_equal(expected_df));
}