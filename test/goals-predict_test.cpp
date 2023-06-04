#include "goals-predict.hpp"
#include "gtest/gtest.h"
#include "util.hpp"
#include "test-util.hpp"

TEST(TransformToRowPerGoalsTest, FullTestCase) {
    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3}, 
        std::make_pair("home", std::vector<std::string>{"Wolves", "Sunderland","Chelsea"}),
        std::make_pair("away", std::vector<std::string>{"Sunderland", "Chelsea","Wolves"}),
        std::make_pair("home_goals", std::vector<unsigned int>{0, 1, 2}),
        std::make_pair("away_goals", std::vector<unsigned int>{2, 1, 0})
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

TEST(OneHotEncodeTest, FullTestCase) {
    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6}, 
        std::make_pair("team", std::vector<std::string>{"Wolves", "Sunderland","Chelsea", "Sunderland", "Chelsea", "Wolves"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Chelsea", "Wolves", "Wolves", "Sunderland", "Chelsea"}),
        std::make_pair("home", std::vector<bool>{true, true, true, false, false, false}),
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0})
    );

    ULDataFrame actual_df = one_hot_encode(std::move(test_df));

    ULDataFrame expected_df;
    expected_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6},
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0, 0, 0, 0, 1}),
        std::make_pair("team_Chelsea", std::vector<unsigned int>{0, 0, 1, 0, 1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1, 0, 1, 0, 0}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 0, 1, 1, 0, 0}),
        std::make_pair("opponent_Chelsea", std::vector<unsigned int>{0, 1, 0, 0, 0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0, 0, 0, 1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 1, 1, 0, 0, 0}),
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

    add_intercept(test_df);

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

    auto is_equal = check_df_equal<unsigned int>(test_df, expected_df);
    EXPECT_TRUE(is_equal);
}

TEST(ConvertToPoissonRegressionDataTest, FullTestCase) {
    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6},
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0}),
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0, 0, 0, 0, 1}),
        std::make_pair("team_Chelsea", std::vector<unsigned int>{0, 0, 1, 0, 1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1, 0, 1, 0, 0}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 0, 1, 1, 0, 0}),
        std::make_pair("opponent_Chelsea", std::vector<unsigned int>{0, 1, 0, 0, 0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0, 0, 0, 1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 1, 1, 0, 0, 0}),
        std::make_pair("intercept", std::vector<unsigned int>{1, 1, 1, 1, 1, 1})
    );

    std::vector<Ptr<PoissonRegressionData>> actual{convert_to_poisson_regression_data(test_df, "goals")};

    std::vector<Ptr<PoissonRegressionData>> expected{
        Ptr<PoissonRegressionData>{new PoissonRegressionData{0, std::vector<unsigned int>{1, 0, 0, 0, 0, 1, 1, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{1, std::vector<unsigned int>{0, 0, 1, 0, 1, 0, 1, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{2, std::vector<unsigned int>{0, 1, 0, 1, 0, 0, 1, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{2, std::vector<unsigned int>{0, 0, 1, 1, 0, 0, 0, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{1, std::vector<unsigned int>{0, 1, 0, 0, 0, 1, 0, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{0, std::vector<unsigned int>{1, 0, 0, 0, 1, 0, 0, 1}}}
    };

    bool is_equal = true;
    for (int i = 0; i < actual.size(); i++) {
        bool y_equal = actual[i]->y() == expected[i]->y();
        bool x_equal = actual[i]->x() == expected[i]->x();
        if (!(y_equal && x_equal)) {
            is_equal = false;
            break;
        }
    }

    if (!is_equal) {
        std::cout << "Actual:" << std::endl;
        for (int i = 0; i < actual.size(); i++) {
            std::cout << actual[i] << std::endl;
        }

        std::cout << "Expected:" << std::endl;
        for (int i = 0; i < expected.size(); i++) {
            std::cout << expected[i] << std::endl;
        }
    }

    EXPECT_TRUE(is_equal);
}