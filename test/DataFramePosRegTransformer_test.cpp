#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include "gtest/gtest.h"
#include "Ptr.hpp"
#include "PoissonRegressionData.hpp"

#include "data-types.hpp"
#include "DataFramePosRegTransformer.hpp"
#include "test-utils.hpp"
#include "utils.hpp"

TEST(OneHotEncodeTest, FullTestCase) {
    DataFramePosRegTransformerImpl transforms;

    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6}, 
        std::make_pair("team", std::vector<std::string>{"Wolves", "Sunderland","Chelsea", "Sunderland", "Chelsea", "Wolves"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Chelsea", "Wolves", "Wolves", "Sunderland", "Chelsea"}),
        std::make_pair("home", std::vector<bool>{true, true, true, false, false, false}),
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0})
    );

    transforms.one_hot_encode(test_df);

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

    auto is_equal = check_df_equal<std::string, bool, unsigned int>(test_df, expected_df);
    EXPECT_TRUE(is_equal);
}

TEST(GetColNamesTest, FullTestCase) {
    DataFramePosRegTransformerImpl transforms;

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

    std::vector<std::string> actual{transforms.get_col_names(test_df)};

    std::vector<std::string> expected{"goals", "team_Wolves", "team_Chelsea", "team_Sunderland", "opponent_Wolves", 
        "opponent_Chelsea", "opponent_Sunderland", "home", "intercept"};
    
    EXPECT_EQ(actual, expected);
}

TEST(ConvertToPoissonRegressionDataTest, FullTestCase) {
    DataFramePosRegTransformerImpl transforms;

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

    std::vector<std::string> x_col_names{"team_Wolves", "team_Chelsea", "team_Sunderland", "opponent_Wolves", 
        "opponent_Chelsea", "opponent_Sunderland", "home", "intercept"};

    using PoissonRegressionData = BOOM::PoissonRegressionData;
    using PoissonRegressionDataPtr = BOOM::Ptr<PoissonRegressionData>;

    std::vector<PoissonRegressionDataPtr> actual{transforms.convert_to_poisson_regression_data(test_df, "goals", x_col_names)};

    std::vector<PoissonRegressionDataPtr> expected{
        PoissonRegressionDataPtr{new PoissonRegressionData{0, std::vector<unsigned int>{1, 0, 0, 0, 0, 1, 1, 1}}},
        PoissonRegressionDataPtr{new PoissonRegressionData{1, std::vector<unsigned int>{0, 0, 1, 0, 1, 0, 1, 1}}},
        PoissonRegressionDataPtr{new PoissonRegressionData{2, std::vector<unsigned int>{0, 1, 0, 1, 0, 0, 1, 1}}},
        PoissonRegressionDataPtr{new PoissonRegressionData{2, std::vector<unsigned int>{0, 0, 1, 1, 0, 0, 0, 1}}},
        PoissonRegressionDataPtr{new PoissonRegressionData{1, std::vector<unsigned int>{0, 1, 0, 0, 0, 1, 0, 1}}},
        PoissonRegressionDataPtr{new PoissonRegressionData{0, std::vector<unsigned int>{1, 0, 0, 0, 1, 0, 0, 1}}}
    };

    bool is_equal{ptr_poisson_regression_data_equal(actual, expected)};
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

TEST(AddMissingColsTest, FullTestCase) {
    DataFramePosRegTransformerImpl transforms;

    ULDataFrame test_df{};
    test_df.load_data(std::vector<unsigned long>{1,2}, 
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 0}),
        std::make_pair("intercept", std::vector<unsigned int>{1, 1})
    );

    std::vector<std::string> col_names{"team_Wolves", "team_Chelsea", "team_Sunderland", "opponent_Wolves",
    "opponent_Chelsea", "opponent_Sunderland", "home", "intercept"};

    transforms.add_missing_cols(test_df, col_names);
    
    ULDataFrame expected_df{};
    expected_df.load_data(std::vector<unsigned long>{1,2}, 
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 0}),
        std::make_pair("intercept", std::vector<unsigned int>{1, 1}),
        std::make_pair("team_Chelsea", std::vector<unsigned int>{0, 0}),
        std::make_pair("opponent_Chelsea", std::vector<unsigned int>{0, 0})
    );

    auto is_equal = check_df_equal<unsigned int>(test_df, expected_df);
    EXPECT_TRUE(is_equal);
}

TEST(GetRowVectorsTest, FullTestCase) {
    DataFramePosRegTransformerImpl transforms;

    ULDataFrame test_df{};
    test_df.load_data(std::vector<unsigned long>{1,2}, 
        std::make_pair("team_Wolves", std::vector<unsigned int>{1, 0}),
        std::make_pair("team_Sunderland", std::vector<unsigned int>{0, 1}),
        std::make_pair("opponent_Wolves", std::vector<unsigned int>{0, 1}),
        std::make_pair("opponent_Sunderland", std::vector<unsigned int>{1, 0}),
        std::make_pair("home", std::vector<unsigned int>{1, 0}),
        std::make_pair("intercept", std::vector<unsigned int>{1, 1}),
        std::make_pair("team_Chelsea", std::vector<unsigned int>{0, 0}),
        std::make_pair("opponent_Chelsea", std::vector<unsigned int>{0, 0})
    );

    std::vector<std::string> col_names{"team_Wolves", "team_Chelsea", "team_Sunderland", "opponent_Wolves",
    "opponent_Chelsea", "opponent_Sunderland", "home", "intercept"};

    std::vector<std::vector<unsigned int>> actual{transforms.get_row_vectors(test_df, col_names)};
    
    std::vector<std::vector<unsigned int>> expected{
        std::vector<unsigned int>{1, 0, 0, 0, 0, 1, 1, 1},
        std::vector<unsigned int>{0, 0, 1, 1, 0, 0, 0, 1}
    };

    EXPECT_EQ(actual, expected);
}