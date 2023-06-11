#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "goals-predict.hpp"
#include "PoissonRegressionTrainer.hpp"

using ::testing::Return;

class MockDataFramePosRegTransformer : public DataFramePosRegTransformer {
public:
    MOCK_METHOD(ULDataFrame, one_hot_encode, (ULDataFrame df), (override));
    MOCK_METHOD(std::vector<std::string>, get_col_names,(ULDataFrame df), (override));
    MOCK_METHOD(std::vector<Ptr<PoissonRegressionData>>, convert_to_poisson_regression_data,(ULDataFrame df, std::string y_col_name, std::vector<std::string> x_col_names), (override));
};

TEST(GetPoissonRegressionModelData, FullTestCase) {
    MockDataFramePosRegTransformer mock_transform{};
    PoissonRegressionTrainer trainer{&mock_transform};

    ULDataFrame test_df;
    test_df.load_data(std::vector<unsigned long>{1,2,3,4,5,6}, 
        std::make_pair("team", std::vector<std::string>{"Wolves", "Sunderland","Chelsea", "Sunderland", "Chelsea", "Wolves"}),
        std::make_pair("opponent", std::vector<std::string>{"Sunderland", "Chelsea", "Wolves", "Wolves", "Sunderland", "Chelsea"}),
        std::make_pair("home", std::vector<bool>{true, true, true, false, false, false}),
        std::make_pair("goals", std::vector<unsigned int>{0, 1, 2, 2, 1, 0}),
        std::make_pair("intercept", std::vector<unsigned int>{1, 1, 1, 1, 1, 1})
    );

    ULDataFrame one_hot_encode_val;
    one_hot_encode_val.load_data(std::vector<unsigned long>{1,2,3,4,5,6},
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
    EXPECT_CALL(mock_transform, one_hot_encode).WillOnce(Return(one_hot_encode_val));

    std::vector<std::string> x_col_names{"team_Wolves", "team_Chelsea", "team_Sunderland", "opponent_Wolves",
        "opponent_Chelsea", "opponent_Sunderland", "home", "intercept"};
    std::vector<std::string> get_col_names_val{};
    std::copy(x_col_names.begin(), x_col_names.end(), std::back_inserter(get_col_names_val));
    get_col_names_val.push_back("goals");

    EXPECT_CALL(mock_transform, get_col_names).WillOnce(Return(get_col_names_val));

    std::vector<Ptr<PoissonRegressionData>> convert_to_poisson_regression_data_val{
        Ptr<PoissonRegressionData>{new PoissonRegressionData{0, std::vector<unsigned int>{1, 0, 0, 0, 0, 1, 1, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{1, std::vector<unsigned int>{0, 0, 1, 0, 1, 0, 1, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{2, std::vector<unsigned int>{0, 1, 0, 1, 0, 0, 1, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{2, std::vector<unsigned int>{0, 0, 1, 1, 0, 0, 0, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{1, std::vector<unsigned int>{0, 1, 0, 0, 0, 1, 0, 1}}},
        Ptr<PoissonRegressionData>{new PoissonRegressionData{0, std::vector<unsigned int>{1, 0, 0, 0, 1, 0, 0, 1}}}
    };
    EXPECT_CALL(mock_transform, convert_to_poisson_regression_data).WillOnce(Return(convert_to_poisson_regression_data_val));

    PoissonRegressionModelData actual = trainer.get_poisson_regression_model_data(test_df, "goals");

    PoissonRegressionModelData expected{convert_to_poisson_regression_data_val, x_col_names};

    EXPECT_EQ(actual, expected);
}