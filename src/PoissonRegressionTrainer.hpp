#include "data-types.hpp"
#include "PoissonRegressionModel.hpp"
#include <DataFrame/DataFrame.h>

// goals of this is to allow training on a dataframe
class PoissonRegressionTrainer {
    private:
        PoissonRegressionModel model;

        BOOM::ConstVectorView dataframe_row_to_boom_vector(hmdf::HeteroVector row);
    public:
        PoissonRegressionTrainer(ULDataFrame df, std::string y_col_name);
        std::vector<double> predict(ULDataFrame &df);
};
/*
home, away, home_goals, away_goals
team, opponent, home, goals (resp) (custom transformation)
x_team, y_team, x_opponent, y_opponent, home (ohe)
*/