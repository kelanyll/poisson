#include "data-types.hpp"
#include "PoissonRegressionModel.hpp"
#include <DataFrame/DataFrame.h>
#include "PoissonRegressionModelData.hpp"
#include "goals-predict.hpp"

/**
 * This is compatible with DataFrame columns of type unsigned int,
 * bool and std::string.
 */
class PoissonRegressionTrainer {
    public:
        PoissonRegressionTrainer();
        PoissonRegressionTrainer(DataFramePosRegTransformer* transforms);
        PoissonRegressionModelData get_poisson_regression_model_data(ULDataFrame df, std::string y_col_name);
        std::vector<std::vector<unsigned int>> generate_x(ULDataFrame df, PoissonRegressionModelData model_data);
    private:
        DataFramePosRegTransformer* transforms;
};