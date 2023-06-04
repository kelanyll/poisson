#include "data-types.hpp"
#include "PoissonRegressionModel.hpp"
#include <DataFrame/DataFrame.h>

/**
 * This is compatible with DataFrame columns of type unsigned int,
 * bool and std::string.
 */
class PoissonRegressionTrainer {
    public:
        PoissonRegressionTrainer(ULDataFrame df, std::string y_col_name);
        std::vector<double> predict(ULDataFrame &df);
    private:
        PoissonRegressionModel model;
};