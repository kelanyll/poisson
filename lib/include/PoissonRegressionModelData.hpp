#include <vector>
#include <string>
#include "Ptr.hpp"
#include "PoissonRegressionData.hpp"
#include "data-types.hpp"

class PoissonRegressionModelData {
public:
    PoissonRegressionModelData(std::vector<Ptr<PoissonRegressionData>> data_val, std::vector<std::string> x_col_names_val);
    const std::vector<Ptr<PoissonRegressionData>> data;
    const std::vector<std::string> x_col_names;
};

bool operator==(const PoissonRegressionModelData& model_data1, const PoissonRegressionModelData& model_data2);