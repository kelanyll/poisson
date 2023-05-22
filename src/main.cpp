#include <iostream>
#include <sstream>
#include "PoissonRegressionModel.hpp"
#include <DataFrame/DataFrame.h>
#include "PoissonRegressionTrainer.hpp"
#include "goals-predict.hpp"

using namespace std;
using namespace BOOM;
using namespace hmdf;

struct Dataset {
    std::vector<Ptr<PoissonRegressionData>> dataset;
    int num_dims;

    Dataset(std::vector<Ptr<PoissonRegressionData>> dataset_val, int num_dims_val) : dataset{dataset_val}, num_dims{num_dims_val} {};
};

struct MyData {
    int y;
    std::vector<double> x;

    MyData(int y_val, std::vector<double> x_val) : y{y_val}, x{x_val} {}
};


Dataset get_dataset();
void print_coefs(PoissonRegressionModel model);

ULDataFrame get_data() {
    ULDataFrame df;
    df.load_data(std::vector<unsigned long>{1,2,3}, 
        std::make_pair("home", std::vector<std::string>{"Wolves", "Sunderland","Chelsea"}),
        std::make_pair("away", std::vector<std::string>{"Sunderland", "Chelsea","Wolves"}),
        std::make_pair("home_goals", std::vector<int>{0, 1, 2}),
        std::make_pair("away_goals", std::vector<int>{2, 1, 0})
    );

    return df;
}

int main() {
    ULDataFrame train_df{transform_to_row_per_goals(get_data())};

    // const auto &col_ref = data_frame.get_column<std::string>("home");
    // cout << col_ref[1] << col_ref[2] << endl;
    // data_frame.write<std::ostream, std::string, std::string, int, int>(cout);
    // std::pair shape = data_frame.shape();
    // cout << "size: " << std::get<0>(shape) << "x" << std::get<1>(shape) << endl;

    PoissonRegressionTrainer model = PoissonRegressionTrainer{std::move(train_df), "goals"};
    
    ULDataFrame test_df{};
    for (double y : model.predict(test_df)) {
        cout << y << ",";
    }
    cout << endl;

    return 0;
}

/*
home, away, home_goals, away_goals -> (ohe) team_.., opponent_.., goals, home
function that reads the csv
function to convert dataframe into expected format (reuse this to convert predicts into the same format)
constructor on Dataset that takes a dataframe and converts it to PoissonRegression
poissonregressionmodel doesn't care what the columns are called but we need it for ease of use
- when we predict i want to be able to pass (team: wolves, opponent: sunderland, home: true)
- then that converts it into arsenal_team: false, ..., wolves_team: true,..., home: true
- so maybe we create an object that takes a dataframe and then trains the model based off it
*/
Dataset get_dataset() {
    int num_dim = 2;
    std::vector<MyData> data{MyData{10, std::vector<double>{1,2}}, MyData{7, std::vector<double>{1,3}}, MyData{5, std::vector<double>{1,4}}, MyData{3,std::vector<double>{1,5}}, MyData{2, std::vector<double>{1,6}}};

    std::vector<Ptr<PoissonRegressionData>> ptrs{};
    std::transform(data.begin(), data.end(), std::back_inserter(ptrs), [](MyData datum) -> Ptr<PoissonRegressionData> {
        datum.x.push_back(1); // intercept
        return Ptr<PoissonRegressionData>{new PoissonRegressionData{datum.y, Vector{datum.x}}};
    });

    return Dataset{ptrs, num_dim + 1};
}

void print_coefs(PoissonRegressionModel model) {
    stringstream oss;
    model.coef().vectorize().write(oss, true);
    cout << "Parameters: " << oss.str();
}
