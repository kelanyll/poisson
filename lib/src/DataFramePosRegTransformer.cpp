#include "DataFramePosRegTransformer.hpp"

ULDataFrame DataFramePosRegTransformerImpl::one_hot_encode(ULDataFrame& df) {
    one_hot_encode_string(df);
    one_hot_encode_bool(df);

    return df;
}

std::vector<std::string> DataFramePosRegTransformerImpl::get_col_names(const ULDataFrame& df) {
    auto all_col_info{df.get_columns_info<unsigned int>()};
    std::vector<std::string> col_names{};
    std::transform(all_col_info.begin(), all_col_info.end(), std::back_inserter(col_names), 
        [](std::tuple<typename ULDataFrame::ColNameType,
            typename ULDataFrame::size_type, std::type_index> col_info) {
                return std::string(std::get<0>(col_info).c_str());
        }
    );

    return col_names;
}

std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> DataFramePosRegTransformerImpl::convert_to_poisson_regression_data(const ULDataFrame& df, const std::string& y_col_name, const std::vector<std::string>& x_col_names) {
    std::vector<BOOM::Ptr<BOOM::PoissonRegressionData>> data{};

    std::vector<const char*> x_col_names_c_str{};
    std::transform(x_col_names.begin(), x_col_names.end(), std::back_inserter(x_col_names_c_str), [](const std::string& col_name) {
        return col_name.c_str();
    });

    for (int i = 0; i < get_num_rows(df); i++) {
        data.push_back(BOOM::Ptr{new BOOM::PoissonRegressionData{
            df.get_row<unsigned int>(i, std::vector<const char*>{y_col_name.c_str()}).get_vector<unsigned int>()[0],
            df.get_row<unsigned int>(i, x_col_names_c_str).get_vector<unsigned int>()
        }});
    }

    return data;
}

ULDataFrame DataFramePosRegTransformerImpl::add_missing_cols(ULDataFrame df, std::vector<std::string> col_names) {
    for (std::string col_name : col_names) {
        if (!df.has_column(col_name.c_str())) {
            df.load_column<unsigned int>(col_name.c_str(), std::vector<unsigned int>(get_num_rows(df), 0));
        }
    }

    return df;
}

std::vector<std::vector<unsigned int>> DataFramePosRegTransformerImpl::get_row_vectors(ULDataFrame df, std::vector<std::string> col_names) {
    std::vector<std::vector<unsigned int>> rows{};

    std::vector<const char*> col_names_c_str{};
    std::transform(col_names.begin(), col_names.end(), std::back_inserter(col_names_c_str), [](std::string& col_name) {
        return col_name.c_str();
    });

    for (int i = 0; i < get_num_rows(df); i++) {
        rows.push_back(df.get_row<unsigned int>(i, col_names_c_str).get_vector<unsigned int>());
    }

    return rows;
}

void DataFramePosRegTransformerImpl::one_hot_encode_string(ULDataFrame& df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<std::string>()) {
        auto col_name = std::get<0>(col_info).c_str();
        auto col_data = df.get_column<std::string>(col_name);

        std::unordered_map<std::string, std::vector<unsigned int>> encoded_cols;

        for (auto val : df.get_col_unique_values<std::string>(col_name)) {
            encoded_cols[std::string{col_name} + "_" + val] = std::vector<unsigned int>(col_data.size(), 0);
        }

        for (int i = 0; i < col_data.size(); i++) {
            encoded_cols[std::string{col_name} + "_" + col_data[i]][i] = 1;
        }

        df.remove_column(col_name);

        for (const auto& pair : encoded_cols) {
            df.load_column<unsigned int>(pair.first.c_str(), std::move(pair.second));
        }
    }
}

void DataFramePosRegTransformerImpl::one_hot_encode_bool(ULDataFrame& df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<bool>()) {
        auto col_name = std::get<0>(col_info).c_str();
        auto col_data = df.get_column<bool>(col_name);

        std::vector<unsigned int> encoded_col(col_data.size());

        std::transform(col_data.begin(), col_data.end(), encoded_col.begin(),
                    [](bool value) { return value ? 1 : 0; });

        df.remove_column(col_name);
        df.load_column<unsigned int>(col_name, std::move(encoded_col));
    }
}
