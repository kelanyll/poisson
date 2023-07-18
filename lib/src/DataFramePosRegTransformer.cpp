#include "DataFramePosRegTransformer.hpp"

void DataFramePosRegTransformerImpl::one_hot_encode(ULDataFrame& df) {
    one_hot_encode_string(df);
    one_hot_encode_bool(df);
}

std::vector<std::string> DataFramePosRegTransformerImpl::get_col_names(const ULDataFrame& df) {
    std::vector<std::tuple<typename ULDataFrame::ColNameType,
            typename ULDataFrame::size_type, std::type_index>> all_col_info{df.get_columns_info<unsigned int>()};
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

    std::vector<const char*> x_col_names_c_str{convert_to_c_str_vec(x_col_names)};

    for (int i = 0; i < get_num_rows(df); i++) {
        data.push_back(BOOM::Ptr{new BOOM::PoissonRegressionData{
            df.get_row<unsigned int>(i, std::vector<const char*>{y_col_name.c_str()}).get_vector<unsigned int>()[0],
            df.get_row<unsigned int>(i, x_col_names_c_str).get_vector<unsigned int>()
        }});
    }

    return data;
}

void DataFramePosRegTransformerImpl::add_missing_cols(ULDataFrame& df, const std::vector<std::string>& col_names) {
    for (std::string col_name : col_names) {
        if (!df.has_column(col_name.c_str())) {
            df.load_column<unsigned int>(col_name.c_str(), std::vector<unsigned int>(get_num_rows(df), 0));
        }
    }
}

std::vector<std::vector<unsigned int>> DataFramePosRegTransformerImpl::get_row_vectors(const ULDataFrame& df, const std::vector<std::string>& col_names) {
    std::vector<std::vector<unsigned int>> rows{};

    std::vector<const char*> col_names_c_str{convert_to_c_str_vec(col_names)};

    for (int i = 0; i < get_num_rows(df); i++) {
        rows.push_back(df.get_row<unsigned int>(i, col_names_c_str).get_vector<unsigned int>());
    }

    return rows;
}

void DataFramePosRegTransformerImpl::one_hot_encode_string(ULDataFrame& df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<std::string>()) {
        auto col_name{std::get<0>(col_info).c_str()};
        std::vector<std::string> col_data{df.get_column<std::string>(col_name)};

        std::unordered_map<std::string, std::vector<unsigned int>> encoded_cols{};

        for (int i = 0; i < col_data.size(); i++) {
            auto encoded_col_name{std::string{col_name} + "_" + col_data[i]};
            if (!encoded_cols.contains(encoded_col_name)) {
                encoded_cols[encoded_col_name] = std::vector<unsigned int>(col_data.size(), 0);
            }
            encoded_cols[encoded_col_name][i] = 1;
        }

        df.remove_column(col_name);

        for (const std::pair<std::string, std::vector<unsigned int>>& pair : encoded_cols) {
            df.load_column<unsigned int>(pair.first.c_str(), std::move(pair.second));
        }
    }
}

void DataFramePosRegTransformerImpl::one_hot_encode_bool(ULDataFrame& df) {
    for (std::tuple<ULDataFrame::ColNameType,
                    ULDataFrame::size_type,
                    std::type_index> col_info : df.get_columns_info<bool>()) {
        auto col_name{std::get<0>(col_info).c_str()};
        std::vector<bool> col_data{df.get_column<bool>(col_name)};

        std::vector<unsigned int> encoded_col(col_data.size());

        std::transform(col_data.begin(), col_data.end(), encoded_col.begin(),
                    [](bool value) { return value ? 1 : 0; });

        df.remove_column(col_name);
        df.load_column<unsigned int>(col_name, std::move(encoded_col));
    }
}
