#include "csv.hpp"

std::vector<std::vector<std::string>> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> col_names{};
    std::vector<std::vector<std::string>> data{};

    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return data;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            col_names.push_back(value);
            data.push_back({}); // Each value starts a new column vector
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        size_t col = 0;
        while (std::getline(ss, value, ',')) {
            data[col++].push_back(value);
        }
    }

    file.close();
    return data;
}