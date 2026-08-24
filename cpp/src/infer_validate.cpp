// Verifies the Phase 5 exit criterion: C++ inference output matches the
// Python model within float tolerance. Loads reference_predictions.csv
// (written by python/training/train.py --export-dir), re-runs each row
// through the branchless TreeModel, and diffs against the Python-computed
// probability.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "spoofwatch/tree_model.hpp"

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

double parse_double(const std::string& s) {
    return std::stod(s);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.bin> <reference_predictions.csv> [tolerance=1e-6]\n", argv[0]);
        std::fprintf(stderr,
            "Cross-validates the branchless C++ TreeModel against Python-computed\n"
            "probabilities exported by `python -m python.training.train --export-dir ...`.\n");
        return 1;
    }

    std::string model_path = argv[1];
    std::string reference_csv = argv[2];
    double tolerance = (argc >= 4) ? std::stod(argv[3]) : 1e-6;

    std::ifstream in(reference_csv);
    if (!in.is_open()) {
        std::fprintf(stderr, "Error: could not open reference file: %s\n", reference_csv.c_str());
        return 1;
    }

    try {
        spoofwatch::TreeModel model(model_path);

        std::string header_line;
        if (!std::getline(in, header_line)) {
            throw std::runtime_error("reference file is empty");
        }
        std::vector<std::string> header = split_csv_line(header_line);
        // Expected columns: order_id, <feature columns...>, proba, label.
        // Feature columns are everything between order_id and proba.
        size_t feature_start = 1;
        size_t feature_end = header.size();
        for (size_t i = 0; i < header.size(); ++i) {
            if (header[i] == "proba") {
                feature_end = i;
                break;
            }
        }
        size_t num_features = feature_end - feature_start;
        if (num_features != model.num_features()) {
            throw std::runtime_error(
                "reference file has " + std::to_string(num_features) +
                " feature columns but model expects " + std::to_string(model.num_features()));
        }

        size_t rows = 0;
        double max_abs_diff = 0.0;
        double sum_abs_diff = 0.0;
        size_t mismatches = 0;

        std::string line;
        std::vector<double> features(num_features);
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> fields = split_csv_line(line);
            for (size_t i = 0; i < num_features; ++i) {
                features[i] = parse_double(fields[feature_start + i]);
            }
            double expected_proba = parse_double(fields[feature_end]);

            double actual_proba = model.predict_proba(features.data(), num_features);
            double diff = std::fabs(actual_proba - expected_proba);
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_abs_diff += diff;
            if (diff > tolerance) {
                ++mismatches;
            }
            ++rows;
        }

        std::printf("Compared %zu rows (%zu trees, %zu features) against Python predictions.\n",
                    rows, model.num_trees(), model.num_features());
        std::printf("  max abs diff:  %.3e\n", max_abs_diff);
        std::printf("  mean abs diff: %.3e\n", rows ? sum_abs_diff / rows : 0.0);
        std::printf("  rows exceeding tolerance %.3e: %zu/%zu\n", tolerance, mismatches, rows);

        if (mismatches > 0) {
            std::fprintf(stderr, "FAIL: C++ inference diverges from the Python model.\n");
            return 1;
        }
        std::printf("PASS: C++ inference matches the Python model within tolerance.\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
