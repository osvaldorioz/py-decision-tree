#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` decision_tree.cpp -o decision_tree`python3.12-config --extension-suffix`
//c++ -O3 -Wall -shared -std=c++120 -fPIC `python3.12 -m pybind11 --includes` decision_tree.cpp -o decision_tree`python3.12-config --extension-suffix`

namespace py = pybind11;

class DecisionTree {
private:
    struct Node {
        std::string feature;
        double threshold;
        std::map<double, double> leaf_values; // Map de etiquetas a frecuencias
        Node* left = nullptr;
        Node* right = nullptr;

        ~Node() {
            delete left;
            delete right;
        }
    };

    Node* root = nullptr;

    // Método para calcular la ganancia de información (simplificado)
    double calculate_gini(const std::vector<double>& labels) {
        std::map<double, int> label_counts;
        for (double label : labels) {
            label_counts[label]++;
        }

        double impurity = 1.0;
        for (const auto& [label, count] : label_counts) {
            double prob = static_cast<double>(count) / labels.size();
            impurity -= prob * prob;
        }
        return impurity;
    }

    // División del dataset en función de un umbral
    void split_dataset(const std::vector<std::vector<double>>& data,
                       const std::vector<double>& labels,
                       int feature_idx,
                       double threshold,
                       std::vector<std::vector<double>>& left_data,
                       std::vector<std::vector<double>>& right_data,
                       std::vector<double>& left_labels,
                       std::vector<double>& right_labels) {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i][feature_idx] <= threshold) {
                left_data.push_back(data[i]);
                left_labels.push_back(labels[i]);
            } else {
                right_data.push_back(data[i]);
                right_labels.push_back(labels[i]);
            }
        }
    }

    // Entrenamiento recursivo del árbol
    Node* build_tree(const std::vector<std::vector<double>>& data,
                     const std::vector<double>& labels,
                     int depth) {
        if (data.empty() || depth <= 0) {
            return nullptr;
        }

        // Condición de parada: si todas las etiquetas son iguales
        if (std::adjacent_find(labels.begin(), labels.end(), std::not_equal_to<>()) == labels.end()) {
            Node* leaf = new Node();
            leaf->leaf_values[labels[0]] = 1.0; // Una etiqueta única
            return leaf;
        }

        // Encuentra la mejor división
        int best_feature = -1;
        double best_threshold = 0.0;
        double best_gini = std::numeric_limits<double>::max();
        std::vector<std::vector<double>> best_left_data, best_right_data;
        std::vector<double> best_left_labels, best_right_labels;

        for (long unsigned int feature_idx = 0; feature_idx < data[0].size(); ++feature_idx) {
            for (const auto& row : data) {
                double threshold = row[feature_idx];

                std::vector<std::vector<double>> left_data, right_data;
                std::vector<double> left_labels, right_labels;
                split_dataset(data, labels, feature_idx, threshold,
                              left_data, right_data, left_labels, right_labels);

                double gini_left = calculate_gini(left_labels);
                double gini_right = calculate_gini(right_labels);
                double gini = (gini_left * left_labels.size() + gini_right * right_labels.size()) / labels.size();

                if (gini < best_gini) {
                    best_gini = gini;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                    best_left_data = left_data;
                    best_right_data = right_data;
                    best_left_labels = left_labels;
                    best_right_labels = right_labels;
                }
            }
        }

        if (best_feature == -1) {
            return nullptr;
        }

        Node* node = new Node();
        node->feature = "Feature_" + std::to_string(best_feature);
        node->threshold = best_threshold;
        node->left = build_tree(best_left_data, best_left_labels, depth - 1);
        node->right = build_tree(best_right_data, best_right_labels, depth - 1);
        return node;
    }

public:
    // Entrenar el árbol
    void fit(const std::vector<std::vector<double>>& data,
             const std::vector<double>& labels,
             int max_depth) {
        root = build_tree(data, labels, max_depth);
    }

    // Predecir una sola muestra
    double predict_sample(const std::vector<double>& sample, Node* node) {
        if (!node->left && !node->right) {
            return node->leaf_values.begin()->first; // Retorna la etiqueta más frecuente
        }

        if (sample[std::stoi(node->feature.substr(8))] <= node->threshold) {
            return predict_sample(sample, node->left);
        } else {
            return predict_sample(sample, node->right);
        }
    }

    // Predecir varias muestras
    std::vector<double> predict(const std::vector<std::vector<double>>& data) {
        std::vector<double> predictions;
        for (const auto& sample : data) {
            predictions.push_back(predict_sample(sample, root));
        }
        return predictions;
    }

    ~DecisionTree() {
        delete root;
    }
};

// Enlace Pybind11
PYBIND11_MODULE(decision_tree, m) {
    py::class_<DecisionTree>(m, "DecisionTree")
        .def(py::init<>())
        .def("fit", &DecisionTree::fit)
        .def("predict", &DecisionTree::predict);
}
