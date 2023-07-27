#include <torch/extension.h>

torch::Tensor radiusSearch(const torch::Tensor &dataXYZ, const torch::Tensor &queriesXYZ, const torch::Tensor &dataLengths, const torch::Tensor &queriesLengths, double radius);
