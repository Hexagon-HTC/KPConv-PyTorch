#include <torch/extension.h>

#include <tuple>

torch::Tensor gridIndexAssignment(const torch::Tensor &dataXYZ, double gridSize);

std::tuple<torch::Tensor, torch::Tensor> gridSamplingXYZ(const torch::Tensor &dataXYZ, const torch::Tensor &dataLengths, double gridSize);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> gridSampling(const torch::Tensor &dataXYZ, const torch::Tensor &features, const torch::Tensor &labels,
                                                                                    const torch::Tensor &dataLengths, double gridSize);
