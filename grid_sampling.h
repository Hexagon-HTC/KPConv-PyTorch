#include <tuple>

#include <torch/extension.h>

torch::Tensor gridIndexAssignment(const torch::Tensor &dataXYZ, double gridSize);grid_

std::tuple<torch::Tensor, torch::Tensor> gridSamplingXYZ(const torch::Tensor &dataXYZ, const torch::Tensor &dataLengths, double gridSize);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> gridSampling(const torch::Tensor &dataXYZ, const torch::Tensor &features, const torch::Tensor &labels,
                                                                                    const torch::Tensor &dataLengths, double gridSize);
