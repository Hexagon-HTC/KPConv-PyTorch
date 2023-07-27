#include <omp.h>
#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#define CHECK_NOT_CUDA(x) AT_ASSERTM(!x.is_cuda(), #x " must be CPU tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

struct PointXYZ
{
    float x, y, z;

    PointXYZ() : x(0.0f), y(0.0f), z(0.0f)
    {
    }

    PointXYZ(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

inline PointXYZ operator*(const PointXYZ P, const float a)
{
    return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline PointXYZ operator/(const PointXYZ P, const float a)
{
    return PointXYZ(P.x / a, P.y / a, P.z / a);
}

inline PointXYZ floorPoint(const PointXYZ point)
{
    return PointXYZ(std::floor(point.x), std::floor(point.y), std::floor(point.z));
}

void calcMinMaxPoint(const float *dataXYZ, int pointCount, PointXYZ &minPoint, PointXYZ &maxPoint)
{
    minPoint = PointXYZ(dataXYZ[0], dataXYZ[1], dataXYZ[2]);
    maxPoint = PointXYZ(dataXYZ[0], dataXYZ[1], dataXYZ[2]);

    for (int idx = 1; idx < pointCount; ++idx)
    {
        minPoint.x = std::min(minPoint.x, dataXYZ[0]);
        maxPoint.x = std::max(maxPoint.x, dataXYZ[0]);

        minPoint.y = std::min(minPoint.y, dataXYZ[1]);
        maxPoint.y = std::max(maxPoint.y, dataXYZ[1]);

        minPoint.z = std::min(minPoint.z, dataXYZ[2]);
        maxPoint.z = std::max(maxPoint.z, dataXYZ[2]);

        dataXYZ += 3;
    }
}

namespace
{
    constexpr int MAX_FEATURE_DIM = 8;
    constexpr int MAX_LABEL_COUNT = 64;

}

struct SampledData
{
    int count;
    int featureDim;

    PointXYZ accumulatedPoint;
    float features[MAX_FEATURE_DIM];
    int labels[MAX_LABEL_COUNT];

    SampledData() : SampledData(0, 0)
    {
    }

    SampledData(int featureDim, int labelCount) : count(0), accumulatedPoint(), featureDim(featureDim)
    {
        for (int featureIdx = 0; featureIdx < featureDim; ++featureIdx)
        {
            features[featureIdx] = 0.0f;
        }

        memset(labels, 0, 4u * static_cast<size_t>(labelCount));
    }

    void updatePoint(const float *point)
    {
        count += 1;

        accumulatedPoint.x += point[0];
        accumulatedPoint.y += point[1];
        accumulatedPoint.z += point[2];
    }

    void updateFeatures(const float *features)
    {
        for (int featureIdx = 0; featureIdx < featureDim; ++featureIdx)
        {
            this->features[featureIdx] += features[featureIdx];
        }
    }

    void updateLabel(int label)
    {
        labels[label] += 1;
    }
};

void calcGridSamplingForBatchXYZ(const float *dataXYZ, std::vector<PointXYZ> &sampledXYZ, int pointCount, float gridSize)
{
    PointXYZ minCorner, maxCorner;
    calcMinMaxPoint(dataXYZ, pointCount, minCorner, maxCorner);

    const PointXYZ originCorner = floorPoint(minCorner / gridSize) * gridSize;

    const int sampleNX = std::floor((maxCorner.x - originCorner.x) / gridSize) + 1;
    const int sampleNY = std::floor((maxCorner.y - originCorner.y) / gridSize) + 1;

    std::unordered_map<int, SampledData> gridMap;

    int iX, iY, iZ;
    int mapKey;

    for (int pointIdx = 0; pointIdx < pointCount; ++pointIdx)
    {
        // We calculate the position of the point in a grid map.
        iX = std::floor((dataXYZ[0] - originCorner.x) / gridSize);
        iY = std::floor((dataXYZ[1] - originCorner.y) / gridSize);
        iZ = std::floor((dataXYZ[2] - originCorner.z) / gridSize);

        mapKey = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

        if (gridMap.count(mapKey) == 0)
        {
            gridMap.emplace(mapKey, SampledData());
        }

        gridMap[mapKey].updatePoint(dataXYZ);
        dataXYZ += 3;
    }

    sampledXYZ.resize(gridMap.size());
    int idx = 0;

    for (const auto &keyValue : gridMap)
    {
        sampledXYZ[idx++] = keyValue.second.accumulatedPoint / keyValue.second.count;
    }
}

void calcGridSamplingForBatch(const float *dataXYZ, const float *features, const int *labels, std::vector<PointXYZ> &sampledXYZ, std::vector<float> &sampledFeatures,
                              std::vector<int> &sampledLabels, int pointCount, int featureDim, float gridSize)
{
    PointXYZ minCorner, maxCorner;
    calcMinMaxPoint(dataXYZ, pointCount, minCorner, maxCorner);

    const PointXYZ originCorner = floorPoint(minCorner / gridSize) * gridSize;

    const int sampleNX = std::floor((maxCorner.x - originCorner.x) / gridSize) + 1;
    const int sampleNY = std::floor((maxCorner.y - originCorner.y) / gridSize) + 1;

    int labelCount = 0;

    if (labels != nullptr)
    {
        labelCount = *std::max_element(labels, labels + pointCount) + 1;
    }

    if (labelCount > MAX_LABEL_COUNT)
    {
        AT_ASSERTM(false, "Exceeded maximum label count, please recompile with higher MAX_LABEL_COUNT");
    }

    std::unordered_map<int, SampledData> gridMap;

    int iX, iY, iZ;
    int mapKey;

    for (int pointIdx = 0; pointIdx < pointCount; ++pointIdx)
    {
        // We calculate the position of the point in a grid map.
        iX = std::floor((dataXYZ[0] - originCorner.x) / gridSize);
        iY = std::floor((dataXYZ[1] - originCorner.y) / gridSize);
        iZ = std::floor((dataXYZ[2] - originCorner.z) / gridSize);

        mapKey = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

        if (gridMap.count(mapKey) == 0)
        {
            gridMap.emplace(mapKey, SampledData(featureDim, labelCount));
        }

        SampledData &curData = gridMap[mapKey];

        curData.updatePoint(dataXYZ);

        if (features != nullptr)
        {
            curData.updateFeatures(features);
            features += featureDim;
        }

        if (labels != nullptr)
        {
            curData.updateLabel(*labels);
            labels += 1;
        }

        dataXYZ += 3;
    }

    sampledXYZ.resize(gridMap.size());

    if (features != nullptr)
    {
        sampledFeatures.resize(gridMap.size() * featureDim);
    }

    if (labels != nullptr)
    {
        sampledLabels.resize(gridMap.size());
    }

    int idx = 0;

    for (const auto &keyValue : gridMap)
    {
        const int count = keyValue.second.count;

        sampledXYZ[idx] = keyValue.second.accumulatedPoint / count;

        if (features != nullptr)
        {
            for (int featureIdx = 0; featureIdx < featureDim; ++featureIdx)
            {
                sampledFeatures[idx * featureDim + featureIdx] = keyValue.second.features[featureIdx] / count;
            }
        }

        if (labels != nullptr)
        {
            sampledLabels[idx] = std::max_element(keyValue.second.labels, keyValue.second.labels + labelCount) - keyValue.second.labels;
        }

        idx += 1;
    }
}

void calcGridSampling(const torch::Tensor &dataXYZ, const torch::Tensor &features, const torch::Tensor &labels, const torch::Tensor &dataLengths, std::vector<PointXYZ> &sampledXYZ,
                      std::vector<float> &sampledFeatures, std::vector<int> &sampledLabels, std::vector<int> &sampledLengths, float gridSize)
{
    const float *const dataXYZPtr = static_cast<const float *>(dataXYZ.data_ptr());
    const int *const dataLengthsPtr = static_cast<const int *>(dataLengths.data_ptr());

    const float *featuresPtr = nullptr;
    const int *labelsPtr = nullptr;

    if (features.numel() > 0)
    {
        featuresPtr = static_cast<const float *>(features.data_ptr());
    }
    if (labels.numel() > 0)
    {
        labelsPtr = static_cast<const int *>(labels.data_ptr());
    }

    const int batchCount = dataLengths.size(0);
    const int featureDim = features.size(1);

    if (featureDim > MAX_FEATURE_DIM)
    {
        AT_ASSERTM(false, "Exceeded maximum feature dimension, please recompile with higher MAX_FEATURE_DIM");
    }

    std::vector<int> dataIndexOffsets(batchCount);
    int dataIndexOffset = 0;

    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
    {
        dataIndexOffsets[batchIdx] = dataIndexOffset;
        dataIndexOffset += dataLengthsPtr[batchIdx];
    }

    std::vector<std::vector<PointXYZ>> curSampledXYZ(batchCount);
    std::vector<std::vector<float>> curSampledFeatures(batchCount);
    std::vector<std::vector<int>> curSampledLabels(batchCount);

    // We compile without OpenMP: even though it provides gains for high batch sizes/point counts, it seems to slow down the training for the typical use-case.
    // #pragma omp parallel for
    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
    {
        if (features.numel() == 0 && labels.numel() == 0)
        {
            calcGridSamplingForBatchXYZ(dataXYZPtr + 3 * dataIndexOffsets[batchIdx], curSampledXYZ[batchIdx], dataLengthsPtr[batchIdx], gridSize);
        }
        else
        {
            const float *curFeaturesPtr = featuresPtr ? featuresPtr + featureDim * dataIndexOffsets[batchIdx] : nullptr;
            const int *curLabelsPtr = labelsPtr ? labelsPtr + dataIndexOffsets[batchIdx] : nullptr;

            calcGridSamplingForBatch(dataXYZPtr + 3 * dataIndexOffsets[batchIdx], curFeaturesPtr, curLabelsPtr, curSampledXYZ[batchIdx], curSampledFeatures[batchIdx],
                                     curSampledLabels[batchIdx], dataLengthsPtr[batchIdx], featureDim, gridSize);
        }

        sampledLengths[batchIdx] = curSampledXYZ[batchIdx].size();
    }

    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
    {
        sampledXYZ.insert(sampledXYZ.end(), curSampledXYZ[batchIdx].begin(), curSampledXYZ[batchIdx].end());

        if (curSampledFeatures[batchIdx].empty() == false)
        {
            sampledFeatures.insert(sampledFeatures.end(), curSampledFeatures[batchIdx].begin(), curSampledFeatures[batchIdx].end());
        }

        if (curSampledLabels[batchIdx].empty() == false)
        {
            sampledLabels.insert(sampledLabels.end(), curSampledLabels[batchIdx].begin(), curSampledLabels[batchIdx].end());
        }
    }
}

void calcGridIndexAssignment(const float *dataXYZ, int pointCount, std::vector<int> &assignedInds, float gridSize)
{
    PointXYZ minCorner, maxCorner;
    calcMinMaxPoint(dataXYZ, pointCount, minCorner, maxCorner);

    const PointXYZ originCorner = floorPoint(minCorner / gridSize) * gridSize;

    const int sampleNX = std::floor((maxCorner.x - originCorner.x) / gridSize) + 1;
    const int sampleNY = std::floor((maxCorner.y - originCorner.y) / gridSize) + 1;

    int iX, iY, iZ;
    assignedInds = std::vector<int>(pointCount);

    for (int pointIdx = 0; pointIdx < pointCount; ++pointIdx)
    {
        // We calculate the position of the point in a grid map.
        iX = std::floor((dataXYZ[0] - originCorner.x) / gridSize);
        iY = std::floor((dataXYZ[1] - originCorner.y) / gridSize);
        iZ = std::floor((dataXYZ[2] - originCorner.z) / gridSize);

        assignedInds[pointIdx] = iX + sampleNX * iY + sampleNX * sampleNY * iZ;
        dataXYZ += 3;
    }
}

torch::Tensor gridIndexAssignment(const torch::Tensor &dataXYZ, double gridSize)
{
    CHECK_NOT_CUDA(dataXYZ);
    IS_CONTIGUOUS(dataXYZ);

    std::vector<int> assignedInds;
    const float *const dataXYZPtr = static_cast<const float *>(dataXYZ.data_ptr());

    calcGridIndexAssignment(dataXYZPtr, dataXYZ.size(0), assignedInds, gridSize);

    torch::Tensor ret = torch::empty({static_cast<int64_t>(assignedInds.size())}, torch::dtype(torch::kInt));
    memcpy(ret.data_ptr(), assignedInds.data(), 4u * assignedInds.size());

    return ret;
}

std::tuple<torch::Tensor, torch::Tensor> gridSamplingXYZ(const torch::Tensor &dataXYZ, const torch::Tensor &dataLengths, double gridSize)
{
    CHECK_NOT_CUDA(dataXYZ);
    IS_CONTIGUOUS(dataXYZ);

    CHECK_NOT_CUDA(dataLengths);
    IS_CONTIGUOUS(dataLengths);

    std::vector<PointXYZ> sampledXYZ;
    std::vector<float> sampledFeatures;
    std::vector<int> sampledLabels;
    std::vector<int> sampledLengths(dataLengths.size(0));

    calcGridSampling(dataXYZ, torch::empty({0L, 0L}), torch::empty({0L}), dataLengths, sampledXYZ, sampledFeatures, sampledLabels, sampledLengths, gridSize);

    torch::Tensor retXYZ = torch::empty({static_cast<int64_t>(sampledXYZ.size()), 3}, torch::dtype(torch::kFloat));
    memcpy(retXYZ.data_ptr(), sampledXYZ.data(), 12u * sampledXYZ.size());

    torch::Tensor retLengths = torch::empty({static_cast<int64_t>(sampledLengths.size())}, torch::dtype(torch::kInt));
    memcpy(retLengths.data_ptr(), sampledLengths.data(), 4u * sampledLengths.size());

    return {retXYZ, retLengths};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> gridSampling(const torch::Tensor &dataXYZ, const torch::Tensor &features, const torch::Tensor &labels,
                                                                                    const torch::Tensor &dataLengths, double gridSize)
{
    CHECK_NOT_CUDA(dataXYZ);
    IS_CONTIGUOUS(dataXYZ);

    CHECK_NOT_CUDA(features);
    IS_CONTIGUOUS(features);

    CHECK_NOT_CUDA(labels);
    IS_CONTIGUOUS(labels);

    CHECK_NOT_CUDA(dataLengths);
    IS_CONTIGUOUS(dataLengths);

    std::vector<PointXYZ> sampledXYZ;
    std::vector<float> sampledFeatures;
    std::vector<int> sampledLabels;
    std::vector<int> sampledLengths(dataLengths.size(0));

    calcGridSampling(dataXYZ, features, labels, dataLengths, sampledXYZ, sampledFeatures, sampledLabels, sampledLengths, gridSize);

    // We assume here that PointXYZ data (x, y, z) is laid out contiguously in memory.
    torch::Tensor retXYZ = torch::empty({static_cast<int64_t>(sampledXYZ.size()), 3}, torch::dtype(torch::kFloat));
    memcpy(retXYZ.data_ptr(), sampledXYZ.data(), 12u * sampledXYZ.size());

    torch::Tensor retFeatures = torch::empty({static_cast<int64_t>(sampledFeatures.size())}, torch::dtype(torch::kFloat));
    memcpy(retFeatures.data_ptr(), sampledFeatures.data(), 4u * sampledFeatures.size());

    torch::Tensor retLabels = torch::empty({static_cast<int64_t>(sampledLabels.size())}, torch::dtype(torch::kInt));
    memcpy(retLabels.data_ptr(), sampledLabels.data(), 4u * sampledLabels.size());

    torch::Tensor retLengths = torch::empty({static_cast<int64_t>(sampledLengths.size())}, torch::dtype(torch::kInt));
    memcpy(retLengths.data_ptr(), sampledLengths.data(), 4u * sampledLengths.size());

    return {retXYZ, retFeatures, retLabels, retLengths};
}

static auto registry1 = torch::RegisterOperators("kpconv_ops::grid_index_assignment", &gridIndexAssignment);
static auto registry2 = torch::RegisterOperators("kpconv_ops::grid_sampling_xyz", &gridSamplingXYZ);
static auto registry3 = torch::RegisterOperators("kpconv_ops::grid_sampling", &gridSampling);
