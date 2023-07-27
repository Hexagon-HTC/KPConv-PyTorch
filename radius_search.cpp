#include <nanoflann.hpp>
#include <omp.h>
#include <torch/script.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#define CHECK_NOT_CUDA(x) AT_ASSERTM(!x.is_cuda(), #x " must be CPU tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

using namespace torch::indexing;

struct PointCloud
{
    PointCloud(size_t size_, const float *points_) : size(size_), points(points_)
    {
    }

    inline size_t kdtree_get_point_count() const
    {
        return size;
    }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return points[3 * idx + dim];
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX &) const
    {
        return false;
    }

    size_t size;
    const float *points;
};

void radiusSearchNanoflann(const torch::Tensor &dataXYZ, const torch::Tensor &queriesXYZ, const torch::Tensor &dataLengths, const torch::Tensor &queriesLengths,
                           std::vector<int> &neighborInds, double radius)
{
    const float radiusSquared = radius * radius;
    const int queryCount = queriesXYZ.sizes()[0];

    std::vector<std::vector<std::pair<size_t, float>>> queryResults(queryCount);

    const float *const dataPtr = static_cast<const float *>(dataXYZ.data_ptr());
    const float *const queryPtr = static_cast<const float *>(queriesXYZ.data_ptr());

    const int *const dataLengthsPtr = static_cast<const int *>(dataLengths.data_ptr());
    const int *const queriesLengthsPtr = static_cast<const int *>(queriesLengths.data_ptr());

    const nanoflann::KDTreeSingleIndexAdaptorParams treeParams(10);
    const nanoflann::SearchParams searchParams(32, 0.0f, true);

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> my_kd_tree_t;

    const int batchCount = dataLengths.sizes()[0];

    std::vector<int> dataIndexOffsets(batchCount);
    std::vector<int> queryIndexOffsets(batchCount);

    int dataIndexOffset = 0;
    int queryIndexOffset = 0;

    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
    {
        dataIndexOffsets[batchIdx] = dataIndexOffset;
        dataIndexOffset += dataLengthsPtr[batchIdx];

        queryIndexOffsets[batchIdx] = queryIndexOffset;
        queryIndexOffset += queriesLengthsPtr[batchIdx];
    }

    std::vector<size_t> maxMatchCounts(batchCount, 0u);

#pragma omp parallel for
    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
    {
        PointCloud curPointCloud(dataLengthsPtr[batchIdx], dataPtr + 3 * dataIndexOffsets[batchIdx]);
        my_kd_tree_t *index = nullptr;

        index = new my_kd_tree_t(3, curPointCloud, treeParams);
        index->buildIndex();

        const float *curQueryPtr = queryPtr + 3 * queryIndexOffsets[batchIdx];

        for (int queryIdx = 0; queryIdx < queriesLengthsPtr[batchIdx]; ++queryIdx)
        {
            const int globalQueryIdx = queryIndexOffsets[batchIdx] + queryIdx;
            queryResults[globalQueryIdx].reserve(maxMatchCounts[batchIdx]);

            const size_t matchCount = index->radiusSearch(curQueryPtr, radiusSquared, queryResults[globalQueryIdx], searchParams);
            curQueryPtr += 3;

            maxMatchCounts[batchIdx] = std::max(matchCount, maxMatchCounts[batchIdx]);
        }

        delete index;
    }

    const size_t maxMatchCount = *std::max_element(maxMatchCounts.begin(), maxMatchCounts.end());
    const int padValue = dataXYZ.sizes()[0];

    neighborInds.resize(queryCount * maxMatchCount, padValue);

#pragma omp parallel for
    for (int batchIdx = 0; batchIdx < batchCount; ++batchIdx)
    {
        for (int queryIdx = 0; queryIdx < queriesLengthsPtr[batchIdx]; ++queryIdx)
        {
            const int globalQueryIdx = queryIndexOffsets[batchIdx] + queryIdx;
            const int globalQueryOffset = globalQueryIdx * maxMatchCount;

            for (int matchIdx = 0; matchIdx < static_cast<int>(queryResults[globalQueryIdx].size()); ++matchIdx)
            {
                neighborInds[globalQueryOffset + matchIdx] = queryResults[globalQueryIdx][matchIdx].first + dataIndexOffsets[batchIdx];
            }
        }
    }
}

torch::Tensor radiusSearch(const torch::Tensor &dataXYZ, const torch::Tensor &queriesXYZ, const torch::Tensor &dataLengths, const torch::Tensor &queriesLengths, double radius)
{
    CHECK_NOT_CUDA(dataXYZ);
    IS_CONTIGUOUS(dataXYZ);

    CHECK_NOT_CUDA(queriesXYZ);
    IS_CONTIGUOUS(queriesXYZ);

    CHECK_NOT_CUDA(dataLengths);
    IS_CONTIGUOUS(dataLengths);

    CHECK_NOT_CUDA(queriesLengths);
    IS_CONTIGUOUS(queriesLengths);

    std::vector<int> neighborInds;
    radiusSearchNanoflann(dataXYZ, queriesXYZ, dataLengths, queriesLengths, neighborInds, radius);

    const int queryCount = queriesXYZ.sizes()[0];
    const int maxNeighborCount = neighborInds.size() / queryCount;

    torch::Tensor ret = torch::empty({static_cast<int64_t>(neighborInds.size())}, torch::dtype(torch::kInt));

    memcpy(ret.data_ptr(), neighborInds.data(), 4u * neighborInds.size());
    return ret.view({queryCount, maxNeighborCount});
}

static auto registry = torch::RegisterOperators("kpconv_ops::radius_search", &radiusSearch);
