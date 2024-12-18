// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <string>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>


namespace py = pybind11;


bool is_block_all_one(const uint64_t* dataPtr, int rowBlockSize, int colBlockSize, int splitNum)
{
    for (size_t i = 0; i < rowBlockSize; i++) {
        for (size_t j = 0; j < colBlockSize; j++) {
            if (*(dataPtr++) != 0x0101010101010101) {
                return false;
            }
        }
        dataPtr += colBlockSize * (splitNum - 1);
    }
    return true;
}

void sub_coarsen_mask(const uint64_t *dataPtr, int rowBlockSize, int colBlockSize, int splitNum,
                      at::Tensor &output, int blockIdxStart, int blockIdxEnd)
{
    if (splitNum == 0) {
        throw std::runtime_error("Split Number must be a positive integer.");
    }
    auto outputPtr = (uint8_t *) output.data_ptr<bool>();
    outputPtr += blockIdxStart;
    for (size_t i = blockIdxStart; i < blockIdxEnd; i++) {
        int blockRowIdx = std::floor(i / splitNum);
        int blockColIdx = i % splitNum;
        int grid_val = is_block_all_one(
            dataPtr + (blockRowIdx * rowBlockSize) * (splitNum * colBlockSize) + (blockColIdx * colBlockSize),
            rowBlockSize, colBlockSize, splitNum);
        *(outputPtr++) = grid_val;
    }
}

void coarsen_mask(const at::Tensor& input, const int splitNum, at::Tensor& output)
{
    int rowDim = input.size(0);
    int colDim = input.size(1);
    if (splitNum == 0) {
        throw std::runtime_error("Split number must be a positive integer.");
    }
    if (rowDim % splitNum != 0 || colDim % splitNum != 0) {
        throw std::runtime_error("Both dims of the input 2-dim matrix must be divisible by split num.");
    }
    int rowBlockSize = rowDim / splitNum;
    int colBlockSize = colDim / splitNum;
    int sizeRatioInt64ToBool = sizeof(uint64_t) / sizeof(bool);
    if (rowBlockSize % sizeRatioInt64ToBool != 0 || colBlockSize % sizeRatioInt64ToBool != 0) {
        throw std::runtime_error("Both dims of the input 2-dim matrix must be divisible by 8 * split_num, to iterate "
                                 "data pointer in uint64 instead of bool.");
    }
    auto dataPtr = (uint64_t*) input.data_ptr<bool>();
    colBlockSize /= sizeRatioInt64ToBool;
    std::vector<std::thread> threads;
    int totalNumBlocks = splitNum * splitNum;
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        throw std::runtime_error("Number of threads must be a positive integer.");
    }
    if (totalNumBlocks < numThreads) {
        numThreads = totalNumBlocks;
    }
    int blockNumPerThread = totalNumBlocks / numThreads;
    for (size_t i = 0; i < numThreads; ++i) {
        int blockIdxStart = i * blockNumPerThread;
        threads.emplace_back(sub_coarsen_mask, dataPtr, rowBlockSize, colBlockSize, splitNum, std::ref(output),
                             blockIdxStart, blockIdxStart + blockNumPerThread);
    }
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

void sub_select_perm_mask(const at::Tensor &input, const std::vector<uint64_t> indList, at::Tensor &output, int subIndCnt,
                          int subStartIdx)
{
    uint64_t seqLen = input.size(0);
    uint64_t indCnt = indList.size();
    auto maskTensorPtr = (uint8_t *) input.data_ptr<bool>();
    auto outputTensorPtr = (uint8_t *) output.data_ptr<bool>();
    uint8_t *subOutputPtr = outputTensorPtr + subStartIdx * indCnt;
    std::vector<uint64_t> rowStartIdxList(subIndCnt);
    for (size_t i = 0; i < subIndCnt; i++) {
        rowStartIdxList[i] = ((uint64_t) indList[subStartIdx + i] * seqLen);
    }

    for (size_t i = 0; i < subIndCnt; i++) {
        uint64_t rowStartIdx = rowStartIdxList[i];
        for (size_t j = 0; j < indCnt; j++) {
            uint64_t colIdx = indList[j];
            uint8_t extractedValue = *(maskTensorPtr + (rowStartIdx + colIdx));
            *(subOutputPtr++) = extractedValue;
        }
    }
}

void select_perm_mask(const at::Tensor &input, const std::vector<uint64_t> indList, at::Tensor &output)
{
    if (input.dim() != 2 || input.size(0) != input.size(1)) {
        throw std::runtime_error("Input mask must be 2-dimensional squared tensor.");
    }
    if (input.scalar_type() != torch::kBool) {
        throw std::runtime_error("The datatype of input mask must be bool.");
    }
    uint64_t indCnt = indList.size();
    std::vector<std::thread> threads;
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        throw std::runtime_error("Number of threads must be a positive integer.");
    }
    if (indCnt % numThreads != 0 || numThreads > indCnt) {
        numThreads = indCnt;
    }
    int subIndCnt = indCnt / numThreads;
    for (size_t i = 0; i < numThreads; ++i) {
        int subStartIdx = i * subIndCnt;
        threads.emplace_back(sub_select_perm_mask, input, indList, std::ref(output), subIndCnt, subStartIdx);
    }
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

// Function to calculate the Euclidean distance between two points
float euclidean_distance(const std::vector<float>& point1, const std::vector<float>& point2)
{
    float sum = 0.0f;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return std::sqrt(sum);
}

// Function to calculate distances between each point and all centroids
std::vector<std::vector<float>> calculate_distances(
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& centroids)
{
    std::vector<std::vector<float>> distances(data.size(), std::vector<float>(centroids.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < centroids.size(); ++j) {
            distances[i][j] = euclidean_distance(data[i], centroids[j]);
        }
    }
    return distances;
}

// Function to find the index of the minimum element in a vector
size_t argmin(const std::vector<float>& dataVec)
{
    return std::distance(dataVec.begin(), std::min_element(dataVec.begin(), dataVec.end()));
}

// FUnction to update centroids
std::vector<std::vector<float>> update_centroids(
    const std::vector<std::vector<float>>& data,
    const std::vector<size_t>& labels,
    size_t numClusters,
    size_t dimensionSize)
{
    std::vector<std::vector<float>> newCentroids(numClusters, std::vector<float>(dimensionSize, 0.0f));
    std::vector<size_t> counts(numClusters, 0);

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < dimensionSize; ++j) {
            newCentroids[labels[i]][j] += data[i][j];
        }
        counts[labels[i]]++;
    }

    for (size_t i = 0; i < numClusters; ++i) {
        if (counts[i] > 0) {
            for (size_t j = 0; j < dimensionSize; ++j) {
                newCentroids[i][j] /= counts[i];
            }
        } else {
            // Reinitialize centroid randomly if no points are assigned to this cluster
            newCentroids[i] = data[std::rand() % data.size()];
        }
    }

    return newCentroids;
}

bool allClose(const std::vector<std::vector<float>>& centroids,
              const std::vector<std::vector<float>>& newCentroids,
              float rtol = 1e-5, float atol = 1e-8)
{
    // Check if the dimensions match
    if (centroids.size() != newCentroids.size()) {
        return false;
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
        if (centroids[i].size() != newCentroids[i].size()) {
            return false;
        }

        for (size_t j = 0; j < centroids[i].size(); ++j) {
            float diff = std::fabs(centroids[i][j] - newCentroids[i][j]);
            float tol = atol + rtol * std::fabs(newCentroids[i][j]);
            if (diff > tol) {
                return false;
            }
        }
    }
    return true;
}

// Function to check if centroids have converged
bool centroids_converged(
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& newCentroids)
{
    return allClose(centroids, newCentroids);
}

std::vector<int> get_num_tasks_on_device(const torch::Tensor& gridMask)
{
    int P = gridMask.size(0);
    std::vector<int> numTaskList(P, 0);

    // 计算每行和每列中0的数量
    for (int i = 0; i < P; ++i) {
        int rowZeroCnt = 0;
        int colZeroCnt = 0;

        // 计算第i行中0的数量
        for (int j = 0; j < P; ++j) {
            if (gridMask[i][j].item<int>() == 0) {
                rowZeroCnt++;
            }
        }

        // 计算第i列中0的数量
        for (int j = 0; j < P; ++j) {
            if (gridMask[j][i].item<int>() == 0) {
                colZeroCnt++;
            }
        }

        // 第i行和第i列的0的数量之和
        numTaskList[i] = rowZeroCnt + colZeroCnt - (gridMask[i][i].item<int>() == 0 ? 1 : 0);
    }

    return numTaskList;
}

std::pair<float, float> get_score(const at::Tensor& mask, size_t cpSize, at::Tensor &gridMask)
{
    if (cpSize == 0) {
        throw std::runtime_error("CP size must be a positive integer.");
    }
    size_t maskSize = mask.size(0);
    coarsen_mask(mask, cpSize, gridMask);
    float totalTaskDensity = 1 - (gridMask.sum().item<float>() / (cpSize * cpSize));
    std::vector<int> numTaskList = get_num_tasks_on_device(gridMask);
    float taskNumDev = 0.0f;
    if (!numTaskList.empty()) {
        float mean = std::accumulate(numTaskList.begin(), numTaskList.end(), 0.0f) / numTaskList.size();
        float sum = 0.0f;
        for (const auto& num : numTaskList) {
            sum += (num - mean) * (num - mean);
        }
        taskNumDev = std::sqrt(sum / numTaskList.size());
    }
    return {totalTaskDensity, taskNumDev};
}

// Kmeans function
std::pair<std::vector<std::vector<float>>, std::vector<size_t>> kmeans(
    const std::vector<std::vector<float>>& data,
    size_t numClusters,
    size_t numIters)
{
    size_t seqLen = data.size();
    size_t dimensionSize = data[0].size();
    // Initialize centroids randomly
    std::vector<std::vector<float>> centroids(numClusters);
    std::srand(0);
    std::vector<size_t> indices(seqLen);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());
    for (size_t i = 0; i < numClusters; ++i) {
        centroids[i] = data[indices[i]];
    }
    std::vector<size_t> labels(seqLen);
    for (size_t iterIdx = 0; iterIdx < numIters; ++iterIdx) {
        // Calculate distances between each point and centroids
        std::vector<std::vector<float>> distances = calculate_distances(data, centroids);
        // Assign labels based on nearest centroid
        for (size_t i = 0; i < seqLen; ++ i) {
            labels[i] = argmin(distances[i]);
        }
        // Update centroids
        std::vector<std::vector<float>> newCentroids = update_centroids(data, labels, numClusters, dimensionSize);
        // Check for convergence
        if (centroids_converged(centroids, newCentroids)) {
            break;
        }
        centroids = newCentroids;
    }
    return {centroids, labels};
}

std::vector<size_t> search_kmeans(
    const at::Tensor& attnMask,
    const std::vector<std::vector<float>>& reducedMask,
    at::Tensor &tmpAttnMask,
    at::Tensor &tmpGridMask,
    at::Tensor &optGridMask,
    at::Tensor &optAttnMask,
    py::list optNumCluster,
    size_t cpSize,
    size_t numIters)
{
    std::vector<size_t> optSeq(attnMask.size(0));
    std::iota(optSeq.begin(), optSeq.end(), 0);
    auto [minTaskDensity, optTaskDev] = get_score(attnMask, cpSize, optGridMask);
    for (int numClusters = 2; numClusters < 9 ; ++numClusters) {
        auto [centroids, labels] = kmeans(reducedMask, numClusters, numIters);
        // Sort indices based on labels
        std::vector<size_t> sortedSeq(labels.size());
        std::iota(sortedSeq.begin(), sortedSeq.end(), 0);
        std::sort(sortedSeq.begin(), sortedSeq.end(), [&labels](size_t i, size_t j) {
            return labels[i] < labels[j];
        });
        select_perm_mask(attnMask, sortedSeq, tmpAttnMask);
        auto [taskDensity, taskNumDev] = get_score(tmpAttnMask, cpSize, tmpGridMask);
        if (taskDensity < minTaskDensity) {
            minTaskDensity = taskDensity;
            optAttnMask.copy_(tmpAttnMask);
            optNumCluster[0] = numClusters;
            optTaskDev = taskNumDev;
            optSeq = sortedSeq;
            optGridMask.copy_(tmpAttnMask);
        } else if (taskDensity == minTaskDensity && taskNumDev < optTaskDev) {
            optAttnMask.copy_(tmpAttnMask);
            optNumCluster[0] = numClusters;
            optTaskDev = taskNumDev;
            optSeq = sortedSeq;
            optGridMask.copy_(tmpGridMask);
        }
    }
    return optSeq;
}

void get_mask_list_with_remap(const at::Tensor& attnMask, at::Tensor& output, std::vector<int> rowIdxSeq, std::vector<int> colIdxSeq)
{
    size_t maskColLen = attnMask.size(1);
    size_t rowIdxLen = rowIdxSeq.size();
    size_t colIdxLen = colIdxSeq.size();
    if (rowIdxLen > output.size(0) || colIdxLen > output.size(1)) {
        throw std::runtime_error("Row or colum index length large than size of attention mask");
    }
    uint8_t *inputPtr = (uint8_t *) attnMask.data_ptr<bool>();
    uint8_t *outputPtr = (uint8_t *) output.data_ptr<bool>();

    for (size_t i = 0; i < rowIdxLen; i++) {
        uint8_t *inputRowStartPtr = inputPtr + rowIdxSeq[i] * maskColLen;
        for (size_t j = 0; j < colIdxLen; j++) {
            *(outputPtr++) = *(inputRowStartPtr + colIdxSeq[j]);
        }
    }
}

void get_mask_list_without_remap(const at::Tensor& attnMask, at::Tensor& output, std::vector<int> blockIdx, int cpSize)
{
    if (cpSize == 0) {
        throw std::runtime_error("CP size must be a positive integer.");
    }
    int sizeRatioInt64ToBool = sizeof(uint64_t) / sizeof(bool);
    int rowGridSize = attnMask.size(0) / cpSize;
    int colGridSize = rowGridSize / sizeRatioInt64ToBool;
    if (rowGridSize % sizeRatioInt64ToBool != 0) {
        throw std::runtime_error("Sequence length on each cp rank must be a multiple of 8");
    }
    int rowStartIdx = blockIdx[0] * rowGridSize;
    int colStartIdx = blockIdx[1] * colGridSize;

    uint64_t *inputPtr = (uint64_t*) attnMask.data_ptr<bool>();
    uint64_t *outputPtr = (uint64_t*) output.data_ptr<bool>();

    uint64_t *currPtr = inputPtr + rowStartIdx * (colGridSize * cpSize) + colStartIdx;
    int numUnitToNextRow = cpSize * colGridSize;

    uint64_t memmoveCnt = 0;
    if (colGridSize > std::numeric_limits<uint64_t>::max() / rowGridSize) {
        throw std::runtime_error("sequence length too long or context parallel size too small");
    }
    uint64_t outputSize = static_cast<uint64_t>(rowGridSize) * colGridSize;

    for (size_t i = 0; i < rowGridSize; i++) {
        if (memmoveCnt + colGridSize > outputSize) {
            throw std::runtime_error("Memory move out of range.");
        }
        memmove(outputPtr, currPtr, colGridSize * sizeof(uint64_t));
        memmoveCnt += colGridSize;
        outputPtr += colGridSize;
        currPtr += numUnitToNextRow;
    }
}

PYBIND11_MODULE(adaptive_cp, m)
{
m.def("coarsen_mask",
      &coarsen_mask,
      "A function that coarse a bool tensor with given split number",
      py::arg("input"), py::arg("splitNum"), py::arg("output"));
m.def("search_kmeans",
      &search_kmeans,
      "Search optimal k-means clustering result among various number of clusters",
      py::arg("attnMask"), py::arg("reduceMask"), py::arg("tmpAttnMask"), py::arg("tmpGridMask"),
      py::arg("optGridMask"), py::arg("optAttnMask"), py::arg("optNumCluster"), py::arg("cpSize"),
      py::arg("numIters"));
m.def("get_mask_list_with_remap",
      &get_mask_list_with_remap,
      py::arg("attnMask"), py::arg("output"), py::arg("rowIdxSeq"), py::arg("colIdxSeq"));
m.def("get_mask_list_without_remap",
      &get_mask_list_without_remap,
      py::arg("attnMask"), py::arg("output"), py::arg("blockIdx"), py::arg("cpSize"));
}
