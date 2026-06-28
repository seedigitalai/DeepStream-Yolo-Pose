#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

/* ------------------------------------------------------------------ */
/* RTMPose-m SimCC parser for DeepStream nvinfer.                      */
/*                                                                     */
/* Decodes SimCC output tensors (simcc_x, simcc_y) into keypoint       */
/* triplets stored in NvDsInferInstanceMaskInfo::mask.                 */
/*                                                                     */
/* Model contract:                                                     */
/*   Input:  [1, 3, 256, 192]  (NCHW, portrait)                       */
/*   Output: simcc_x  [1, 17, 384]   (2× horizontal binning)          */
/*           simcc_y  [1, 17, 512]   (2× vertical binning)            */
/*                                                                     */
/* The parser emits one NvDsInferInstanceMaskInfo per batch element    */
/* with 17 keypoints × (x, y, confidence).  Coordinates are in         */
/* network input space (0..192, 0..256).                               */
/* ------------------------------------------------------------------ */

extern "C" bool NvDsInferParseRtmpose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList);

/* ---- helpers ------------------------------------------------------ */

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -10.0f, 10.0f)));
}

/* Sub-pixel refinement: expectation over ±hw bins around the peak.
   Returns the refined bin position (fractional). */
static float refineBin(const float* scores, int peak, int nbins, int hw = 2)
{
    int L = std::max(0, peak - hw);
    int R = std::min(nbins, peak + hw + 1);
    float num = 0.0f, den = 0.0f;
    for (int i = L; i < R; ++i) {
        float w = sigmoid(scores[i]);
        num += static_cast<float>(i) * w;
        den += w;
    }
    return den > 0.0f ? num / den : static_cast<float>(peak);
}

/* Decode one batch element of SimCC into 17 keypoints.
   simcc_x: [K, Xbins]  simcc_y: [K, Ybins]
   K = 17 (COCO keypoints)
   Returns keypoints in network input pixel space. */
static void decodeSimcc(
    const float* simcc_x, int K, int Xbins,
    const float* simcc_y, int Ybins,
    float netW, float netH,
    float* outKps)  // outKps[K * 3]  {x, y, conf}
{
    for (int k = 0; k < K; ++k) {
        const float* rowX = simcc_x + k * Xbins;
        const float* rowY = simcc_y + k * Ybins;

        // Argmax on raw logits
        int mx = 0, my = 0;
        float vx = rowX[0], vy = rowY[0];
        for (int i = 1; i < Xbins; ++i) {
            if (rowX[i] > vx) { vx = rowX[i]; mx = i; }
        }
        for (int i = 1; i < Ybins; ++i) {
            if (rowY[i] > vy) { vy = rowY[i]; my = i; }
        }

        // Sub-pixel refinement with sigmoid weighting
        float rx = refineBin(rowX, mx, Xbins);
        float ry = refineBin(rowY, my, Ybins);

        // SimCC uses 2× binning → divide by 2 to get pixel coords
        float px = rx / 2.0f;
        float py = ry / 2.0f;

        // Confidence: geometric mean of peak sigmoid scores
        float conf = std::sqrt(sigmoid(rowX[mx]) * sigmoid(rowY[my]));

        outKps[k * 3 + 0] = std::clamp(px, 0.0f, netW);
        outKps[k * 3 + 1] = std::clamp(py, 0.0f, netH);
        outKps[k * 3 + 2] = conf;
    }
}

/* ---- main entry point --------------------------------------------- */

static bool NvDsInferParseCustomRtmpose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    if (outputLayersInfo.size() < 2) {
        std::cerr << "[Rtmpose] ERROR: expected 2 output layers, got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    // Locate simcc_x and simcc_y by name
    const float* simccXBuf = nullptr;
    const float* simccYBuf = nullptr;
    int K = 0, Xbins = 0, Ybins = 0;
    int batchSize = 1;

    for (auto const& layer : outputLayersInfo) {
        if (!layer.buffer || layer.inferDims.numDims < 3) continue;

        // Batch dimension may be -1 (dynamic); clamp to 1
        int B = static_cast<int>(layer.inferDims.d[0]);
        if (B <= 0) B = 1;
        batchSize = std::max(batchSize, B);

        int k = static_cast<int>(layer.inferDims.d[1]);
        int bins = static_cast<int>(layer.inferDims.d[2]);

        if (std::strstr(layer.layerName, "simcc_x") ||
            std::strstr(layer.layerName, "x")) {
            simccXBuf = static_cast<const float*>(layer.buffer);
            K = k;
            Xbins = bins;
        } else if (std::strstr(layer.layerName, "simcc_y") ||
                   std::strstr(layer.layerName, "y")) {
            simccYBuf = static_cast<const float*>(layer.buffer);
            K = k;
            Ybins = bins;
        }
    }

    if (!simccXBuf || !simccYBuf || K == 0 || Xbins == 0 || Ybins == 0) {
        std::cerr << "[Rtmpose] ERROR: could not locate simcc_x / simcc_y tensors"
                  << std::endl;
        return false;
    }

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const size_t kptsFloatCount = static_cast<size_t>(K) * 3;

    objectList.clear();

    for (int b = 0; b < batchSize; ++b) {
        NvDsInferInstanceMaskInfo obj{};

        obj.mask = new float[kptsFloatCount];
        obj.mask_size = sizeof(float) * kptsFloatCount;
        obj.mask_width = networkInfo.width;
        obj.mask_height = networkInfo.height;

        decodeSimcc(
            simccXBuf + b * K * Xbins, K, Xbins,
            simccYBuf + b * K * Ybins, Ybins,
            netW, netH,
            obj.mask);

        // Set a dummy detection confidence so nvinfer doesn't filter it out
        obj.detectionConfidence = 0.999f;
        obj.classId = 0;

        objectList.push_back(obj);
    }

    return true;
}

/* ---- exported symbol ---------------------------------------------- */

extern "C" bool NvDsInferParseRtmpose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    return NvDsInferParseCustomRtmpose(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}
