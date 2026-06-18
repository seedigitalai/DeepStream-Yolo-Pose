/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "nvdsinfer_custom_impl.h"

#include <cassert>
#include <algorithm>
#include <iostream>

#define NMS_THRESH 0.45

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static float
clamp(float val, float minVal, float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

static float
overlap1D(float x1min, float x1max, float x2min, float x2max)
{
  if (x1min > x2min) {
    std::swap(x1min, x2min);
    std::swap(x1max, x2max);
  }
  return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
}

static float
computeIoU(NvDsInferInstanceMaskInfo& bbox1, NvDsInferInstanceMaskInfo& bbox2)
{
  float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
  float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
  float area1 = (bbox1.width) * (bbox1.height);
  float area2 = (bbox2.width) * (bbox2.height);
  float overlap2D = overlapX * overlapY;
  float u = area1 + area2 - overlap2D;
  return u == 0 ? 0 : overlap2D / u;
}

static std::vector<NvDsInferInstanceMaskInfo>
nonMaximumSuppression(std::vector<NvDsInferInstanceMaskInfo> binfo)
{
  std::stable_sort(binfo.begin(), binfo.end(), [](const NvDsInferInstanceMaskInfo& b1,
      const NvDsInferInstanceMaskInfo& b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
  });

  std::vector<NvDsInferInstanceMaskInfo> out;

  for (auto i : binfo) {
    bool keep = true;
    for (auto j : out) {
      if (keep) {
        float overlap = computeIoU(i, j);
        keep = overlap <= NMS_THRESH;
      }
      else {
        break;
      }
    }
    if (keep) {
      out.push_back(i);
    }
    else {
      delete[] i.mask;
    }
  }

  return out;
}

static std::vector<NvDsInferInstanceMaskInfo>
nmsAllClasses(std::vector<NvDsInferInstanceMaskInfo>& binfo)
{
  std::vector<NvDsInferInstanceMaskInfo> result = nonMaximumSuppression(binfo);
  return result;
}

struct PoseTensorView
{
  const float* output = nullptr;
  size_t proposals = 0;
  size_t channels = 0;
  bool channelMajor = false;

  float at(size_t proposal, size_t channel) const
  {
    return channelMajor ? output[channel * proposals + proposal] : output[proposal * channels + channel];
  }
};

// YOLO26-Pose exports for this repo are [57, proposals], and older
// DeepStream-Yolo-Pose exports may arrive as [proposals, channels].
// The 57-channel contract is:
// x1,y1,x2,y2,score,class,kpt0_x,kpt0_y,kpt0_score,...
// Keep the legacy DeepStream-Yolo-Pose layout too, where keypoints start at 5.
static size_t
poseKeypointOffset(size_t channelsSize)
{
  if ((channelsSize - 5) % 3 == 0) {
    return 5;
  }
  if ((channelsSize - 6) % 3 == 0) {
    return 6;
  }
  return 0;
}

static void
addPoseProposal(const PoseTensorView& tensor, size_t kptOffset, uint netW, uint netH, size_t n,
    NvDsInferInstanceMaskInfo& b)
{
  size_t kptsSize = tensor.channels - kptOffset;
  b.mask = new float[kptsSize];
  for (size_t p = 0; p < kptsSize / 3; ++p) {
    b.mask[p * 3 + 0] = clamp(tensor.at(n, kptOffset + p * 3 + 0), 0, netW);
    b.mask[p * 3 + 1] = clamp(tensor.at(n, kptOffset + p * 3 + 1), 0, netH);
    b.mask[p * 3 + 2] = tensor.at(n, kptOffset + p * 3 + 2);
  }
  b.mask_width = netW;
  b.mask_height = netH;
  b.mask_size = sizeof(float) * kptsSize;
}

static void
addBBoxProposal(float x1, float y1, float x2, float y2, uint netW, uint netH, int maxIndex, float maxProb,
    NvDsInferInstanceMaskInfo& b)
{
  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  if (b.width < 1 || b.height < 1) {
    return;
  }

  b.detectionConfidence = maxProb;
  // nvinfer maps this numeric classId to NvDsObjectMeta::obj_label via
  // labelfile-path; the pose keypoint payload itself does not carry labels.
  b.classId = maxIndex;
}

static std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloPose(const PoseTensorView& tensor, uint netW, uint netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferInstanceMaskInfo> objects;
  const size_t kptOffset = poseKeypointOffset(tensor.channels);
  if (tensor.proposals == 0 || tensor.channels < 8 || kptOffset == 0) {
    return objects;
  }

  for (size_t n = 0; n < tensor.proposals; ++n) {
    float maxProb = tensor.at(n, 4);

    if (maxProb < preclusterThreshold[0]) {
      continue;
    }

    float x1 = tensor.at(n, 0);
    float y1 = tensor.at(n, 1);
    float x2 = tensor.at(n, 2);
    float y2 = tensor.at(n, 3);

    NvDsInferInstanceMaskInfo b;

    addBBoxProposal(x1, y1, x2, y2, netW, netH, 0, maxProb, b);
    addPoseProposal(tensor, kptOffset, netW, netH, n, b);

    objects.push_back(b);
  }

  return objects;
}

static bool
NvDsInferParseCustomYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR - Could not find output layer" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];

  PoseTensorView tensor;
  tensor.output = (const float*) (output.buffer);
  tensor.proposals = output.inferDims.d[0];
  tensor.channels = output.inferDims.d[1];

  if (output.inferDims.numDims >= 2 && output.inferDims.d[0] <= 128 &&
      output.inferDims.d[1] > output.inferDims.d[0]) {
    tensor.channels = output.inferDims.d[0];
    tensor.proposals = output.inferDims.d[1];
    tensor.channelMajor = true;
  }

  std::vector<NvDsInferInstanceMaskInfo> objects = decodeTensorYoloPose(tensor, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);

  objectList.clear();
  objectList = nmsAllClasses(objects);

  return true;
}

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloPose(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPose);
