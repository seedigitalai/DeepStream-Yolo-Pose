#include "pose_inference_config.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kDefaultMergeIouThreshold = 0.45f;

NvDsYoloPoseInferenceConfig defaultConfig()
{
  NvDsYoloPoseInferenceConfig config{};
  config.mode = NVDS_YOLO_POSE_INFERENCE_SINGLE_PASS;
  config.merge_iou_threshold = kDefaultMergeIouThreshold;
  return config;
}

NvDsYoloPoseInferenceConfig g_config = defaultConfig();

void writeError(char *error, size_t error_size, const char *message)
{
  if (!error || error_size == 0) {
    return;
  }
  std::snprintf(error, error_size, "%s", message);
}

bool validateConfig(const NvDsYoloPoseInferenceConfig& config, char *error, size_t error_size)
{
  if (config.merge_iou_threshold < 0.0f || config.merge_iou_threshold > 1.0f) {
    writeError(error, error_size, "merge_iou_threshold must be in [0, 1]");
    return false;
  }

  if (config.mode == NVDS_YOLO_POSE_INFERENCE_SINGLE_PASS) {
    return true;
  }

  if (config.mode != NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW) {
    writeError(error, error_size, "unsupported pose inference mode");
    return false;
  }

  if (config.window_width <= 0 || config.window_height <= 0 ||
      config.stride_x <= 0 || config.stride_y <= 0) {
    writeError(error, error_size,
        "sliding-window mode requires positive window and stride values");
    return false;
  }

  return true;
}

bool validatePreprocessConfig(const NvDsYoloPosePreprocessConfig& config,
    char *error, size_t error_size)
{
  if (!config.output_path || config.output_path[0] == '\0') {
    writeError(error, error_size, "preprocess output_path is required");
    return false;
  }
  if (!config.tensor_name || config.tensor_name[0] == '\0') {
    writeError(error, error_size, "preprocess tensor_name is required");
    return false;
  }
  if (!config.custom_lib_path || config.custom_lib_path[0] == '\0') {
    writeError(error, error_size, "preprocess custom_lib_path is required");
    return false;
  }
  if (config.frame_width <= 0 || config.frame_height <= 0 ||
      config.source_count <= 0 || config.target_gie_unique_id <= 0 ||
      config.preprocess_unique_id <= 0) {
    writeError(error, error_size, "preprocess frame, source, and unique-id values must be positive");
    return false;
  }
  if (!validateConfig(config.inference, error, error_size)) {
    return false;
  }
  if (config.inference.mode != NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW) {
    writeError(error, error_size, "preprocess config requires sliding-window mode");
    return false;
  }
  return true;
}

std::vector<int> windowStarts(int frame_size, int window_size, int stride)
{
  std::vector<int> starts;
  if (frame_size <= 0 || window_size <= 0 || stride <= 0) {
    return starts;
  }
  if (frame_size <= window_size) {
    starts.push_back(0);
    return starts;
  }

  for (int start = 0; start + window_size < frame_size; start += stride) {
    starts.push_back(start);
  }

  const int final_start = frame_size - window_size;
  if (starts.empty() || starts.back() != final_start) {
    starts.push_back(final_start);
  }
  return starts;
}

std::string roiParams(const std::vector<int>& xs, const std::vector<int>& ys,
    int window_width, int window_height)
{
  std::ostringstream out;
  bool first = true;
  for (int y : ys) {
    for (int x : xs) {
      if (!first) {
        out << ';';
      }
      first = false;
      out << x << ';' << y << ';' << window_width << ';' << window_height;
    }
  }
  return out.str();
}

float overlap1D(float a_min, float a_max, float b_min, float b_max)
{
  if (a_min > b_min) {
    std::swap(a_min, b_min);
    std::swap(a_max, b_max);
  }
  return a_max < b_min ? 0.0f : std::min(a_max, b_max) - b_min;
}

} // namespace

extern "C" void
nvds_yolo_pose_reset_inference_config(void)
{
  g_config = defaultConfig();
}

extern "C" void
nvds_yolo_pose_get_inference_config(NvDsYoloPoseInferenceConfig *config)
{
  if (!config) {
    return;
  }
  *config = g_config;
}

extern "C" int
nvds_yolo_pose_set_inference_config(const NvDsYoloPoseInferenceConfig *config,
    char *error, size_t error_size)
{
  if (!config) {
    writeError(error, error_size, "config must not be null");
    return 0;
  }

  if (!validateConfig(*config, error, error_size)) {
    return 0;
  }

  g_config = *config;
  return 1;
}

extern "C" int
nvds_yolo_pose_write_preprocess_config(const NvDsYoloPosePreprocessConfig *config,
    char *error, size_t error_size)
{
  if (!config) {
    writeError(error, error_size, "preprocess config must not be null");
    return 0;
  }
  if (!validatePreprocessConfig(*config, error, error_size)) {
    return 0;
  }

  const auto xs = windowStarts(config->frame_width, config->inference.window_width,
      config->inference.stride_x);
  const auto ys = windowStarts(config->frame_height, config->inference.window_height,
      config->inference.stride_y);
  if (xs.empty() || ys.empty()) {
    writeError(error, error_size, "failed to generate sliding-window ROI list");
    return 0;
  }

  const int windows_per_source = static_cast<int>(xs.size() * ys.size());
  const int tensor_batch = windows_per_source * config->source_count;

  std::ofstream out(config->output_path);
  if (!out) {
    writeError(error, error_size, "failed to open preprocess output_path");
    return 0;
  }

  out << "[property]\n"
      << "enable=1\n"
      << "target-unique-ids=" << config->target_gie_unique_id << "\n"
      << "network-input-order=0\n"
      << "process-on-frame=1\n"
      << "unique-id=" << config->preprocess_unique_id << "\n"
      << "gpu-id=" << config->gpu_id << "\n"
      << "maintain-aspect-ratio=1\n"
      << "symmetric-padding=1\n"
      << "processing-width=" << config->inference.window_width << "\n"
      << "processing-height=" << config->inference.window_height << "\n"
      << "network-input-shape=" << tensor_batch << ";3;"
      << config->inference.window_height << ';'
      << config->inference.window_width << "\n"
      << "scaling-buf-pool-size=" << tensor_batch << "\n"
      << "tensor-buf-pool-size=" << tensor_batch << "\n"
      << "network-color-format=0\n"
      << "tensor-data-type=0\n"
      << "scaling-pool-memory-type=0\n"
      << "scaling-pool-compute-hw=0\n"
      << "tensor-name=" << config->tensor_name << "\n"
      << "scaling-filter=1\n"
      << "custom-lib-path=" << config->custom_lib_path << "\n"
      << "custom-tensor-preparation-function=CustomTensorPreparation\n\n"
      << "[user-configs]\n"
      << "pixel-normalization-factor=0.003921568\n"
      << "offsets=0;0;0\n\n"
      << "[group-0]\n"
      << "src-ids=";

  for (int source_id = 0; source_id < config->source_count; ++source_id) {
    if (source_id > 0) {
      out << ';';
    }
    out << source_id;
  }
  out << "\n"
      << "custom-input-transformation-function=CustomAsyncTransformation\n"
      << "process-on-roi=1\n"
      << "draw-roi=0\n";

  const std::string rois = roiParams(xs, ys, config->inference.window_width,
      config->inference.window_height);
  for (int source_id = 0; source_id < config->source_count; ++source_id) {
    out << "roi-params-src-" << source_id << '=' << rois << "\n";
  }

  if (!out) {
    writeError(error, error_size, "failed to write preprocess config");
    return 0;
  }

  return 1;
}

extern "C" int
nvds_yolo_pose_preprocess_tensor_batch_size(const NvDsYoloPosePreprocessConfig *config,
    int *tensor_batch_size, char *error, size_t error_size)
{
  if (!config || !tensor_batch_size) {
    writeError(error, error_size, "preprocess config and tensor_batch_size are required");
    return 0;
  }
  if (!validatePreprocessConfig(*config, error, error_size)) {
    return 0;
  }

  const auto xs = windowStarts(config->frame_width, config->inference.window_width,
      config->inference.stride_x);
  const auto ys = windowStarts(config->frame_height, config->inference.window_height,
      config->inference.stride_y);
  if (xs.empty() || ys.empty()) {
    writeError(error, error_size, "failed to generate sliding-window ROI list");
    return 0;
  }

  *tensor_batch_size = static_cast<int>(xs.size() * ys.size()) * config->source_count;
  return 1;
}

extern "C" int
nvds_yolo_pose_remap_keypoints_to_frame(float *mask, size_t mask_size,
    unsigned *mask_width, unsigned *mask_height,
    const NvDsYoloPoseRoiTransform *transform, char *error, size_t error_size)
{
  constexpr size_t kBytesPerKeypoint = sizeof(float) * 3;
  if (!mask || mask_size == 0 || mask_size % kBytesPerKeypoint != 0 ||
      !mask_width || !mask_height || !transform) {
    writeError(error, error_size, "keypoint remap requires mask, dimensions, and ROI transform");
    return 0;
  }
  if (transform->scale_ratio_x <= 0.0 || transform->scale_ratio_y <= 0.0 ||
      transform->frame_width == 0 || transform->frame_height == 0) {
    writeError(error, error_size, "keypoint remap transform has invalid scale or frame size");
    return 0;
  }

  const size_t keypoints = mask_size / kBytesPerKeypoint;
  for (size_t i = 0; i < keypoints; ++i) {
    float& x = mask[i * 3 + 0];
    float& y = mask[i * 3 + 1];
    x = static_cast<float>(
        (static_cast<double>(x) - transform->offset_left) / transform->scale_ratio_x +
        transform->roi_left);
    y = static_cast<float>(
        (static_cast<double>(y) - transform->offset_top) / transform->scale_ratio_y +
        transform->roi_top);
  }

  *mask_width = transform->frame_width;
  *mask_height = transform->frame_height;
  return 1;
}

extern "C" float
nvds_yolo_pose_box_iou(const NvDsYoloPoseObjectBox *a,
    const NvDsYoloPoseObjectBox *b)
{
  if (!a || !b || a->width <= 0.0f || a->height <= 0.0f ||
      b->width <= 0.0f || b->height <= 0.0f) {
    return 0.0f;
  }
  const float overlap_x = overlap1D(a->left, a->left + a->width,
      b->left, b->left + b->width);
  const float overlap_y = overlap1D(a->top, a->top + a->height,
      b->top, b->top + b->height);
  const float intersection = overlap_x * overlap_y;
  const float area_a = a->width * a->height;
  const float area_b = b->width * b->height;
  const float denominator = area_a + area_b - intersection;
  return denominator <= 0.0f ? 0.0f : intersection / denominator;
}

extern "C" int
nvds_yolo_pose_should_suppress_duplicate(const NvDsYoloPoseObjectBox *candidate,
    const NvDsYoloPoseObjectBox *kept, float merge_iou_threshold)
{
  if (!candidate || !kept || candidate->class_id != kept->class_id ||
      merge_iou_threshold < 0.0f || merge_iou_threshold > 1.0f) {
    return 0;
  }
  return nvds_yolo_pose_box_iou(candidate, kept) > merge_iou_threshold ? 1 : 0;
}
