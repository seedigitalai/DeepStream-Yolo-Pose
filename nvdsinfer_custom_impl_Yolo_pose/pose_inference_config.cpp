#include "pose_inference_config.h"

#include <cstdio>

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
