#include "pose_inference_config.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

static void assert_default_config()
{
  NvDsYoloPoseInferenceConfig config{};
  nvds_yolo_pose_get_inference_config(&config);

  assert(config.mode == NVDS_YOLO_POSE_INFERENCE_SINGLE_PASS);
  assert(config.window_width == 0);
  assert(config.window_height == 0);
  assert(config.stride_x == 0);
  assert(config.stride_y == 0);
  assert(std::fabs(config.merge_iou_threshold - 0.45f) < 1e-6f);
}

static void assert_set_single_pass_config()
{
  NvDsYoloPoseInferenceConfig config{};
  config.mode = NVDS_YOLO_POSE_INFERENCE_SINGLE_PASS;
  config.merge_iou_threshold = 0.3f;

  assert(nvds_yolo_pose_set_inference_config(&config, nullptr, 0) == 1);

  NvDsYoloPoseInferenceConfig stored{};
  nvds_yolo_pose_get_inference_config(&stored);
  assert(stored.mode == NVDS_YOLO_POSE_INFERENCE_SINGLE_PASS);
  assert(std::fabs(stored.merge_iou_threshold - 0.3f) < 1e-6f);
}

static void assert_set_sliding_window_config()
{
  NvDsYoloPoseInferenceConfig config{};
  config.mode = NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW;
  config.window_width = 640;
  config.window_height = 640;
  config.stride_x = 480;
  config.stride_y = 480;
  config.merge_iou_threshold = 0.4f;

  assert(nvds_yolo_pose_set_inference_config(&config, nullptr, 0) == 1);

  NvDsYoloPoseInferenceConfig stored{};
  nvds_yolo_pose_get_inference_config(&stored);
  assert(stored.mode == NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW);
  assert(stored.window_width == 640);
  assert(stored.window_height == 640);
  assert(stored.stride_x == 480);
  assert(stored.stride_y == 480);
  assert(std::fabs(stored.merge_iou_threshold - 0.4f) < 1e-6f);
}

static void assert_reject_invalid_config()
{
  NvDsYoloPoseInferenceConfig config{};
  config.mode = NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW;
  config.window_width = 0;
  config.window_height = 640;
  config.stride_x = 480;
  config.stride_y = 480;
  config.merge_iou_threshold = 0.4f;

  char error[128] = {};
  assert(nvds_yolo_pose_set_inference_config(&config, error, sizeof(error)) == 0);
  assert(std::strlen(error) > 0);
}

int main()
{
  nvds_yolo_pose_reset_inference_config();
  assert_default_config();
  assert_set_single_pass_config();
  assert_set_sliding_window_config();
  assert_reject_invalid_config();
  std::cout << "pose inference config defaults passed\n";
  return 0;
}
