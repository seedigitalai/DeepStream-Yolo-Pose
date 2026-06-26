#include "pose_inference_config.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

static std::string read_file(const char *path)
{
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void assert_write_preprocess_config()
{
  const char *path = "/tmp/yolo_pose_preprocess_test.txt";
  NvDsYoloPosePreprocessConfig config{};
  config.output_path = path;
  config.frame_width = 1280;
  config.frame_height = 720;
  config.source_count = 2;
  config.gpu_id = 0;
  config.preprocess_unique_id = 5;
  config.target_gie_unique_id = 1;
  config.tensor_name = "input";
  config.custom_lib_path = "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so";
  config.inference.mode = NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW;
  config.inference.window_width = 640;
  config.inference.window_height = 640;
  config.inference.stride_x = 480;
  config.inference.stride_y = 480;
  config.inference.merge_iou_threshold = 0.45f;

  char error[256] = {};
  int tensor_batch_size = 0;
  assert(nvds_yolo_pose_preprocess_tensor_batch_size(
      &config, &tensor_batch_size, error, sizeof(error)) == 1);
  assert(tensor_batch_size == 12);
  assert(nvds_yolo_pose_write_preprocess_config(&config, error, sizeof(error)) == 1);

  const std::string body = read_file(path);
  assert(body.find("target-unique-ids=1") != std::string::npos);
  assert(body.find("network-input-shape=12;3;640;640") != std::string::npos);
  assert(body.find("tensor-name=input") != std::string::npos);
  assert(body.find("src-ids=0;1") != std::string::npos);
  assert(body.find("roi-params-src-0=0;0;640;640;480;0;640;640;640;0;640;640;0;80;640;640;480;80;640;640;640;80;640;640") != std::string::npos);
  assert(body.find("roi-params-src-1=0;0;640;640;480;0;640;640;640;0;640;640;0;80;640;640;480;80;640;640;640;80;640;640") != std::string::npos);
  std::remove(path);
}

static void assert_remap_keypoints_to_frame()
{
  float mask[] = {
    320.0f, 160.0f, 0.9f,
    480.0f, 640.0f, 0.8f,
  };
  unsigned mask_width = 640;
  unsigned mask_height = 640;

  NvDsYoloPoseRoiTransform transform{};
  transform.roi_left = 480.0;
  transform.roi_top = 80.0;
  transform.roi_width = 640.0;
  transform.roi_height = 640.0;
  transform.scale_ratio_x = 1.0;
  transform.scale_ratio_y = 1.0;
  transform.offset_left = 0.0;
  transform.offset_top = 0.0;
  transform.frame_width = 1280;
  transform.frame_height = 720;

  char error[128] = {};
  assert(nvds_yolo_pose_remap_keypoints_to_frame(
      mask, sizeof(mask), &mask_width, &mask_height, &transform,
      error, sizeof(error)) == 1);
  assert(std::fabs(mask[0] - 800.0f) < 1e-6f);
  assert(std::fabs(mask[1] - 240.0f) < 1e-6f);
  assert(std::fabs(mask[3] - 960.0f) < 1e-6f);
  assert(std::fabs(mask[4] - 720.0f) < 1e-6f);
  assert(mask_width == 1280);
  assert(mask_height == 720);
}

static void assert_duplicate_suppression()
{
  NvDsYoloPoseObjectBox kept{};
  kept.left = 100.0f;
  kept.top = 100.0f;
  kept.width = 100.0f;
  kept.height = 100.0f;
  kept.confidence = 0.9f;
  kept.class_id = 0;

  NvDsYoloPoseObjectBox duplicate = kept;
  duplicate.left = 110.0f;
  duplicate.top = 110.0f;
  duplicate.confidence = 0.8f;

  NvDsYoloPoseObjectBox separate = kept;
  separate.left = 260.0f;

  assert(nvds_yolo_pose_box_iou(&duplicate, &kept) > 0.45f);
  assert(nvds_yolo_pose_should_suppress_duplicate(&duplicate, &kept, 0.45f) == 1);
  assert(nvds_yolo_pose_should_suppress_duplicate(&separate, &kept, 0.45f) == 0);
}

int main()
{
  nvds_yolo_pose_reset_inference_config();
  assert_default_config();
  assert_set_single_pass_config();
  assert_set_sliding_window_config();
  assert_reject_invalid_config();
  assert_write_preprocess_config();
  assert_remap_keypoints_to_frame();
  assert_duplicate_suppression();
  std::cout << "pose inference config defaults passed\n";
  return 0;
}
