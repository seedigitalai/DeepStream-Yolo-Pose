#ifndef POSE_INFERENCE_CONFIG_H
#define POSE_INFERENCE_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
  NVDS_YOLO_POSE_INFERENCE_SINGLE_PASS = 0,
  NVDS_YOLO_POSE_INFERENCE_SLIDING_WINDOW = 1,
} NvDsYoloPoseInferenceMode;

typedef struct {
  NvDsYoloPoseInferenceMode mode;
  int window_width;
  int window_height;
  int stride_x;
  int stride_y;
  float merge_iou_threshold;
} NvDsYoloPoseInferenceConfig;

typedef struct {
  const char *output_path;
  int frame_width;
  int frame_height;
  int source_count;
  int gpu_id;
  int preprocess_unique_id;
  int target_gie_unique_id;
  const char *tensor_name;
  const char *custom_lib_path;
  NvDsYoloPoseInferenceConfig inference;
} NvDsYoloPosePreprocessConfig;

void nvds_yolo_pose_reset_inference_config(void);
void nvds_yolo_pose_get_inference_config(NvDsYoloPoseInferenceConfig *config);
int nvds_yolo_pose_set_inference_config(const NvDsYoloPoseInferenceConfig *config,
    char *error, size_t error_size);
int nvds_yolo_pose_write_preprocess_config(const NvDsYoloPosePreprocessConfig *config,
    char *error, size_t error_size);
int nvds_yolo_pose_preprocess_tensor_batch_size(const NvDsYoloPosePreprocessConfig *config,
    int *tensor_batch_size, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
