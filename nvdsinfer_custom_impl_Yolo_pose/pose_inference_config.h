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

void nvds_yolo_pose_reset_inference_config(void);
void nvds_yolo_pose_get_inference_config(NvDsYoloPoseInferenceConfig *config);
int nvds_yolo_pose_set_inference_config(const NvDsYoloPoseInferenceConfig *config,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
