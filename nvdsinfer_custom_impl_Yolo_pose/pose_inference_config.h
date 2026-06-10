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

typedef struct {
  double roi_left;
  double roi_top;
  double roi_width;
  double roi_height;
  double scale_ratio_x;
  double scale_ratio_y;
  double offset_left;
  double offset_top;
  unsigned frame_width;
  unsigned frame_height;
} NvDsYoloPoseRoiTransform;

typedef struct {
  float left;
  float top;
  float width;
  float height;
  float confidence;
  int class_id;
} NvDsYoloPoseObjectBox;

void nvds_yolo_pose_reset_inference_config(void);
void nvds_yolo_pose_get_inference_config(NvDsYoloPoseInferenceConfig *config);
int nvds_yolo_pose_set_inference_config(const NvDsYoloPoseInferenceConfig *config,
    char *error, size_t error_size);
int nvds_yolo_pose_write_preprocess_config(const NvDsYoloPosePreprocessConfig *config,
    char *error, size_t error_size);
int nvds_yolo_pose_preprocess_tensor_batch_size(const NvDsYoloPosePreprocessConfig *config,
    int *tensor_batch_size, char *error, size_t error_size);
int nvds_yolo_pose_remap_keypoints_to_frame(float *mask, size_t mask_size,
    unsigned *mask_width, unsigned *mask_height,
    const NvDsYoloPoseRoiTransform *transform, char *error, size_t error_size);
float nvds_yolo_pose_box_iou(const NvDsYoloPoseObjectBox *a,
    const NvDsYoloPoseObjectBox *b);
int nvds_yolo_pose_should_suppress_duplicate(const NvDsYoloPoseObjectBox *candidate,
    const NvDsYoloPoseObjectBox *kept, float merge_iou_threshold);

#ifdef __cplusplus
}
#endif

#endif
