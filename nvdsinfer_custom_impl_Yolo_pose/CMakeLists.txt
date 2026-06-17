# Wrap upstream Makefile

set(YOLO_POSE_IMPL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/libnvdsinfer_custom_impl_Yolo_pose.so)

# Build target
add_custom_target(nvdsinfer_custom_impl_yolo_pose ALL
    COMMAND $(MAKE) -j${CMAKE_BUILD_PARALLEL_LEVEL}
        CUDA_VER=${CUDA_VER}
    # Copy output to the main build dir for when we're running without installing
    COMMAND ${CMAKE_COMMAND} -E copy ${YOLO_POSE_IMPL_LIB} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${YOLO_POSE_IMPL_LIB}
    USES_TERMINAL
    VERBATIM
)

# Install the built library
install(FILES ${YOLO_POSE_IMPL_LIB}
    DESTINATION lib
)

# Hook into CMake's "clean" target
set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_CLEAN_FILES
        ${YOLO_POSE_IMPL_LIB}
        "${CMAKE_CURRENT_SOURCE_DIR}/*.o"
)
