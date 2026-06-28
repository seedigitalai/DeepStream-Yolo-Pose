# Wrap upstream Makefile

set(RTMPOSE_IMPL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/libnvdsinfer_custom_impl_Rtmpose.so)

# Build target
add_custom_target(nvdsinfer_custom_impl_rtmpose ALL
    COMMAND $(MAKE) -j${CMAKE_BUILD_PARALLEL_LEVEL}
        CUDA_VER="${CUDA_VER}"
    # Copy output to the main build dir for when we're running without installing
    COMMAND ${CMAKE_COMMAND} -E copy ${RTMPOSE_IMPL_LIB} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS ${RTMPOSE_IMPL_LIB}
    USES_TERMINAL
    VERBATIM
)

# Install the built library
install(FILES ${RTMPOSE_IMPL_LIB}
    DESTINATION lib
)

# Hook into CMake's "clean" target
set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_CLEAN_FILES
        ${RTMPOSE_IMPL_LIB}
        "${CMAKE_CURRENT_SOURCE_DIR}/*.o"
)
