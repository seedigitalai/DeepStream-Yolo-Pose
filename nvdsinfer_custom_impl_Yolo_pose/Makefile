CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CXX:= g++

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_pose.so

CFLAGS:= -Wall -std=c++17 -shared -fPIC -Wno-error=deprecated-declarations

CUDA_INCLUDE_DIRS:= \
	/usr/local/cuda-$(CUDA_VER)/include \
	/usr/local/cuda-$(CUDA_VER)/targets/x86_64-linux/include \
	$(wildcard $(HOME)/.local/lib/python*/site-packages/nvidia/cuda_runtime/include) \
	$(wildcard $(HOME)/.local/lib/python*/site-packages/nvidia/cuda_nvcc/include)

CUDA_LIB_DIRS:= \
	/usr/local/cuda-$(CUDA_VER)/lib64 \
	/usr/local/cuda-$(CUDA_VER)/targets/x86_64-linux/lib \
	$(wildcard $(HOME)/.local/lib/python*/site-packages/nvidia/cuda_runtime/lib)

CUDA_RUNTIME_LIB:=$(firstword $(wildcard $(addsuffix /libcudart.so,$(CUDA_LIB_DIRS))) \
	$(wildcard $(addsuffix /libcudart.so.12,$(CUDA_LIB_DIRS))))

# Allow building with DEBUG=1 to enable debug symbols and disable optimizations
ifeq ($(DEBUG),1)
	CFLAGS += -g -O0
else
	CFLAGS += -O2
endif

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes $(addprefix -I,$(CUDA_INCLUDE_DIRS))

LIBS+= -lnvinfer $(addprefix -L,$(CUDA_LIB_DIRS)) $(or $(CUDA_RUNTIME_LIB),-lcudart) -lstdc++fs

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

SRCS:= $(wildcard *.cpp)

OBJS:= $(addsuffix .o, $(basename $(SRCS)))

all: $(TARGET_LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(TARGET_LIB) : $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LFLAGS)

clean:
	rm -rf $(OBJS) $(TARGET_LIB)
