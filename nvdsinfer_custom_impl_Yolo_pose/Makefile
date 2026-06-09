CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CXX:= g++
CUDA_RUNTIME_LIB?=-lcudart

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_pose.so

CFLAGS:= -Wall -std=c++17 -shared -fPIC -Wno-error=deprecated-declarations

# Allow building with DEBUG=1 to enable debug symbols and disable optimizations
ifeq ($(DEBUG),1)
	CFLAGS += -g -O0
else
	CFLAGS += -O2
endif

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include $(EXTRA_CUDA_INCLUDE)

LIBS+= -lnvinfer -L/usr/local/cuda-$(CUDA_VER)/lib64 $(CUDA_RUNTIME_LIB) -lstdc++fs

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
