CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CXX:= g++

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_pose.so

CFLAGS:= -Wall -std=c++17 -shared -fPIC -Wno-error=deprecated-declarations

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include $(EXTRA_CUDA_INCLUDE)

ifeq ($(EXTRA_CUDART_LIB),)
	CUDART_LINK:= -lcudart
else
	CUDART_LINK:= $(EXTRA_CUDART_LIB)
endif

LIBS+= -lnvinfer -L/usr/local/cuda-$(CUDA_VER)/lib64 $(EXTRA_CUDA_LIB) $(CUDART_LINK) -lstdc++fs

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
