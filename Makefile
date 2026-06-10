CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CFLAGS:= -Wall -Wfatal-errors -fPIC -Wno-deprecated-declarations

APP:= deepstream

DS_SDK_ROOT:= /opt/nvidia/deepstream/deepstream

LIB_INSTALL_DIR?= $(DS_SDK_ROOT)/lib/

CUDA_INCLUDE_DIRS:= \
	/usr/local/cuda-$(CUDA_VER)/include \
	/usr/local/cuda-$(CUDA_VER)/targets/x86_64-linux/include

ifneq ($(strip $(EXTRA_CUDA_INCLUDE)),)
CUDA_INCLUDE_DIRS+= $(EXTRA_CUDA_INCLUDE)
endif

CUDA_LIB_DIRS:= \
	/usr/local/cuda-$(CUDA_VER)/lib64 \
	/usr/local/cuda-$(CUDA_VER)/targets/x86_64-linux/lib

ifneq ($(strip $(EXTRA_CUDA_LIB_DIRS)),)
CUDA_LIB_DIRS+= $(EXTRA_CUDA_LIB_DIRS)
endif

CUDA_RUNTIME_LIB:=$(firstword $(wildcard $(addsuffix /libcudart.so,$(CUDA_LIB_DIRS))) \
	$(wildcard $(addsuffix /libcudart.so.12,$(CUDA_LIB_DIRS))))

TARGET_DEVICE= $(shell gcc -dumpmachine | cut -f1 -d -)
ifeq ($(TARGET_DEVICE), aarch64)
	CFLAGS+= -DPLATFORM_TEGRA
endif

SRCS:= $(wildcard *.c)
SRCS+= $(wildcard modules/*.c)

INCS:= $(wildcard *.h)
INCS+= $(wildcard modules/*.h)

PKGS:= gstreamer-1.0

OBJS:= $(addsuffix .o, $(basename $(SRCS)))

CFLAGS+= -I$(DS_SDK_ROOT)/sources/apps/apps-common/includes -I$(DS_SDK_ROOT)/sources/includes \
	     $(addprefix -I,$(CUDA_INCLUDE_DIRS))

CFLAGS+= `pkg-config --cflags $(PKGS)`
LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper $(addprefix -L,$(CUDA_LIB_DIRS)) $(or $(CUDA_RUNTIME_LIB),-lcudart) \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)
