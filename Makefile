GPU=0
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=0
NNPACK=0
RISCV=0
VERSION=0
LLVM_CONFIG=/export/home4/np106j/llvm-EPI-development-toolchain-cross/bin/llvm-config

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
ifeq ($(RISCV), 1)
	EXEC=darknet.riscv1.0
endif
ifeq ($(VERSION), 07)
	EXEC=darknet.riscv0.7
endif
OBJDIR=./obj/

CC=gcc
CPP=g++
ifeq ($(RISCV), 1)
LLVM_CONFIG=/export/home4/np106j/llvm-EPI-development-toolchain-cross/bin/llvm-config
ifeq ($(VERSION), 07)
LLVM_CONFIG=/export/home4/np106j/llvm-EPI-0.7-development-toolchain-cross/bin/llvm-config
endif
CC=$(shell $(LLVM_CONFIG) --bindir)/clang
CPP=$(shell $(LLVM_CONFIG) --bindir)/clang++
endif
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
CFLAGS=-O3 -Wall 
COMMON=-Iinclude/ -Isrc/
LIBS=-lm -lpthread
LDFLAGS=
OPTS=
ifeq ($(RISCV), 1)
OPTS=-DRISCV
LDFLAGS= $(shell $(LLVM_CONFIG) --ldflags)  
LIBS=-lm -lpthread
COMMON= -Iinclude/ -Isrc/ -I$(shell $(LLVM_CONFIG) --includedir)
CFLAGS= $(shell $(LLVM_CONFIG) --cflags) --target=riscv64-unknown-linux-gnu -march=rv64g -mcmodel=medany -O3 -static   -fsave-optimization-record -DUSE_RISCV_VECTOR -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -mepi   -v -fno-vectorize -fno-slp-vectorize  
endif

ifeq ($(NNPACK), 1)
CFLAGS += -DNNPACK
endif
ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)


ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o winograd.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o  list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o 
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h 

#all: obj backup results  $(EXEC)
#all: obj backup results  
#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj backup results  $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC)  $(CFLAGS) $(COMMON) $^ -o $@ $(ALIB) $(LDFLAGS) $(LIBS) -v

#	$(CC) $(CFLAGS) $(COMMON) -o $@ $^ $(LDFLAGS) $(ALIB) $(LIBS) -v
#	$(CC)  $(CFLAGS) $(COMMON) $^ ../NNPACKRISCV/lib.a -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

#$(SLIB): $(OBJS)
#	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(CFLAGS) $(COMMON) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(COMMON) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean distclean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ)
distclean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)


