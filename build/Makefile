DIR := $(shell pwd)/..

TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_ROOT=/usr/local/cuda-9.0/

LINKFLAGS := -std=c++11 -arch=sm_61 -rdc=true --default-stream per-thread -O2
CUDAFLAGS := ${LINKFLAGS} -dc 
FLAGS := -std=c++11 -O2

NVCC := nvcc
CXX := g++

LDCUDAFLAGS := -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
LDLINKFLAGS:= -DGOOGLE_CUDA=1 -lcudadevrt -lcudart -shared 
LDFLAGS := -shared -fPIC -I ${TF_INC} -I ${CUDA_ROOT}/include -I ${TF_INC}/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_LIB} -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0

SOURCES := $(DIR)/recurrent.cc
LINKSOURCES := $(DIR)/file_link.o
CUDASOURCE := $(DIR)/cuda_op_recurrent.cu

all: librecurrent.so

librecurrent.so:
	${NVCC} $(CUDAFLAGS) $(CUDASOURCE) -o  $(CUDASOURCE).o $(LDCUDAFLAGS)
	${NVCC} $(LINKFLAGS) $(CUDASOURCE).o $(LDLINKFLAGS) -o $(LINKSOURCES)
	$(CXX) $(FLAGS) $(SOURCES) $(LINKSOURCES)  -o librecurrent.so $(LDFLAGS)

clean:
	rm -rf librecurrent.so

remake: clean all