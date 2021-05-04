
EXECUTABLE := sssp

INC := input_graph.h

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=bin
SRCDIR=src
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++11
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

SRCS := $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/*.cu)
OBJS := $(patsubst $(SRCDIR)/%, $(OBJDIR)/%.o, $(basename $(SRCS)))


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/$(INC)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
