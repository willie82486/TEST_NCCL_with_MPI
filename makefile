# Example compile command (adjust paths as needed):
# mpicxx -I/path/to/nccl/include -L/path/to/nccl/lib -lnccl -lcudart -o test_nccl test_nccl.cpp

# Run on 4 nodes, 8 GPUs each => total 32 ranks
# mpirun -np 32 -x NCCL_DEBUG=INFO -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./test_nccl

# Compiler
CXX = mpicxx


# Paths to NCCL
NCCL_PATH = /usr/local/cuda
INCLUDE = $(NCCL_PATH)/include
LIB_PATH = $(NCCL_PATH)/lib64

# Compiler and linker flags
CXXFLAGS = -O3 -D_GNU_SOURCE -lm -fopenmp -I$(INCLUDE)
LDFLAGS = -L$(LIB_PATH) -lnccl -lcudart

# Source files
SRCS := $(wildcard *.cpp)

# Object files
OBJS = $(patsubst %.cpp, obj/%.o, $(SRCS))

# Executable name
TARGET = test_nccl

.PHONY: all clean run

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Rule to compile the source files
obj/%.o: %.cpp
	@mkdir -p obj
	$(CXX) $(CXXFLAGS) -I $(INCLUDE) -c $< -o $@

# Clean rule
clean:
	rm -rf obj/*.o $(TARGET)

# Run the program
run:
# rsync -av ~/test_nccl node1:~/
# mpirun -np 4 -x NCCL_DEBUG=INFO -x CUDA_VISIBLE_DEVICES=0,1,2,3 ./$(TARGET)
	mpirun -np 4 -bind-to none --map-by ppr:2:node --hostfile Hostfile ./$(TARGET)
