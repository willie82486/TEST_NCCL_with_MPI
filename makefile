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
	mpirun -np 16 -bind-to none --map-by ppr:8:node --hostfile Hostfile ./$(TARGET)
