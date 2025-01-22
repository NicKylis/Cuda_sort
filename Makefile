# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -O2

# Common source files
COMMON_SRCS = main.cu

# Version-specific source files
VERSION_SRCS_v0 = v0.cu
VERSION_SRCS_v1 = v1.cu
VERSION_SRCS_v2 = v2.cu

# Object files (version-specific will be appended dynamically)
OBJS = main.o

# Output binary name
TARGET = main

# Default rule
all:
	@echo "Specify a version to build, e.g., 'make v0'"

# Version-specific build rules
v0: $(OBJS) v0.o
	$(NVCC) $(CFLAGS) -DVERSION=0 -o $(TARGET) $(OBJS) v0.o

v1: $(OBJS) v1.o
	$(NVCC) $(CFLAGS) -DVERSION=1 -o $(TARGET) $(OBJS) v1.o

v2: $(OBJS) v2.o
	$(NVCC) $(CFLAGS) -DVERSION=2 -o $(TARGET) $(OBJS) v2.o

# Compilation step for common and version-specific sources
main.o: main.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

v0.o: v0.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

v1.o: v1.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

v2.o: v2.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f *.o $(TARGET)

# Phony targets
.PHONY: all clean v0 v1 v2
