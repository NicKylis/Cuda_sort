# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -O2

# Version-specific directories and source files
V0_DIR = v0
V1_DIR = v1
V2_DIR = v2

# Version-specific source files
V0_SRCS = $(V0_DIR)/main.cu $(V0_DIR)/v0.cu
V1_SRCS = $(V1_DIR)/main.cu $(V1_DIR)/v1.cu
V2_SRCS = $(V2_DIR)/main.cu $(V2_DIR)/v2.cu

# Output binary name
TARGET = main

# Default rule (build v2 if no version is specified)
all: v2

# Version-specific build rules
v0: $(V0_SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $^

v1: $(V1_SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $^

v2: $(V2_SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $^

# Clean up
clean:
	rm -f $(TARGET)

# Help command
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  v0      Build version 0 (uses v0.cu and v0.h)"
	@echo "  v1      Build version 1 (uses v1.cu and v1.h)"
	@echo "  v2      Build version 2 (uses v2.cu and v2.h) [default]"
	@echo "  clean   Remove the generated 'main' binary"
	@echo ""
	@echo "If no target is specified, 'make' defaults to 'v2'."

# Phony targets
.PHONY: all clean v0 v1 v2 help
