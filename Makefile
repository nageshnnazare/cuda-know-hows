# ============================================================================
# CUDA Tutorial Makefile
# ============================================================================
# 
# This Makefile builds all CUDA tutorial examples from basic to advanced.
#
# Usage:
#   make all          - Build all examples
#   make clean        - Remove all build artifacts
#   make run          - Run all examples
#   make debug        - Build with debug symbols
#   make <target>     - Build specific example (e.g., make 02_first_kernel)
#
# Requirements:
#   - CUDA Toolkit (nvcc compiler)
#   - cuBLAS library (for matrix operations)
#
# ============================================================================

# Compiler and flags
NVCC := nvcc
CUDA_PATH ?= /usr/local/cuda

# Architecture flags - adjust for your GPU
# Find your compute capability: nvidia-smi --query-gpu=compute_cap --format=csv
# Common values:
#   sm_35 - Kepler (K20, K40)
#   sm_52 - Maxwell (GTX 9xx)
#   sm_61 - Pascal (GTX 10xx, Titan X)
#   sm_70 - Volta (V100, Titan V)
#   sm_75 - Turing (RTX 20xx, T4)
#   sm_80 - Ampere (A100, RTX 30xx)
#   sm_89 - Ada (RTX 40xx)
ARCH := sm_75

# Compiler flags
NVCCFLAGS := -arch=$(ARCH)
NVCCFLAGS += -std=c++11
NVCCFLAGS += -O3                      # Optimization level
NVCCFLAGS += --use_fast_math          # Use fast math approximations
NVCCFLAGS += -Xcompiler -Wall         # Enable all warnings
NVCCFLAGS += -Xcompiler -Wextra       # Extra warnings
NVCCFLAGS += -lineinfo                # Line info for profiler

# Debug flags (use with: make debug)
DEBUG_FLAGS := -g -G                  # Debug symbols for host and device
DEBUG_FLAGS += -O0                    # No optimization
DEBUG_FLAGS += --ptxas-options=-v     # Verbose PTX assembly

# Include paths
INCLUDES := -I$(CUDA_PATH)/include

# Library paths
LDFLAGS := -L$(CUDA_PATH)/lib64
LIBS := -lcudart

# Libraries for specific examples
CUBLAS_LIBS := -lcublas
NVTX_LIBS := -lnvToolsExt
COOP_FLAGS := -rdc=true -lcudadevrt

# Source files
SOURCES := $(wildcard *.cu)

# Executables (remove .cu extension)
EXECUTABLES := $(basename $(SOURCES))

# Special executables that need extra libraries
MATRIX_OPS := 05_matrix_operations
ADVANCED := 08_advanced_topics
SCIENTIFIC := 14_scientific_computing
DEEP_LEARNING := 21_deep_learning
IMAGE_PROCESSING := 12_image_processing
SORTING := 13_sorting_algorithms

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean run debug help check-cuda info

# Default target
all: check-cuda $(EXECUTABLES)
	@echo "$(GREEN)✓ All examples built successfully!$(NC)"
	@echo ""
	@echo "Run examples with:"
	@echo "  ./02_first_kernel"
	@echo "  ./03_memory_model"
	@echo "  etc."
	@echo ""
	@echo "Or run all: make run"

# Check CUDA installation
check-cuda:
	@echo "$(BLUE)Checking CUDA installation...$(NC)"
	@which nvcc > /dev/null || (echo "$(RED)Error: nvcc not found. Please install CUDA Toolkit.$(NC)" && exit 1)
	@which nvidia-smi > /dev/null || (echo "$(YELLOW)Warning: nvidia-smi not found. GPU may not be available.$(NC)")
	@echo "$(GREEN)✓ CUDA found$(NC)"
	@nvcc --version | head -n 1

# Build regular examples
02_first_kernel: 02_first_kernel.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

03_memory_model: 03_memory_model.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

04_thread_organization: 04_thread_organization.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

# Matrix operations needs cuBLAS
05_matrix_operations: 05_matrix_operations.cu
	@echo "$(BLUE)Building $@ (with cuBLAS)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS) $(CUBLAS_LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

06_shared_memory: 06_shared_memory.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

07_streams_async: 07_streams_async.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

# Advanced topics needs dynamic parallelism and cooperative groups
08_advanced_topics: 08_advanced_topics.cu
	@echo "$(BLUE)Building $@ (with dynamic parallelism)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(COOP_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"
	@echo "$(YELLOW)Note: Dynamic parallelism requires compute capability >= 3.5$(NC)"

# New practical examples
12_image_processing: 12_image_processing.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

13_sorting_algorithms: 13_sorting_algorithms.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

14_scientific_computing: 14_scientific_computing.cu
	@echo "$(BLUE)Building $@ (with cuRAND)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS) -lcurand
	@echo "$(GREEN)✓ Built $@$(NC)"

16_graph_algorithms: 16_graph_algorithms.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

17_advanced_memory: 17_advanced_memory.cu
	@echo "$(BLUE)Building $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

18_ml_primitives: 18_ml_primitives.cu
	@echo "$(BLUE)Building $@ (with cuBLAS & cuRAND)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS) $(CUBLAS_LIBS) -lcurand
	@echo "$(GREEN)✓ Built $@$(NC)"

19_multi_gpu: 19_multi_gpu.cu
	@echo "$(BLUE)Building $@ (Multi-GPU programming)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

21_deep_learning: 21_deep_learning.cu
	@echo "$(BLUE)Building $@ (with cuRAND)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS) -lcurand
	@echo "$(GREEN)✓ Built $@$(NC)"

gpu_locks_and_synchronization: gpu_locks_and_synchronization.cu
	@echo "$(BLUE)Building $@ (locks & sync guide)...$(NC)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)
	@echo "$(GREEN)✓ Built $@$(NC)"

# Debug build
debug: NVCCFLAGS = $(DEBUG_FLAGS)
debug: clean all
	@echo "$(GREEN)✓ Debug build complete$(NC)"
	@echo "Use cuda-gdb to debug: cuda-gdb ./02_first_kernel"

# Clean build artifacts
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -f $(EXECUTABLES)
	@rm -f *.o
	@rm -f *.nsys-rep *.ncu-rep *.prof *.nvvp
	@rm -f *.log
	@echo "$(GREEN)✓ Clean complete$(NC)"

# Run all examples
run: all
	@echo ""
	@echo "$(BLUE)╔═══════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     Running All CUDA Tutorial Examples           ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@for exe in $(EXECUTABLES); do \
		if [ -f $$exe ]; then \
			echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"; \
			echo "$(GREEN)Running: $$exe$(NC)"; \
			echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"; \
			./$$exe || echo "$(RED)✗ $$exe failed$(NC)"; \
			echo ""; \
		fi \
	done
	@echo "$(GREEN)✓ All examples completed$(NC)"

# Profile with Nsight Systems
profile-%: %
	@echo "$(BLUE)Profiling $< with Nsight Systems...$(NC)"
	@which nsys > /dev/null || (echo "$(RED)Error: nsys not found. Install NVIDIA Nsight Systems.$(NC)" && exit 1)
	nsys profile -o $<_profile --stats=true ./$<
	@echo "$(GREEN)✓ Profile saved to $<_profile.nsys-rep$(NC)"
	@echo "View with: nsys-ui $<_profile.nsys-rep"

# Analyze with Nsight Compute
analyze-%: %
	@echo "$(BLUE)Analyzing $< with Nsight Compute...$(NC)"
	@which ncu > /dev/null || (echo "$(RED)Error: ncu not found. Install NVIDIA Nsight Compute.$(NC)" && exit 1)
	ncu --set full -o $<_analysis ./$<
	@echo "$(GREEN)✓ Analysis saved to $<_analysis.ncu-rep$(NC)"
	@echo "View with: ncu-ui $<_analysis.ncu-rep"

# Memory check
memcheck-%: %
	@echo "$(BLUE)Running cuda-memcheck on $<...$(NC)"
	@which cuda-memcheck > /dev/null || (echo "$(RED)Error: cuda-memcheck not found.$(NC)" && exit 1)
	cuda-memcheck --leak-check full ./$<

# Show device info
info:
	@echo "$(BLUE)╔═══════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║           CUDA Device Information                 ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@which nvidia-smi > /dev/null && nvidia-smi || echo "$(RED)nvidia-smi not available$(NC)"
	@echo ""
	@echo "$(BLUE)CUDA Toolkit:$(NC)"
	@nvcc --version
	@echo ""
	@echo "$(BLUE)Compute Architectures Supported:$(NC)"
	@nvcc --help | grep -A 20 "gpu-architecture" | grep sm_ || echo "Run 'nvcc --help' for details"

# Help target
help:
	@echo "$(BLUE)╔═══════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          CUDA Tutorial Makefile Help              ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Building:$(NC)"
	@echo "  make all              - Build all examples"
	@echo "  make <name>           - Build specific example"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make debug            - Build with debug symbols"
	@echo ""
	@echo "$(GREEN)Running:$(NC)"
	@echo "  make run              - Run all examples"
	@echo "  ./<example>           - Run specific example"
	@echo ""
	@echo "$(GREEN)Profiling & Analysis:$(NC)"
	@echo "  make profile-<name>   - Profile with Nsight Systems"
	@echo "  make analyze-<name>   - Analyze with Nsight Compute"
	@echo "  make memcheck-<name>  - Check memory errors"
	@echo ""
	@echo "$(GREEN)Information:$(NC)"
	@echo "  make info             - Show CUDA device info"
	@echo "  make check-cuda       - Verify CUDA installation"
	@echo "  make help             - Show this help"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make 02_first_kernel"
	@echo "  make profile-03_memory_model"
	@echo "  make analyze-05_matrix_operations"
	@echo "  make memcheck-06_shared_memory"
	@echo ""
	@echo "$(YELLOW)Available Tutorials:$(NC)"
	@echo "  02_first_kernel                - Your first CUDA kernel"
	@echo "  03_memory_model                - Memory hierarchy"
	@echo "  04_thread_organization         - Thread indexing"
	@echo "  05_matrix_operations           - Matrix multiplication"
	@echo "  06_shared_memory               - Shared memory optimization"
	@echo "  07_streams_async               - Asynchronous operations"
	@echo "  08_advanced_topics             - Advanced features"
	@echo "  12_image_processing            - Convolution, edge detection"
	@echo "  13_sorting_algorithms          - Bitonic, radix sort"
	@echo "  14_scientific_computing        - Heat equation, N-body"
	@echo "  16_graph_algorithms            - BFS, PageRank, shortest paths"
	@echo "  17_advanced_memory             - Texture, zero-copy, unified mem"
	@echo "  18_ml_primitives               - GEMM, activations, batch norm"
	@echo "  19_multi_gpu                   - Multi-GPU, P2P, load balancing"
	@echo "  21_deep_learning               - Neural networks, CNNs"
	@echo "  gpu_locks_and_synchronization  - Atomics, locks, lock-free"
	@echo ""

# Detect architecture automatically (requires running GPU)
detect-arch:
	@echo "$(BLUE)Detecting GPU compute capability...$(NC)"
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | awk '{gsub(/\./,""); print "sm_" $$1}'
	@echo ""
	@echo "To use this architecture, edit the Makefile and set:"
	@echo "ARCH := $$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | awk '{gsub(/\./,""); print "sm_" $$1}')"

# Install check - verify all tools are available
install-check:
	@echo "$(BLUE)Checking installation...$(NC)"
	@echo ""
	@echo -n "CUDA Toolkit (nvcc):     "
	@which nvcc > /dev/null && echo "$(GREEN)✓ Found$(NC)" || echo "$(RED)✗ Not found$(NC)"
	@echo -n "NVIDIA Driver:           "
	@which nvidia-smi > /dev/null && echo "$(GREEN)✓ Found$(NC)" || echo "$(RED)✗ Not found$(NC)"
	@echo -n "cuda-memcheck:           "
	@which cuda-memcheck > /dev/null && echo "$(GREEN)✓ Found$(NC)" || echo "$(YELLOW)⚠ Not found$(NC)"
	@echo -n "cuda-gdb:                "
	@which cuda-gdb > /dev/null && echo "$(GREEN)✓ Found$(NC)" || echo "$(YELLOW)⚠ Not found$(NC)"
	@echo -n "Nsight Systems (nsys):   "
	@which nsys > /dev/null && echo "$(GREEN)✓ Found$(NC)" || echo "$(YELLOW)⚠ Not found$(NC)"
	@echo -n "Nsight Compute (ncu):    "
	@which ncu > /dev/null && echo "$(GREEN)✓ Found$(NC)" || echo "$(YELLOW)⚠ Not found$(NC)"
	@echo ""
	@echo "$(GREEN)✓ = Required and found$(NC)"
	@echo "$(YELLOW)⚠ = Optional, install for full functionality$(NC)"
	@echo "$(RED)✗ = Required but not found$(NC)"

# Test build - quick sanity check
test: 02_first_kernel
	@echo "$(BLUE)Running quick test...$(NC)"
	@./02_first_kernel > /dev/null && echo "$(GREEN)✓ Test passed$(NC)" || echo "$(RED)✗ Test failed$(NC)"

# ============================================================================
# Special targets
# ============================================================================

.PHONY: examples list

# List all examples
list:
	@echo "$(BLUE)Available examples:$(NC)"
	@echo ""
	@for src in $(SOURCES); do \
		echo "  - $$(basename $$src .cu)"; \
	done

# Build only specific category
examples-basic: 02_first_kernel 03_memory_model 04_thread_organization
	@echo "$(GREEN)✓ Basic examples built$(NC)"

examples-intermediate: 05_matrix_operations 06_shared_memory
	@echo "$(GREEN)✓ Intermediate examples built$(NC)"

examples-advanced: 07_streams_async 08_advanced_topics
	@echo "$(GREEN)✓ Advanced examples built$(NC)"

examples-applications: 12_image_processing 13_sorting_algorithms 14_scientific_computing 16_graph_algorithms 17_advanced_memory 18_ml_primitives 19_multi_gpu 21_deep_learning gpu_locks_and_synchronization
	@echo "$(GREEN)✓ Application examples built$(NC)"

# ============================================================================
# Auto-generated dependencies (optional)
# ============================================================================

# Automatically detect source file changes
-include $(SOURCES:.cu=.d)

%.d: %.cu
	@$(NVCC) -M $(NVCCFLAGS) $(INCLUDES) $< | sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' > $@

# ============================================================================
# Notes
# ============================================================================
#
# Compute Capability Reference:
# - Check your GPU: nvidia-smi --query-gpu=name,compute_cap --format=csv
# - Update ARCH variable for your GPU
#
# Profiling:
# - Use 'make profile-<example>' for timeline view
# - Use 'make analyze-<example>' for kernel details
# - Results viewable in Nsight Systems/Compute GUI
#
# Debugging:
# - Build with 'make debug'
# - Use cuda-gdb: cuda-gdb ./example
# - Or use 'make memcheck-<example>'
#
# For more information, see README.md
#
# ============================================================================

