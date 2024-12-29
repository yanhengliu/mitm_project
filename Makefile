# Compiler and flags
CC = mpicc
CFLAGS = -O2 -Wall -std=c99 -Iinclude
LDFLAGS = -lm

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))
BIN_FILE = $(BIN_DIR)/mitm

# Default target
all: $(BIN_FILE)

# Build the main program
$(BIN_FILE): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run tests
test:
	$(MAKE) -C $(TEST_DIR)

# Phony targets
.PHONY: all clean test
