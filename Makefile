.PHONY: all build train play test clean deps

# Default target
all: build

# Install dependencies
deps:
	go mod tidy

# Build both binaries
build: deps
	go build -o bin/train ./cmd/train
	go build -o bin/play ./cmd/play

# Run training (headless)
train: build
	./bin/train -episodes=5000 -log-freq=100 -save-freq=500

# Run training with custom parameters
train-quick: build
	./bin/train -episodes=1000 -log-freq=50 -save-freq=200

train-long: build
	./bin/train -episodes=20000 -log-freq=500 -save-freq=1000

# Run visual game with trained model
play: build
	./bin/play -model=models/snake_dqn.gob

# Run visual game with random actions (no model needed)
play-random: build
	./bin/play -random

# Run tests
test:
	go test -v ./...

# Run tests with coverage
test-cover:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f coverage.out coverage.html

# Clean everything including models
clean-all: clean
	rm -rf models/

# Format code
fmt:
	go fmt ./...

# Lint code
lint:
	go vet ./...

# Show help
help:
	@echo "Autonomous Snake Game - Makefile targets:"
	@echo ""
	@echo "  make deps       - Install/update dependencies"
	@echo "  make build      - Build train and play binaries"
	@echo "  make train      - Train for 5000 episodes"
	@echo "  make train-quick - Train for 1000 episodes (quick test)"
	@echo "  make train-long - Train for 20000 episodes"
	@echo "  make play       - Watch trained agents play"
	@echo "  make play-random - Watch random agents play"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make clean-all  - Remove build artifacts and models"
	@echo ""
	@echo "Training flags (use with go run):"
	@echo "  -episodes N     - Number of training episodes"
	@echo "  -model PATH     - Model save/load path"
	@echo "  -board N        - Board size (width and height)"
	@echo "  -seed N         - Random seed"
	@echo ""
	@echo "Play flags:"
	@echo "  -model PATH     - Model to load"
	@echo "  -random         - Run without model (random actions)"
	@echo "  -grid N         - Cell size in pixels"
