package ai

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
)

// QNetwork represents a feedforward neural network for Q-value estimation
type QNetwork struct {
	// Layer 1: Input -> Hidden1
	W1 [][]float64 // [inputSize][hiddenSize1]
	B1 []float64   // [hiddenSize1]

	// Layer 2: Hidden1 -> Hidden2
	W2 [][]float64 // [hiddenSize1][hiddenSize2]
	B2 []float64   // [hiddenSize2]

	// Layer 3: Hidden2 -> Output
	W3 [][]float64 // [hiddenSize2][outputSize]
	B3 []float64   // [outputSize]

	// Dimensions
	InputSize   int
	HiddenSize1 int
	HiddenSize2 int
	OutputSize  int

	// Learning rate
	LearningRate float64

	// RNG for initialization
	rng *rand.Rand
}

// NewQNetwork creates a new neural network with Xavier initialization
func NewQNetwork(inputSize, hiddenSize1, hiddenSize2, outputSize int, lr float64, seed int64) *QNetwork {
	rng := rand.New(rand.NewSource(seed))
	net := &QNetwork{
		InputSize:    inputSize,
		HiddenSize1:  hiddenSize1,
		HiddenSize2:  hiddenSize2,
		OutputSize:   outputSize,
		LearningRate: lr,
		rng:          rng,
	}

	// Initialize weights with Xavier initialization
	net.W1 = xavierInit(inputSize, hiddenSize1, rng)
	net.B1 = make([]float64, hiddenSize1)

	net.W2 = xavierInit(hiddenSize1, hiddenSize2, rng)
	net.B2 = make([]float64, hiddenSize2)

	net.W3 = xavierInit(hiddenSize2, outputSize, rng)
	net.B3 = make([]float64, outputSize)

	return net
}

// xavierInit initializes weights using Xavier/Glorot initialization
func xavierInit(fanIn, fanOut int, rng *rand.Rand) [][]float64 {
	stddev := math.Sqrt(2.0 / float64(fanIn+fanOut))
	weights := make([][]float64, fanIn)
	for i := 0; i < fanIn; i++ {
		weights[i] = make([]float64, fanOut)
		for j := 0; j < fanOut; j++ {
			weights[i][j] = rng.NormFloat64() * stddev
		}
	}
	return weights
}

// Forward performs a forward pass through the network
func (n *QNetwork) Forward(input []float64) []float64 {
	// Layer 1: input -> hidden1 with ReLU
	h1 := n.linearForward(input, n.W1, n.B1)
	h1 = relu(h1)

	// Layer 2: hidden1 -> hidden2 with ReLU
	h2 := n.linearForward(h1, n.W2, n.B2)
	h2 = relu(h2)

	// Layer 3: hidden2 -> output (no activation for Q-values)
	output := n.linearForward(h2, n.W3, n.B3)

	return output
}

// ForwardWithCache performs forward pass and caches activations for backprop
func (n *QNetwork) ForwardWithCache(input []float64) ([]float64, *forwardCache) {
	cache := &forwardCache{
		input: make([]float64, len(input)),
	}
	copy(cache.input, input)

	// Layer 1
	z1 := n.linearForward(input, n.W1, n.B1)
	cache.z1 = z1
	h1 := relu(z1)
	cache.h1 = h1

	// Layer 2
	z2 := n.linearForward(h1, n.W2, n.B2)
	cache.z2 = z2
	h2 := relu(z2)
	cache.h2 = h2

	// Layer 3
	output := n.linearForward(h2, n.W3, n.B3)

	return output, cache
}

type forwardCache struct {
	input    []float64
	z1, h1   []float64
	z2, h2   []float64
}

// linearForward computes y = xW + b
func (n *QNetwork) linearForward(input []float64, weights [][]float64, bias []float64) []float64 {
	outputSize := len(bias)
	output := make([]float64, outputSize)

	for j := 0; j < outputSize; j++ {
		sum := bias[j]
		for i := 0; i < len(input); i++ {
			sum += input[i] * weights[i][j]
		}
		output[j] = sum
	}

	return output
}

// relu applies ReLU activation
func relu(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			result[i] = v
		}
	}
	return result
}

// reluDerivative computes ReLU derivative
func reluDerivative(z []float64) []float64 {
	result := make([]float64, len(z))
	for i, v := range z {
		if v > 0 {
			result[i] = 1.0
		}
	}
	return result
}

// Backward performs backpropagation and updates weights
// target is the target Q-value for the taken action
func (n *QNetwork) Backward(cache *forwardCache, output []float64, targetAction int, targetQ float64) {
	// Compute output layer error (only for the target action)
	dOutput := make([]float64, n.OutputSize)
	dOutput[targetAction] = output[targetAction] - targetQ

	// Backprop through layer 3
	dH2 := n.linearBackward(cache.h2, n.W3, n.B3, dOutput, true)

	// Apply ReLU derivative
	dZ2 := elementMul(dH2, reluDerivative(cache.z2))

	// Backprop through layer 2
	dH1 := n.linearBackward(cache.h1, n.W2, n.B2, dZ2, true)

	// Apply ReLU derivative
	dZ1 := elementMul(dH1, reluDerivative(cache.z1))

	// Backprop through layer 1
	n.linearBackward(cache.input, n.W1, n.B1, dZ1, true)
}

// linearBackward computes gradients and updates weights
func (n *QNetwork) linearBackward(input []float64, weights [][]float64, bias []float64, dOutput []float64, update bool) []float64 {
	inputSize := len(input)
	outputSize := len(dOutput)

	// Compute gradient w.r.t. input
	dInput := make([]float64, inputSize)
	for i := 0; i < inputSize; i++ {
		for j := 0; j < outputSize; j++ {
			dInput[i] += weights[i][j] * dOutput[j]
		}
	}

	if update {
		// Update weights and biases
		lr := n.LearningRate
		for i := 0; i < inputSize; i++ {
			for j := 0; j < outputSize; j++ {
				weights[i][j] -= lr * input[i] * dOutput[j]
			}
		}
		for j := 0; j < outputSize; j++ {
			bias[j] -= lr * dOutput[j]
		}
	}

	return dInput
}

// elementMul performs element-wise multiplication
func elementMul(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// CopyFrom copies weights from another network
func (n *QNetwork) CopyFrom(other *QNetwork) {
	copyMatrix(n.W1, other.W1)
	copy(n.B1, other.B1)
	copyMatrix(n.W2, other.W2)
	copy(n.B2, other.B2)
	copyMatrix(n.W3, other.W3)
	copy(n.B3, other.B3)
}

// copyMatrix copies a 2D matrix
func copyMatrix(dst, src [][]float64) {
	for i := range src {
		copy(dst[i], src[i])
	}
}

// Clone creates a deep copy of the network
func (n *QNetwork) Clone() *QNetwork {
	clone := NewQNetwork(n.InputSize, n.HiddenSize1, n.HiddenSize2, n.OutputSize, n.LearningRate, 0)
	clone.CopyFrom(n)
	return clone
}

// NetworkWeights holds serializable network weights
type NetworkWeights struct {
	W1           [][]float64
	B1           []float64
	W2           [][]float64
	B2           []float64
	W3           [][]float64
	B3           []float64
	InputSize    int
	HiddenSize1  int
	HiddenSize2  int
	OutputSize   int
	LearningRate float64
}

// legacyNetworkWeights is the old format with unused 2D bias fields
type legacyNetworkWeights struct {
	W1, B1       [][]float64
	B1Vec        []float64
	W2, B2       [][]float64
	B2Vec        []float64
	W3, B3       [][]float64
	B3Vec        []float64
	InputSize    int
	HiddenSize1  int
	HiddenSize2  int
	OutputSize   int
	LearningRate float64
}

// Save saves the network weights to a file
func (n *QNetwork) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	weights := NetworkWeights{
		W1:           n.W1,
		B1:           n.B1,
		W2:           n.W2,
		B2:           n.B2,
		W3:           n.W3,
		B3:           n.B3,
		InputSize:    n.InputSize,
		HiddenSize1:  n.HiddenSize1,
		HiddenSize2:  n.HiddenSize2,
		OutputSize:   n.OutputSize,
		LearningRate: n.LearningRate,
	}

	encoder := gob.NewEncoder(file)
	return encoder.Encode(weights)
}

// LoadNetwork loads network weights from a file
// Supports both new and legacy formats for backward compatibility
func LoadNetwork(path string) (*QNetwork, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Try loading with new format first
	var weights NetworkWeights
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&weights); err != nil {
		// If that fails, try legacy format
		file.Seek(0, 0) // Reset file position
		var legacyWeights legacyNetworkWeights
		decoder = gob.NewDecoder(file)
		if err := decoder.Decode(&legacyWeights); err != nil {
			return nil, err
		}
		// Convert legacy format to new format
		weights = NetworkWeights{
			W1:           legacyWeights.W1,
			B1:           legacyWeights.B1Vec,
			W2:           legacyWeights.W2,
			B2:           legacyWeights.B2Vec,
			W3:           legacyWeights.W3,
			B3:           legacyWeights.B3Vec,
			InputSize:    legacyWeights.InputSize,
			HiddenSize1:  legacyWeights.HiddenSize1,
			HiddenSize2:  legacyWeights.HiddenSize2,
			OutputSize:   legacyWeights.OutputSize,
			LearningRate: legacyWeights.LearningRate,
		}
	}

	net := &QNetwork{
		W1:           weights.W1,
		B1:           weights.B1,
		W2:           weights.W2,
		B2:           weights.B2,
		W3:           weights.W3,
		B3:           weights.B3,
		InputSize:    weights.InputSize,
		HiddenSize1:  weights.HiddenSize1,
		HiddenSize2:  weights.HiddenSize2,
		OutputSize:   weights.OutputSize,
		LearningRate: weights.LearningRate,
		rng:          rand.New(rand.NewSource(0)),
	}

	return net, nil
}

// MaxIndex returns the index of the maximum value
func MaxIndex(values []float64) int {
	maxIdx := 0
	maxVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// Max returns the maximum value
func Max(values []float64) float64 {
	maxVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
		}
	}
	return maxVal
}
