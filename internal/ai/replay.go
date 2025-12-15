package ai

import "math/rand"

// Experience represents a single transition
type Experience struct {
	State     []float64
	Action    Action
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores experiences for training
type ReplayBuffer struct {
	buffer   []Experience
	capacity int
	position int
	size     int
	rng      *rand.Rand
}

// NewReplayBuffer creates a new replay buffer with given capacity
func NewReplayBuffer(capacity int, seed int64) *ReplayBuffer {
	return &ReplayBuffer{
		buffer:   make([]Experience, capacity),
		capacity: capacity,
		position: 0,
		size:     0,
		rng:      rand.New(rand.NewSource(seed)),
	}
}

// Add adds an experience to the buffer
func (rb *ReplayBuffer) Add(exp Experience) {
	// Make copies of slices to avoid aliasing
	stateCopy := make([]float64, len(exp.State))
	copy(stateCopy, exp.State)

	nextStateCopy := make([]float64, len(exp.NextState))
	copy(nextStateCopy, exp.NextState)

	rb.buffer[rb.position] = Experience{
		State:     stateCopy,
		Action:    exp.Action,
		Reward:    exp.Reward,
		NextState: nextStateCopy,
		Done:      exp.Done,
	}

	rb.position = (rb.position + 1) % rb.capacity
	if rb.size < rb.capacity {
		rb.size++
	}
}

// Sample returns a random batch of experiences
func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
	if batchSize > rb.size {
		batchSize = rb.size
	}

	batch := make([]Experience, batchSize)
	indices := rb.rng.Perm(rb.size)[:batchSize]

	for i, idx := range indices {
		batch[i] = rb.buffer[idx]
	}

	return batch
}

// Size returns the current number of experiences in the buffer
func (rb *ReplayBuffer) Size() int {
	return rb.size
}

// IsFull returns true if the buffer has reached capacity
func (rb *ReplayBuffer) IsFull() bool {
	return rb.size == rb.capacity
}

// Clear empties the buffer
func (rb *ReplayBuffer) Clear() {
	rb.position = 0
	rb.size = 0
}
