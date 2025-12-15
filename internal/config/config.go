package config

// GameConfig holds game-related configuration
type GameConfig struct {
	BoardWidth  int
	BoardHeight int
	GridSize    int // pixels per cell for rendering
}

// DefaultGameConfig returns sensible defaults
func DefaultGameConfig() GameConfig {
	return GameConfig{
		BoardWidth:  20,
		BoardHeight: 20,
		GridSize:    20,
	}
}

// TrainingConfig holds training hyperparameters
type TrainingConfig struct {
	// Neural Network
	InputSize    int
	HiddenSize1  int
	HiddenSize2  int
	OutputSize   int
	LearningRate float64

	// DQN
	Gamma        float64
	EpsilonStart float64
	EpsilonMin   float64
	EpsilonDecay float64

	// Training
	BatchSize     int
	BufferSize    int
	TargetUpdate  int
	Episodes      int
	MaxStepsPerEp int

	// Persistence
	SaveFrequency int
	ModelPath     string
}

// DefaultTrainingConfig returns sensible defaults
func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		// Neural Network
		InputSize:    22,
		HiddenSize1:  128,
		HiddenSize2:  64,
		OutputSize:   3,
		LearningRate: 0.001,

		// DQN
		Gamma:        0.99,
		EpsilonStart: 1.0,
		EpsilonMin:   0.01,
		EpsilonDecay: 0.995,

		// Training
		BatchSize:     64,
		BufferSize:    100000,
		TargetUpdate:  1000,
		Episodes:      10000,
		MaxStepsPerEp: 1000,

		// Persistence
		SaveFrequency: 500,
		ModelPath:     "models/snake_dqn.gob",
	}
}
