package ai

import (
	"math/rand"

	"autonomous-snake/internal/config"
)

// DQNAgent implements the Deep Q-Network algorithm
type DQNAgent struct {
	PolicyNet    *QNetwork
	TargetNet    *QNetwork
	ReplayBuffer *ReplayBuffer

	// Hyperparameters
	Gamma        float64 // Discount factor
	Epsilon      float64 // Current exploration rate
	EpsilonMin   float64
	EpsilonDecay float64
	BatchSize    int

	// Training state
	StepCount     int
	TargetUpdate  int // Steps between target network updates
	TrainInterval int // Steps between training updates

	rng *rand.Rand
}

// NewDQNAgent creates a new DQN agent with the given configuration
func NewDQNAgent(cfg config.TrainingConfig, seed int64) *DQNAgent {
	rng := rand.New(rand.NewSource(seed))

	policyNet := NewQNetwork(
		cfg.InputSize,
		cfg.HiddenSize1,
		cfg.HiddenSize2,
		cfg.OutputSize,
		cfg.LearningRate,
		rng.Int63(),
	)

	targetNet := policyNet.Clone()

	replayBuffer := NewReplayBuffer(cfg.BufferSize, rng.Int63())

	return &DQNAgent{
		PolicyNet:     policyNet,
		TargetNet:     targetNet,
		ReplayBuffer:  replayBuffer,
		Gamma:         cfg.Gamma,
		Epsilon:       cfg.EpsilonStart,
		EpsilonMin:    cfg.EpsilonMin,
		EpsilonDecay:  cfg.EpsilonDecay,
		BatchSize:     cfg.BatchSize,
		StepCount:     0,
		TargetUpdate:  cfg.TargetUpdate,
		TrainInterval: 4, // Train every 4 steps
		rng:           rng,
	}
}

// SelectAction chooses an action using epsilon-greedy policy
func (a *DQNAgent) SelectAction(state []float64) Action {
	// Epsilon-greedy exploration
	if a.rng.Float64() < a.Epsilon {
		return Action(a.rng.Intn(NumActions))
	}

	// Exploit: choose best action according to Q-network
	qValues := a.PolicyNet.Forward(state)
	return Action(MaxIndex(qValues))
}

// SelectActionGreedy chooses the best action (no exploration)
func (a *DQNAgent) SelectActionGreedy(state []float64) Action {
	qValues := a.PolicyNet.Forward(state)
	return Action(MaxIndex(qValues))
}

// Remember stores an experience in the replay buffer
func (a *DQNAgent) Remember(state []float64, action Action, reward float64, nextState []float64, done bool) {
	a.ReplayBuffer.Add(Experience{
		State:     state,
		Action:    action,
		Reward:    reward,
		NextState: nextState,
		Done:      done,
	})
}

// Train performs a training step if enough experiences are available
func (a *DQNAgent) Train() float64 {
	a.StepCount++

	// Don't train if not enough experiences
	if a.ReplayBuffer.Size() < a.BatchSize {
		return 0.0
	}

	// Only train every few steps
	if a.StepCount%a.TrainInterval != 0 {
		return 0.0
	}

	// Sample batch
	batch := a.ReplayBuffer.Sample(a.BatchSize)

	// Train on batch
	totalLoss := 0.0
	for _, exp := range batch {
		loss := a.trainOnExperience(exp)
		totalLoss += loss
	}

	// Update target network periodically
	if a.StepCount%a.TargetUpdate == 0 {
		a.UpdateTargetNetwork()
	}

	return totalLoss / float64(len(batch))
}

// trainOnExperience trains on a single experience
func (a *DQNAgent) trainOnExperience(exp Experience) float64 {
	// Compute target Q-value
	var targetQ float64
	if exp.Done {
		targetQ = exp.Reward
	} else {
		// Use target network for stability (Double DQN style)
		nextQValues := a.TargetNet.Forward(exp.NextState)
		maxNextQ := Max(nextQValues)
		targetQ = exp.Reward + a.Gamma*maxNextQ
	}

	// Forward pass with cache
	output, cache := a.PolicyNet.ForwardWithCache(exp.State)

	// Compute loss for logging
	currentQ := output[exp.Action]
	loss := (currentQ - targetQ) * (currentQ - targetQ) * 0.5

	// Backward pass
	a.PolicyNet.Backward(cache, output, int(exp.Action), targetQ)

	return loss
}

// UpdateTargetNetwork copies weights from policy network to target network
func (a *DQNAgent) UpdateTargetNetwork() {
	a.TargetNet.CopyFrom(a.PolicyNet)
}

// DecayEpsilon reduces exploration rate
func (a *DQNAgent) DecayEpsilon() {
	a.Epsilon *= a.EpsilonDecay
	if a.Epsilon < a.EpsilonMin {
		a.Epsilon = a.EpsilonMin
	}
}

// SetEpsilon sets the exploration rate directly
func (a *DQNAgent) SetEpsilon(eps float64) {
	a.Epsilon = eps
}

// Save saves the agent's policy network
func (a *DQNAgent) Save(path string) error {
	return a.PolicyNet.Save(path)
}

// Load loads weights into the agent's networks
func (a *DQNAgent) Load(path string) error {
	net, err := LoadNetwork(path)
	if err != nil {
		return err
	}
	a.PolicyNet = net
	a.TargetNet = net.Clone()
	return nil
}

// GetQValues returns Q-values for all actions given a state
func (a *DQNAgent) GetQValues(state []float64) []float64 {
	return a.PolicyNet.Forward(state)
}

// AgentState holds serializable agent state for checkpointing
type AgentState struct {
	Epsilon   float64
	StepCount int
}

// GetState returns the agent's current training state
func (a *DQNAgent) GetState() AgentState {
	return AgentState{
		Epsilon:   a.Epsilon,
		StepCount: a.StepCount,
	}
}

// SetState restores agent training state
func (a *DQNAgent) SetState(state AgentState) {
	a.Epsilon = state.Epsilon
	a.StepCount = state.StepCount
}
