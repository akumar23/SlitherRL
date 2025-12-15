package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"autonomous-snake/internal/ai"
	"autonomous-snake/internal/config"
	"autonomous-snake/internal/game"
)

func main() {
	// Parse command line flags
	episodes := flag.Int("episodes", 10000, "Number of training episodes")
	modelPath := flag.String("model", "models/snake_dqn.gob", "Path to save/load model")
	loadModel := flag.String("load", "", "Path to load existing model from")
	boardSize := flag.Int("board", 20, "Board width and height")
	saveFreq := flag.Int("save-freq", 500, "Save model every N episodes")
	logFreq := flag.Int("log-freq", 100, "Log stats every N episodes")
	seed := flag.Int64("seed", 0, "Random seed (0 for time-based)")
	flag.Parse()

	if *seed == 0 {
		*seed = time.Now().UnixNano()
	}

	// Configuration
	gameCfg := config.GameConfig{
		BoardWidth:  *boardSize,
		BoardHeight: *boardSize,
		GridSize:    20,
	}

	trainCfg := config.DefaultTrainingConfig()
	trainCfg.Episodes = *episodes
	trainCfg.SaveFrequency = *saveFreq
	trainCfg.ModelPath = *modelPath

	// Create agent
	agent := ai.NewDQNAgent(trainCfg, *seed)

	// Load existing model if specified
	if *loadModel != "" {
		if err := agent.Load(*loadModel); err != nil {
			log.Printf("Warning: Could not load model from %s: %v", *loadModel, err)
		} else {
			log.Printf("Loaded model from %s", *loadModel)
		}
	}

	// Create game
	g := game.NewGame(gameCfg, *seed)

	// Training stats
	totalRewards := make([]float64, 2)
	totalWins := [2]int{0, 0}
	totalTies := 0
	totalSteps := 0
	episodeLengths := make([]int, 0, *logFreq)

	log.Printf("Starting training for %d episodes...", *episodes)
	log.Printf("Board: %dx%d, Epsilon: %.2f -> %.2f", *boardSize, *boardSize, trainCfg.EpsilonStart, trainCfg.EpsilonMin)

	startTime := time.Now()

	for ep := 1; ep <= *episodes; ep++ {
		state := g.Reset()
		episodeReward := [2]float64{0, 0}
		steps := 0

		for !state.GameOver && steps < trainCfg.MaxStepsPerEp {
			steps++

			// Encode states for both snakes
			state0 := ai.EncodeState(state, 0)
			state1 := ai.EncodeState(state, 1)

			// Select actions
			action0 := agent.SelectAction(state0)
			action1 := agent.SelectAction(state1)

			// Convert to directions
			dir0 := ai.ActionToDirection(state.Snakes[0].Direction, action0)
			dir1 := ai.ActionToDirection(state.Snakes[1].Direction, action1)

			// Store previous state for shaping reward
			prevState := g.Clone().State

			// Step game
			result := g.Step([2]game.Direction{dir0, dir1})

			// Encode next states
			nextState0 := ai.EncodeState(state, 0)
			nextState1 := ai.EncodeState(state, 1)

			// Calculate total rewards including shaping
			reward0 := result.Rewards[0] + ai.CalculateShapingReward(prevState, state, 0)
			reward1 := result.Rewards[1] + ai.CalculateShapingReward(prevState, state, 1)

			// Store experiences
			agent.Remember(state0, action0, reward0, nextState0, result.Died[0] || result.GameOver)
			agent.Remember(state1, action1, reward1, nextState1, result.Died[1] || result.GameOver)

			// Train
			agent.Train()

			episodeReward[0] += reward0
			episodeReward[1] += reward1
		}

		// Update stats
		totalRewards[0] += episodeReward[0]
		totalRewards[1] += episodeReward[1]
		totalSteps += steps
		episodeLengths = append(episodeLengths, steps)

		if state.Winner == 0 {
			totalWins[0]++
		} else if state.Winner == 1 {
			totalWins[1]++
		} else {
			totalTies++
		}

		// Decay epsilon
		agent.DecayEpsilon()

		// Log progress
		if ep%*logFreq == 0 {
			avgLen := 0.0
			for _, l := range episodeLengths {
				avgLen += float64(l)
			}
			avgLen /= float64(len(episodeLengths))

			elapsed := time.Since(startTime)
			epsPerSec := float64(ep) / elapsed.Seconds()

			log.Printf("Episode %d/%d | Epsilon: %.4f | Avg Length: %.1f | Wins: %d/%d | Ties: %d | %.1f eps/s",
				ep, *episodes, agent.Epsilon, avgLen, totalWins[0], totalWins[1], totalTies, epsPerSec)

			// Reset periodic stats
			episodeLengths = episodeLengths[:0]
		}

		// Save model
		if ep%*saveFreq == 0 {
			if err := os.MkdirAll("models", 0755); err != nil {
				log.Printf("Warning: Could not create models directory: %v", err)
			}
			if err := agent.Save(*modelPath); err != nil {
				log.Printf("Warning: Could not save model: %v", err)
			} else {
				log.Printf("Saved model to %s", *modelPath)
			}
		}
	}

	// Final save
	if err := os.MkdirAll("models", 0755); err != nil {
		log.Printf("Warning: Could not create models directory: %v", err)
	}
	if err := agent.Save(*modelPath); err != nil {
		log.Printf("Error saving final model: %v", err)
	} else {
		log.Printf("Training complete. Model saved to %s", *modelPath)
	}

	// Print final stats
	elapsed := time.Since(startTime)
	fmt.Printf("\n=== Training Summary ===\n")
	fmt.Printf("Episodes: %d\n", *episodes)
	fmt.Printf("Total Time: %v\n", elapsed.Round(time.Second))
	fmt.Printf("Episodes/sec: %.1f\n", float64(*episodes)/elapsed.Seconds())
	fmt.Printf("Snake 0 Wins: %d (%.1f%%)\n", totalWins[0], 100*float64(totalWins[0])/float64(*episodes))
	fmt.Printf("Snake 1 Wins: %d (%.1f%%)\n", totalWins[1], 100*float64(totalWins[1])/float64(*episodes))
	fmt.Printf("Ties: %d (%.1f%%)\n", totalTies, 100*float64(totalTies)/float64(*episodes))
	fmt.Printf("Final Epsilon: %.4f\n", agent.Epsilon)
}
