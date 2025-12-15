package main

import (
	"flag"
	"log"
	"time"

	"autonomous-snake/internal/ai"
	"autonomous-snake/internal/config"
	"autonomous-snake/internal/game"
	"autonomous-snake/internal/render"
)

func main() {
	// Parse command line flags
	modelPath := flag.String("model", "models/snake_dqn.gob", "Path to load model from")
	boardSize := flag.Int("board", 20, "Board width and height")
	gridSize := flag.Int("grid", 20, "Cell size in pixels")
	seed := flag.Int64("seed", 0, "Random seed (0 for time-based)")
	noModel := flag.Bool("random", false, "Run with random actions (no model)")
	flag.Parse()

	if *seed == 0 {
		*seed = time.Now().UnixNano()
	}

	// Configuration
	gameCfg := config.GameConfig{
		BoardWidth:  *boardSize,
		BoardHeight: *boardSize,
		GridSize:    *gridSize,
	}

	trainCfg := config.DefaultTrainingConfig()

	// Create game
	g := game.NewGame(gameCfg, *seed)

	// Load agent
	var agent *ai.DQNAgent
	if !*noModel {
		agent = ai.NewDQNAgent(trainCfg, *seed)
		if err := agent.Load(*modelPath); err != nil {
			log.Printf("Warning: Could not load model from %s: %v", *modelPath, err)
			log.Printf("Running with untrained agent (random-ish behavior)")
		} else {
			log.Printf("Loaded model from %s", *modelPath)
		}
		// Disable exploration for playback
		agent.SetEpsilon(0)
	} else {
		log.Printf("Running with random actions (no model)")
	}

	// Create and run renderer
	renderer := render.NewRenderer(g, agent, gameCfg)

	log.Printf("Starting game...")
	log.Printf("Controls: Space=Pause, Up/Down=Speed, R=Reset, Q=Quit")

	if err := renderer.Run(); err != nil {
		log.Printf("Game ended: %v", err)
	}
}
