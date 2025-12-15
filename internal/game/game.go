package game

import (
	"math/rand"

	"autonomous-snake/internal/config"
)

// Food represents food on the board
type Food struct {
	Position Position
	Active   bool
}

// GameState represents the complete state of a game
type GameState struct {
	Width    int
	Height   int
	Snakes   [2]*Snake
	Food     Food
	Turn     int
	GameOver bool
	Winner   int // -1 = tie, 0 = snake 0 wins, 1 = snake 1 wins
}

// StepResult contains the result of a game step
type StepResult struct {
	Rewards  [2]float64
	AteFood  [2]bool
	Died     [2]bool
	GameOver bool
	Winner   int
}

// Game manages the game logic
type Game struct {
	State *GameState
	rng   *rand.Rand
}

// NewGame creates a new game instance
func NewGame(cfg config.GameConfig, seed int64) *Game {
	rng := rand.New(rand.NewSource(seed))
	g := &Game{
		State: &GameState{
			Width:  cfg.BoardWidth,
			Height: cfg.BoardHeight,
		},
		rng: rng,
	}
	g.Reset()
	return g
}

// Reset resets the game to initial state
func (g *Game) Reset() *GameState {
	width := g.State.Width
	height := g.State.Height

	// Place snake 0 on the left side, facing right
	snake0Start := Position{X: 3, Y: height / 2}
	g.State.Snakes[0] = NewSnake(0, snake0Start, Right, 3)

	// Place snake 1 on the right side, facing left
	snake1Start := Position{X: width - 4, Y: height / 2}
	g.State.Snakes[1] = NewSnake(1, snake1Start, Left, 3)

	// Spawn initial food
	g.spawnFood()

	g.State.Turn = 0
	g.State.GameOver = false
	g.State.Winner = -1

	return g.State
}

// spawnFood places food at a random empty position
func (g *Game) spawnFood() {
	// Collect all occupied positions
	occupied := make(map[Position]bool)
	for _, snake := range g.State.Snakes {
		if snake != nil {
			for _, pos := range snake.Body {
				occupied[pos] = true
			}
		}
	}

	// Find all empty positions
	var emptyPositions []Position
	for x := 0; x < g.State.Width; x++ {
		for y := 0; y < g.State.Height; y++ {
			pos := Position{X: x, Y: y}
			if !occupied[pos] {
				emptyPositions = append(emptyPositions, pos)
			}
		}
	}

	// Pick a random empty position
	if len(emptyPositions) > 0 {
		idx := g.rng.Intn(len(emptyPositions))
		g.State.Food = Food{
			Position: emptyPositions[idx],
			Active:   true,
		}
	} else {
		g.State.Food.Active = false
	}
}

// Step advances the game by one turn
// actions[0] is the direction for snake 0, actions[1] for snake 1
func (g *Game) Step(actions [2]Direction) StepResult {
	result := StepResult{
		Winner: -1,
	}

	if g.State.GameOver {
		result.GameOver = true
		result.Winner = g.State.Winner
		return result
	}

	g.State.Turn++

	// Check which snakes will eat food this turn (before moving)
	willEat := [2]bool{false, false}
	for i := 0; i < 2; i++ {
		snake := g.State.Snakes[i]
		if snake.Alive {
			nextHead := snake.NextHead(actions[i])
			if g.State.Food.Active && nextHead.Equals(g.State.Food.Position) {
				willEat[i] = true
			}
		}
	}

	// Move both snakes simultaneously
	for i := 0; i < 2; i++ {
		snake := g.State.Snakes[i]
		if snake.Alive {
			snake.Move(actions[i], willEat[i])
		}
	}

	// Handle food eating
	for i := 0; i < 2; i++ {
		if willEat[i] {
			result.AteFood[i] = true
			g.State.Snakes[i].Score++
		}
	}

	// Spawn new food if eaten
	if willEat[0] || willEat[1] {
		g.spawnFood()
	}

	// Check collisions
	collisions := CheckAllCollisions(g.State.Snakes, g.State.Width, g.State.Height)

	// Process deaths
	for i := 0; i < 2; i++ {
		if len(collisions[i]) > 0 {
			g.State.Snakes[i].Kill()
			result.Died[i] = true
		}
	}

	// Calculate rewards
	result.Rewards = g.calculateRewards(result.AteFood, result.Died)

	// Check game over
	alive0 := g.State.Snakes[0].Alive
	alive1 := g.State.Snakes[1].Alive

	if !alive0 && !alive1 {
		g.State.GameOver = true
		g.State.Winner = -1 // Tie
		result.GameOver = true
		result.Winner = -1
	} else if !alive0 {
		g.State.GameOver = true
		g.State.Winner = 1 // Snake 1 wins
		result.GameOver = true
		result.Winner = 1
	} else if !alive1 {
		g.State.GameOver = true
		g.State.Winner = 0 // Snake 0 wins
		result.GameOver = true
		result.Winner = 0
	}

	return result
}

// calculateRewards computes rewards for each snake
func (g *Game) calculateRewards(ateFood, died [2]bool) [2]float64 {
	var rewards [2]float64

	for i := 0; i < 2; i++ {
		otherIdx := 1 - i

		if died[i] {
			rewards[i] = -1.0 // Death penalty
		} else {
			// Survival bonus
			rewards[i] = 0.01

			// Food reward
			if ateFood[i] {
				rewards[i] += 0.5
			}

			// Win bonus if opponent died
			if died[otherIdx] {
				rewards[i] += 1.0
			}
		}
	}

	return rewards
}

// GetState returns a copy of the current game state
func (g *Game) GetState() *GameState {
	return g.State
}

// Clone creates a deep copy of the game for simulation
func (g *Game) Clone() *Game {
	newGame := &Game{
		State: &GameState{
			Width:    g.State.Width,
			Height:   g.State.Height,
			Turn:     g.State.Turn,
			GameOver: g.State.GameOver,
			Winner:   g.State.Winner,
			Food: Food{
				Position: g.State.Food.Position,
				Active:   g.State.Food.Active,
			},
		},
		rng: rand.New(rand.NewSource(g.rng.Int63())),
	}

	// Deep copy snakes
	for i := 0; i < 2; i++ {
		s := g.State.Snakes[i]
		body := make([]Position, len(s.Body))
		copy(body, s.Body)
		newGame.State.Snakes[i] = &Snake{
			ID:        s.ID,
			Body:      body,
			Direction: s.Direction,
			Alive:     s.Alive,
			Score:     s.Score,
			Grew:      s.Grew,
		}
	}

	return newGame
}

// IsValidAction checks if an action is valid for a snake (not a 180-degree turn)
func IsValidAction(currentDir, newDir Direction) bool {
	return newDir != currentDir.Opposite()
}

// ManhattanDistance calculates the Manhattan distance between two positions
func ManhattanDistance(p1, p2 Position) int {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return dx + dy
}
