package ai

import (
	"autonomous-snake/internal/game"
)

// Action represents a relative action for the agent
type Action int

const (
	GoStraight Action = iota
	TurnLeft
	TurnRight
	NumActions = 3
)

// ActionToDirection converts a relative action to absolute direction
func ActionToDirection(currentDir game.Direction, action Action) game.Direction {
	switch action {
	case GoStraight:
		return currentDir
	case TurnLeft:
		return currentDir.TurnLeft()
	case TurnRight:
		return currentDir.TurnRight()
	}
	return currentDir
}

// StateSize is the number of features in the state vector
const StateSize = 22

// EncodeState converts game state to a neural network input vector
// The state is encoded from the perspective of the specified snake
func EncodeState(state *game.GameState, snakeID int) []float64 {
	features := make([]float64, StateSize)
	idx := 0

	snake := state.Snakes[snakeID]
	otherSnake := state.Snakes[1-snakeID]

	if !snake.Alive {
		return features // All zeros for dead snake
	}

	head := snake.Head()
	dir := snake.Direction

	// 1. Danger detection - straight, left, right (3 values) [0-2]
	straightPos := snake.NextHead(dir)
	leftPos := snake.NextHead(dir.TurnLeft())
	rightPos := snake.NextHead(dir.TurnRight())

	features[idx] = boolToFloat(isDanger(straightPos, snakeID, state))
	idx++
	features[idx] = boolToFloat(isDanger(leftPos, snakeID, state))
	idx++
	features[idx] = boolToFloat(isDanger(rightPos, snakeID, state))
	idx++

	// 2. Current direction - one-hot (4 values) [3-6]
	features[idx+int(dir)] = 1.0
	idx += 4

	// 3. Food direction relative to head (4 values) [7-10]
	if state.Food.Active {
		foodPos := state.Food.Position
		features[idx] = boolToFloat(foodPos.Y < head.Y)   // Food up
		features[idx+1] = boolToFloat(foodPos.Y > head.Y) // Food down
		features[idx+2] = boolToFloat(foodPos.X < head.X) // Food left
		features[idx+3] = boolToFloat(foodPos.X > head.X) // Food right
	}
	idx += 4

	// 4. Opponent direction relative to head (4 values) [11-14]
	if otherSnake.Alive {
		oppHead := otherSnake.Head()
		features[idx] = boolToFloat(oppHead.Y < head.Y)   // Opponent up
		features[idx+1] = boolToFloat(oppHead.Y > head.Y) // Opponent down
		features[idx+2] = boolToFloat(oppHead.X < head.X) // Opponent left
		features[idx+3] = boolToFloat(oppHead.X > head.X) // Opponent right
	}
	idx += 4

	// 5. Distance to opponent head - normalized (1 value) [15]
	if otherSnake.Alive {
		oppHead := otherSnake.Head()
		maxDist := float64(state.Width + state.Height)
		dist := float64(game.ManhattanDistance(head, oppHead))
		features[idx] = 1.0 - (dist / maxDist) // Higher when closer
	}
	idx++

	// 6. Distance to nearest opponent body segment - normalized (1 value) [16]
	if otherSnake.Alive {
		minDist := float64(state.Width + state.Height)
		for _, segment := range otherSnake.Body {
			dist := float64(game.ManhattanDistance(head, segment))
			if dist < minDist {
				minDist = dist
			}
		}
		maxDist := float64(state.Width + state.Height)
		features[idx] = 1.0 - (minDist / maxDist)
	}
	idx++

	// 7. Own length - normalized (1 value) [17]
	maxLength := float64(state.Width * state.Height / 2)
	features[idx] = float64(snake.Length()) / maxLength
	idx++

	// 8. Opponent length - normalized (1 value) [18]
	if otherSnake.Alive {
		features[idx] = float64(otherSnake.Length()) / maxLength
	}
	idx++

	// 9. Extended danger - 2 steps ahead (3 values) [19-21]
	// Check if going straight would lead to danger in 2 steps
	// Create temporary snake instances to use NextHead method
	tempSnake := &game.Snake{Body: []game.Position{straightPos}}
	straight2 := tempSnake.NextHead(dir)
	tempSnake.Body[0] = leftPos
	left2 := tempSnake.NextHead(dir.TurnLeft())
	tempSnake.Body[0] = rightPos
	right2 := tempSnake.NextHead(dir.TurnRight())

	features[idx] = boolToFloat(isDangerExtended(straightPos, straight2, snakeID, state))
	idx++
	features[idx] = boolToFloat(isDangerExtended(leftPos, left2, snakeID, state))
	idx++
	features[idx] = boolToFloat(isDangerExtended(rightPos, right2, snakeID, state))

	return features
}

// isDanger checks if a position is dangerous
func isDanger(pos game.Position, snakeID int, state *game.GameState) bool {
	return game.IsDangerPosition(pos, snakeID, state.Snakes, state.Width, state.Height)
}

// isDangerExtended checks if both step1 and step2 positions are dangerous
func isDangerExtended(step1, step2 game.Position, snakeID int, state *game.GameState) bool {
	// If step1 is danger, we won't reach step2
	if isDanger(step1, snakeID, state) {
		return true
	}
	return isDanger(step2, snakeID, state)
}

// boolToFloat converts bool to 0.0 or 1.0
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// CalculateShapingReward computes distance-based reward shaping
// Call this BEFORE the step to compare with AFTER
func CalculateShapingReward(prevState, newState *game.GameState, snakeID int) float64 {
	prevSnake := prevState.Snakes[snakeID]
	newSnake := newState.Snakes[snakeID]

	if !prevSnake.Alive || !newSnake.Alive {
		return 0.0
	}

	if !prevState.Food.Active || !newState.Food.Active {
		return 0.0
	}

	prevDist := game.ManhattanDistance(prevSnake.Head(), prevState.Food.Position)
	newDist := game.ManhattanDistance(newSnake.Head(), newState.Food.Position)

	if newDist < prevDist {
		return 0.1 // Moving toward food
	} else if newDist > prevDist {
		return -0.1 // Moving away from food
	}
	return 0.0
}
