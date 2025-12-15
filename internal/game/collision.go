package game

// CollisionType represents the type of collision that occurred
type CollisionType int

const (
	NoCollision CollisionType = iota
	WallCollision
	SelfCollision
	OtherSnakeCollision
	HeadToHeadCollision
)

// CollisionResult contains information about a collision check
type CollisionResult struct {
	Type     CollisionType
	SnakeID  int  // Which snake was hit (for OtherSnakeCollision)
	Position Position
}

// CheckWallCollision checks if a position is outside the board bounds
func CheckWallCollision(pos Position, width, height int) bool {
	return pos.X < 0 || pos.X >= width || pos.Y < 0 || pos.Y >= height
}

// CheckSelfCollision checks if the snake collides with its own body
// Note: This should be called AFTER the snake has moved
func CheckSelfCollision(snake *Snake) bool {
	if !snake.Alive || len(snake.Body) < 2 {
		return false
	}
	head := snake.Head()
	// Check against body (excluding head)
	return snake.ContainsPosition(head, true)
}

// CheckSnakeCollision checks if snake1's head collides with snake2's body
func CheckSnakeCollision(snake1, snake2 *Snake) bool {
	if !snake1.Alive || !snake2.Alive {
		return false
	}
	head := snake1.Head()
	// Check if head hits any part of the other snake's body
	return snake2.ContainsPosition(head, false)
}

// CheckHeadToHeadCollision checks if two snakes' heads occupy the same position
func CheckHeadToHeadCollision(snake1, snake2 *Snake) bool {
	if !snake1.Alive || !snake2.Alive {
		return false
	}
	return snake1.Head().Equals(snake2.Head())
}

// CheckFoodCollision checks if a snake's head is at the food position
func CheckFoodCollision(snake *Snake, foodPos Position) bool {
	if !snake.Alive {
		return false
	}
	return snake.Head().Equals(foodPos)
}

// CheckAllCollisions performs all collision checks for a game state
// Returns collision results for each snake
func CheckAllCollisions(snakes [2]*Snake, width, height int) [2][]CollisionResult {
	var results [2][]CollisionResult

	for i := 0; i < 2; i++ {
		snake := snakes[i]
		if !snake.Alive {
			continue
		}

		head := snake.Head()

		// Check wall collision
		if CheckWallCollision(head, width, height) {
			results[i] = append(results[i], CollisionResult{
				Type:     WallCollision,
				Position: head,
			})
		}

		// Check self collision
		if CheckSelfCollision(snake) {
			results[i] = append(results[i], CollisionResult{
				Type:     SelfCollision,
				Position: head,
			})
		}
	}

	// Check inter-snake collisions
	if snakes[0].Alive && snakes[1].Alive {
		// Check head-to-head first
		if CheckHeadToHeadCollision(snakes[0], snakes[1]) {
			results[0] = append(results[0], CollisionResult{
				Type:     HeadToHeadCollision,
				SnakeID:  1,
				Position: snakes[0].Head(),
			})
			results[1] = append(results[1], CollisionResult{
				Type:     HeadToHeadCollision,
				SnakeID:  0,
				Position: snakes[1].Head(),
			})
		} else {
			// Check if snake 0's head hits snake 1's body
			if CheckSnakeCollision(snakes[0], snakes[1]) {
				results[0] = append(results[0], CollisionResult{
					Type:     OtherSnakeCollision,
					SnakeID:  1,
					Position: snakes[0].Head(),
				})
			}
			// Check if snake 1's head hits snake 0's body
			if CheckSnakeCollision(snakes[1], snakes[0]) {
				results[1] = append(results[1], CollisionResult{
					Type:     OtherSnakeCollision,
					SnakeID:  0,
					Position: snakes[1].Head(),
				})
			}
		}
	}

	return results
}

// IsDangerPosition checks if a position would be dangerous for a snake
// Used for state encoding
func IsDangerPosition(pos Position, snakeID int, snakes [2]*Snake, width, height int) bool {
	// Wall danger
	if CheckWallCollision(pos, width, height) {
		return true
	}

	// Self-body danger (excluding head since we're checking future position)
	ownSnake := snakes[snakeID]
	if ownSnake.ContainsPosition(pos, true) {
		return true
	}

	// Other snake danger
	otherID := 1 - snakeID
	otherSnake := snakes[otherID]
	if otherSnake.Alive && otherSnake.ContainsPosition(pos, false) {
		return true
	}

	return false
}
