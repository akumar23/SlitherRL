package game

// Direction represents the movement direction
type Direction int

const (
	Up Direction = iota
	Down
	Left
	Right
)

// Opposite returns the opposite direction
func (d Direction) Opposite() Direction {
	switch d {
	case Up:
		return Down
	case Down:
		return Up
	case Left:
		return Right
	case Right:
		return Left
	}
	return d
}

// TurnLeft returns the direction after turning left
func (d Direction) TurnLeft() Direction {
	switch d {
	case Up:
		return Left
	case Left:
		return Down
	case Down:
		return Right
	case Right:
		return Up
	}
	return d
}

// TurnRight returns the direction after turning right
func (d Direction) TurnRight() Direction {
	switch d {
	case Up:
		return Right
	case Right:
		return Down
	case Down:
		return Left
	case Left:
		return Up
	}
	return d
}

// Position represents a coordinate on the board
type Position struct {
	X, Y int
}

// Add returns a new position offset by dx, dy
func (p Position) Add(dx, dy int) Position {
	return Position{X: p.X + dx, Y: p.Y + dy}
}

// Equals checks if two positions are the same
func (p Position) Equals(other Position) bool {
	return p.X == other.X && p.Y == other.Y
}

// Snake represents a snake in the game
type Snake struct {
	ID        int
	Body      []Position // Head is Body[0]
	Direction Direction
	Alive     bool
	Score     int
	Grew      bool // Whether snake grew this turn (for collision resolution)
}

// NewSnake creates a new snake at the given position
func NewSnake(id int, head Position, dir Direction, length int) *Snake {
	body := make([]Position, length)
	body[0] = head

	// Build body behind the head based on direction
	dx, dy := 0, 0
	switch dir {
	case Up:
		dy = 1 // body extends downward
	case Down:
		dy = -1
	case Left:
		dx = 1 // body extends to the right
	case Right:
		dx = -1
	}

	for i := 1; i < length; i++ {
		body[i] = Position{
			X: head.X + dx*i,
			Y: head.Y + dy*i,
		}
	}

	return &Snake{
		ID:        id,
		Body:      body,
		Direction: dir,
		Alive:     true,
		Score:     0,
		Grew:      false,
	}
}

// Head returns the snake's head position
func (s *Snake) Head() Position {
	return s.Body[0]
}

// Length returns the snake's current length
func (s *Snake) Length() int {
	return len(s.Body)
}

// NextHead returns where the head will be after moving in the given direction
func (s *Snake) NextHead(dir Direction) Position {
	head := s.Head()
	switch dir {
	case Up:
		return head.Add(0, -1)
	case Down:
		return head.Add(0, 1)
	case Left:
		return head.Add(-1, 0)
	case Right:
		return head.Add(1, 0)
	}
	return head
}

// Move moves the snake in the given direction
// If grow is true, the snake grows by one segment
func (s *Snake) Move(dir Direction, grow bool) {
	if !s.Alive {
		return
	}

	// Prevent 180-degree turns
	if dir == s.Direction.Opposite() {
		dir = s.Direction
	}

	newHead := s.NextHead(dir)
	s.Direction = dir
	s.Grew = grow

	// Prepend new head
	newBody := make([]Position, 0, len(s.Body)+1)
	newBody = append(newBody, newHead)
	newBody = append(newBody, s.Body...)

	// Remove tail unless growing
	if !grow {
		newBody = newBody[:len(newBody)-1]
	}

	s.Body = newBody
}

// ContainsPosition checks if the snake's body contains the given position
// If excludeHead is true, skips checking the head
func (s *Snake) ContainsPosition(pos Position, excludeHead bool) bool {
	startIdx := 0
	if excludeHead {
		startIdx = 1
	}
	for i := startIdx; i < len(s.Body); i++ {
		if s.Body[i].Equals(pos) {
			return true
		}
	}
	return false
}

// Kill marks the snake as dead
func (s *Snake) Kill() {
	s.Alive = false
}

// Reset resets the snake to initial state
func (s *Snake) Reset(head Position, dir Direction, length int) {
	newSnake := NewSnake(s.ID, head, dir, length)
	s.Body = newSnake.Body
	s.Direction = dir
	s.Alive = true
	s.Score = 0
	s.Grew = false
}
