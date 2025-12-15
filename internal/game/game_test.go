package game

import (
	"testing"

	"autonomous-snake/internal/config"
)

func TestNewSnake(t *testing.T) {
	snake := NewSnake(0, Position{X: 5, Y: 5}, Right, 3)

	if snake.ID != 0 {
		t.Errorf("expected ID 0, got %d", snake.ID)
	}
	if len(snake.Body) != 3 {
		t.Errorf("expected body length 3, got %d", len(snake.Body))
	}
	if !snake.Alive {
		t.Error("expected snake to be alive")
	}
	if snake.Head() != (Position{X: 5, Y: 5}) {
		t.Errorf("expected head at (5,5), got %v", snake.Head())
	}
}

func TestSnakeMove(t *testing.T) {
	snake := NewSnake(0, Position{X: 5, Y: 5}, Right, 3)
	initialLen := len(snake.Body)

	snake.Move(Right, false)

	if snake.Head() != (Position{X: 6, Y: 5}) {
		t.Errorf("expected head at (6,5), got %v", snake.Head())
	}
	if len(snake.Body) != initialLen {
		t.Errorf("expected same length after move, got %d", len(snake.Body))
	}
}

func TestSnakeMoveGrow(t *testing.T) {
	snake := NewSnake(0, Position{X: 5, Y: 5}, Right, 3)
	initialLen := len(snake.Body)

	snake.Move(Right, true)

	if len(snake.Body) != initialLen+1 {
		t.Errorf("expected length %d after grow, got %d", initialLen+1, len(snake.Body))
	}
}

func TestSnakePreventUTurn(t *testing.T) {
	snake := NewSnake(0, Position{X: 5, Y: 5}, Right, 3)

	snake.Move(Left, false) // Try to turn 180 degrees

	// Should continue moving right, not left
	if snake.Direction != Right {
		t.Errorf("expected direction Right after U-turn attempt, got %v", snake.Direction)
	}
	if snake.Head() != (Position{X: 6, Y: 5}) {
		t.Errorf("expected head at (6,5), got %v", snake.Head())
	}
}

func TestDirectionTurns(t *testing.T) {
	tests := []struct {
		start       Direction
		expectedL   Direction
		expectedR   Direction
		expectedOpp Direction
	}{
		{Up, Left, Right, Down},
		{Down, Right, Left, Up},
		{Left, Down, Up, Right},
		{Right, Up, Down, Left},
	}

	for _, tt := range tests {
		if got := tt.start.TurnLeft(); got != tt.expectedL {
			t.Errorf("%v.TurnLeft() = %v, want %v", tt.start, got, tt.expectedL)
		}
		if got := tt.start.TurnRight(); got != tt.expectedR {
			t.Errorf("%v.TurnRight() = %v, want %v", tt.start, got, tt.expectedR)
		}
		if got := tt.start.Opposite(); got != tt.expectedOpp {
			t.Errorf("%v.Opposite() = %v, want %v", tt.start, got, tt.expectedOpp)
		}
	}
}

func TestWallCollision(t *testing.T) {
	tests := []struct {
		pos      Position
		expected bool
	}{
		{Position{0, 0}, false},
		{Position{9, 9}, false},
		{Position{-1, 0}, true},
		{Position{0, -1}, true},
		{Position{10, 0}, true},
		{Position{0, 10}, true},
	}

	for _, tt := range tests {
		if got := CheckWallCollision(tt.pos, 10, 10); got != tt.expected {
			t.Errorf("CheckWallCollision(%v, 10, 10) = %v, want %v", tt.pos, got, tt.expected)
		}
	}
}

func TestSelfCollision(t *testing.T) {
	snake := NewSnake(0, Position{X: 5, Y: 5}, Right, 5)

	// No collision with initial state
	if CheckSelfCollision(snake) {
		t.Error("expected no self collision initially")
	}

	// Create a collision by moving the head to overlap body
	snake.Body[0] = snake.Body[2] // Put head on body segment

	if !CheckSelfCollision(snake) {
		t.Error("expected self collision when head overlaps body")
	}
}

func TestNewGame(t *testing.T) {
	cfg := config.GameConfig{
		BoardWidth:  20,
		BoardHeight: 20,
		GridSize:    20,
	}
	g := NewGame(cfg, 42)

	if g.State.Width != 20 || g.State.Height != 20 {
		t.Errorf("expected 20x20 board, got %dx%d", g.State.Width, g.State.Height)
	}
	if g.State.Snakes[0] == nil || g.State.Snakes[1] == nil {
		t.Error("expected two snakes")
	}
	if !g.State.Snakes[0].Alive || !g.State.Snakes[1].Alive {
		t.Error("expected both snakes to be alive")
	}
	if !g.State.Food.Active {
		t.Error("expected food to be active")
	}
	if g.State.GameOver {
		t.Error("expected game not over initially")
	}
}

func TestGameStep(t *testing.T) {
	cfg := config.GameConfig{
		BoardWidth:  20,
		BoardHeight: 20,
		GridSize:    20,
	}
	g := NewGame(cfg, 42)

	initialTurn := g.State.Turn
	result := g.Step([2]Direction{Right, Left})

	if g.State.Turn != initialTurn+1 {
		t.Errorf("expected turn %d, got %d", initialTurn+1, g.State.Turn)
	}
	if result.GameOver && !g.State.GameOver {
		t.Error("result.GameOver should match state.GameOver")
	}
}

func TestGameReset(t *testing.T) {
	cfg := config.GameConfig{
		BoardWidth:  20,
		BoardHeight: 20,
		GridSize:    20,
	}
	g := NewGame(cfg, 42)

	// Make some moves
	for i := 0; i < 5; i++ {
		g.Step([2]Direction{Right, Left})
	}

	// Reset
	g.Reset()

	if g.State.Turn != 0 {
		t.Errorf("expected turn 0 after reset, got %d", g.State.Turn)
	}
	if g.State.GameOver {
		t.Error("expected game not over after reset")
	}
	if !g.State.Snakes[0].Alive || !g.State.Snakes[1].Alive {
		t.Error("expected both snakes alive after reset")
	}
}

func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		p1, p2   Position
		expected int
	}{
		{Position{0, 0}, Position{0, 0}, 0},
		{Position{0, 0}, Position{3, 4}, 7},
		{Position{5, 5}, Position{2, 1}, 7},
		{Position{-1, -1}, Position{1, 1}, 4},
	}

	for _, tt := range tests {
		if got := ManhattanDistance(tt.p1, tt.p2); got != tt.expected {
			t.Errorf("ManhattanDistance(%v, %v) = %d, want %d", tt.p1, tt.p2, got, tt.expected)
		}
	}
}

func TestHeadToHeadCollision(t *testing.T) {
	snake1 := NewSnake(0, Position{X: 5, Y: 5}, Right, 3)
	snake2 := NewSnake(1, Position{X: 7, Y: 5}, Left, 3)

	// Initially no collision
	if CheckHeadToHeadCollision(snake1, snake2) {
		t.Error("expected no head-to-head collision initially")
	}

	// Move them to same position
	snake1.Move(Right, false)
	snake2.Move(Left, false)

	// Now both heads at (6, 5)
	if !CheckHeadToHeadCollision(snake1, snake2) {
		t.Errorf("expected head-to-head collision, snake1 head: %v, snake2 head: %v",
			snake1.Head(), snake2.Head())
	}
}
