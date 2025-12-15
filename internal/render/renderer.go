package render

import (
	"errors"
	"fmt"
	"image/color"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"

	"autonomous-snake/internal/ai"
	"autonomous-snake/internal/config"
	"autonomous-snake/internal/game"
)

// ErrQuit is returned when the user quits the game
var ErrQuit = errors.New("user quit game")

// Colors for rendering
var (
	ColorBackground = color.RGBA{20, 20, 20, 255}
	ColorGrid       = color.RGBA{40, 40, 40, 255}
	ColorSnake0     = color.RGBA{76, 175, 80, 255}  // Green
	ColorSnake0Head = color.RGBA{129, 199, 132, 255}
	ColorSnake1     = color.RGBA{33, 150, 243, 255} // Blue
	ColorSnake1Head = color.RGBA{100, 181, 246, 255}
	ColorFood       = color.RGBA{244, 67, 54, 255}  // Red
	ColorDead       = color.RGBA{128, 128, 128, 255}
	ColorText       = color.RGBA{255, 255, 255, 255}
)

// GameRenderer handles rendering the game using Ebiten
type GameRenderer struct {
	game     *game.Game
	agent    *ai.DQNAgent
	cfg      config.GameConfig
	trainCfg config.TrainingConfig

	// Rendering state
	screenWidth  int
	screenHeight int
	cellSize     int
	offsetX      int
	offsetY      int

	// Game speed control
	ticksPerStep int
	tickCount    int
	paused       bool
	speed        int // 1-5, where 3 is normal

	// Stats
	gamesPlayed int
	wins        [2]int
	ties        int

	// Game over pause
	gameOverPause bool
	gameOverTicks int
}

// NewRenderer creates a new game renderer
func NewRenderer(g *game.Game, agent *ai.DQNAgent, cfg config.GameConfig) *GameRenderer {
	cellSize := cfg.GridSize
	boardWidth := cfg.BoardWidth * cellSize
	boardHeight := cfg.BoardHeight * cellSize

	// Add padding for UI (top header + bottom stats/controls)
	screenWidth := boardWidth + 40
	screenHeight := boardHeight + 100

	return &GameRenderer{
		game:         g,
		agent:        agent,
		cfg:          cfg,
		trainCfg:     config.DefaultTrainingConfig(),
		screenWidth:  screenWidth,
		screenHeight: screenHeight,
		cellSize:     cellSize,
		offsetX:      20,
		offsetY:      60,
		ticksPerStep: 10,
		tickCount:    0,
		paused:       false,
		speed:        3,
		gamesPlayed:  0,
	}
}

// gameOverDelayTicks is how long to pause after game over (at 60 TPS)
const gameOverDelayTicks = 120 // ~2 seconds

// Update is called every tick (60 times per second by default)
func (r *GameRenderer) Update() error {
	// Handle input
	if err := r.handleInput(); err != nil {
		return err
	}

	if r.paused {
		return nil
	}

	// Handle game over pause
	if r.gameOverPause {
		r.gameOverTicks++
		if r.gameOverTicks >= gameOverDelayTicks {
			r.gameOverPause = false
			r.gameOverTicks = 0
			r.game.Reset()
		}
		return nil
	}

	r.tickCount++
	if r.tickCount < r.ticksPerStep {
		return nil
	}
	r.tickCount = 0

	// Check if game is over
	if r.game.State.GameOver {
		// Record result
		r.gamesPlayed++
		if r.game.State.Winner == 0 {
			r.wins[0]++
		} else if r.game.State.Winner == 1 {
			r.wins[1]++
		} else {
			r.ties++
		}

		// Start game over pause
		r.gameOverPause = true
		r.gameOverTicks = 0
		return nil
	}

	// Get AI actions
	state := r.game.State
	state0 := ai.EncodeState(state, 0)
	state1 := ai.EncodeState(state, 1)

	var action0, action1 ai.Action
	if r.agent != nil {
		action0 = r.agent.SelectActionGreedy(state0)
		action1 = r.agent.SelectActionGreedy(state1)
	} else {
		// Random actions if no agent
		action0 = ai.Action(rand.Intn(int(ai.NumActions)))
		action1 = ai.Action(rand.Intn(int(ai.NumActions)))
	}

	// Convert to directions
	dir0 := ai.ActionToDirection(state.Snakes[0].Direction, action0)
	dir1 := ai.ActionToDirection(state.Snakes[1].Direction, action1)

	// Step game
	r.game.Step([2]game.Direction{dir0, dir1})

	return nil
}

// handleInput processes keyboard input
func (r *GameRenderer) handleInput() error {
	// Pause/unpause
	if inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		r.paused = !r.paused
	}

	// Speed control
	if inpututil.IsKeyJustPressed(ebiten.KeyUp) || inpututil.IsKeyJustPressed(ebiten.KeyEqual) {
		r.speed++
		if r.speed > 5 {
			r.speed = 5
		}
		r.updateSpeed()
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyDown) || inpututil.IsKeyJustPressed(ebiten.KeyMinus) {
		r.speed--
		if r.speed < 1 {
			r.speed = 1
		}
		r.updateSpeed()
	}

	// Reset game
	if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		r.game.Reset()
	}

	// Quit
	if inpututil.IsKeyJustPressed(ebiten.KeyQ) || inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
		return ErrQuit
	}

	return nil
}

// updateSpeed adjusts tick rate based on speed setting
func (r *GameRenderer) updateSpeed() {
	speeds := []int{30, 15, 10, 5, 2}
	r.ticksPerStep = speeds[r.speed-1]
}

// Draw renders the current game state
func (r *GameRenderer) Draw(screen *ebiten.Image) {
	// Clear background
	screen.Fill(ColorBackground)

	// Draw grid
	r.drawGrid(screen)

	// Draw food
	r.drawFood(screen)

	// Draw snakes
	r.drawSnake(screen, r.game.State.Snakes[0], ColorSnake0, ColorSnake0Head)
	r.drawSnake(screen, r.game.State.Snakes[1], ColorSnake1, ColorSnake1Head)

	// Draw UI
	r.drawUI(screen)
}

// drawGrid draws the game grid
func (r *GameRenderer) drawGrid(screen *ebiten.Image) {
	boardWidth := r.cfg.BoardWidth * r.cellSize
	boardHeight := r.cfg.BoardHeight * r.cellSize

	// Draw grid lines
	for x := 0; x <= r.cfg.BoardWidth; x++ {
		px := float64(r.offsetX + x*r.cellSize)
		ebitenutil.DrawLine(screen, px, float64(r.offsetY), px, float64(r.offsetY+boardHeight), ColorGrid)
	}
	for y := 0; y <= r.cfg.BoardHeight; y++ {
		py := float64(r.offsetY + y*r.cellSize)
		ebitenutil.DrawLine(screen, float64(r.offsetX), py, float64(r.offsetX+boardWidth), py, ColorGrid)
	}
}

// drawFood draws the food
func (r *GameRenderer) drawFood(screen *ebiten.Image) {
	if !r.game.State.Food.Active {
		return
	}

	pos := r.game.State.Food.Position
	r.drawCell(screen, pos.X, pos.Y, ColorFood, 2)
}

// drawSnake draws a snake
func (r *GameRenderer) drawSnake(screen *ebiten.Image, snake *game.Snake, bodyColor, headColor color.RGBA) {
	if snake == nil {
		return
	}

	useColor := bodyColor
	useHeadColor := headColor
	if !snake.Alive {
		useColor = ColorDead
		useHeadColor = ColorDead
	}

	// Draw body (tail to head)
	for i := len(snake.Body) - 1; i >= 1; i-- {
		pos := snake.Body[i]
		r.drawCell(screen, pos.X, pos.Y, useColor, 1)
	}

	// Draw head
	if len(snake.Body) > 0 {
		head := snake.Head()
		r.drawCell(screen, head.X, head.Y, useHeadColor, 1)
	}
}

// drawCell draws a cell at the given grid position
func (r *GameRenderer) drawCell(screen *ebiten.Image, gx, gy int, c color.RGBA, padding int) {
	x := float64(r.offsetX + gx*r.cellSize + padding)
	y := float64(r.offsetY + gy*r.cellSize + padding)
	w := float64(r.cellSize - padding*2)
	h := float64(r.cellSize - padding*2)

	ebitenutil.DrawRect(screen, x, y, w, h, c)
}

// drawUI draws the user interface elements
func (r *GameRenderer) drawUI(screen *ebiten.Image) {
	state := r.game.State

	// Title
	title := "Autonomous Snake Battle"
	if r.paused {
		title += " [PAUSED]"
	}
	ebitenutil.DebugPrintAt(screen, title, 10, 10)

	// Snake stats
	snake0Info := fmt.Sprintf("Green Snake: Length %d, Score %d", state.Snakes[0].Length(), state.Snakes[0].Score)
	if !state.Snakes[0].Alive {
		snake0Info += " [DEAD]"
	}
	ebitenutil.DebugPrintAt(screen, snake0Info, 10, 30)

	// Divider
	ebitenutil.DebugPrintAt(screen, "|", r.screenWidth/2-10, 30)

	snake1Info := fmt.Sprintf("Blue Snake: Length %d, Score %d", state.Snakes[1].Length(), state.Snakes[1].Score)
	if !state.Snakes[1].Alive {
		snake1Info += " [DEAD]"
	}
	ebitenutil.DebugPrintAt(screen, snake1Info, r.screenWidth/2+5, 30)

	// Game over message
	if state.GameOver {
		var msg string
		if state.Winner == 0 {
			msg = "GREEN WINS!"
		} else if state.Winner == 1 {
			msg = "BLUE WINS!"
		} else {
			msg = "TIE!"
		}
		centerX := r.screenWidth/2 - len(msg)*3
		centerY := r.screenHeight / 2
		ebitenutil.DebugPrintAt(screen, msg, centerX, centerY)
	}

	// Bottom stats (first line below board)
	statsY := r.offsetY + r.cfg.BoardHeight*r.cellSize + 8
	statsInfo := fmt.Sprintf("Games: %d   Green Wins: %d   Blue Wins: %d   Ties: %d   Turn: %d",
		r.gamesPlayed, r.wins[0], r.wins[1], r.ties, state.Turn)
	ebitenutil.DebugPrintAt(screen, statsInfo, 10, statsY)

	// Controls help (second line below board)
	helpY := statsY + 18
	help := "Space: Pause   Up/Down: Speed   R: Reset   Q: Quit"
	ebitenutil.DebugPrintAt(screen, help, 10, helpY)
}

// Layout returns the game's screen dimensions
func (r *GameRenderer) Layout(outsideWidth, outsideHeight int) (int, int) {
	return r.screenWidth, r.screenHeight
}

// Run starts the game loop
func (r *GameRenderer) Run() error {
	ebiten.SetWindowSize(r.screenWidth*2, r.screenHeight*2)
	ebiten.SetWindowTitle("Autonomous Snake Battle")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	err := ebiten.RunGame(r)
	if errors.Is(err, ErrQuit) {
		return nil // Normal exit
	}
	return err
}
