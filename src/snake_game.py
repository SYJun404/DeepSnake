import os
import random
import time

import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
from pygame import mixer


class SnakeGame:
    """
    Snake Game Environment

    Action:
        0 = UP
        1 = LEFT
        2 = RIGHT
        3 = DOWN

    Training interface:
        game.reset()
        done, info = game.step(action)

    info:
        snake_size
        snake_head_pos
        prev_snake_head_pos
        food_pos
        food_obtained

    Training mode:
        SnakeGame(..., silent_mode=True)

    Human/UI mode:
        SnakeGame(..., silent_mode=False)

    Optional grid lines on the board:
        SnakeGame(..., grid_lines=True)
    """

    # ==============================================================
    # Colors
    # ==============================================================

    BG_COLOR = (31, 32, 35)

    PANEL_COLOR = (47, 48, 51)
    PANEL_LIGHT = (53, 54, 58)
    PANEL_BORDER = (66, 67, 71)

    BOARD_COLOR = (0, 0, 0)
    BOARD_BORDER = (27, 28, 30)

    GRID_LINE_COLOR = (44, 45, 48)

    TEXT_COLOR = (235, 235, 235)
    MUTED_TEXT = (145, 145, 148)

    GREEN = (92, 255, 105)
    GREEN_BRIGHT = (120, 255, 130)
    GREEN_DARK = (45, 150, 62)

    YELLOW = (255, 195, 35)
    YELLOW_BRIGHT = (255, 215, 70)

    RED = (255, 80, 80)

    DIRECTION_OFFSET = {  # noqa: RUF012
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }
    # ==============================================================
    # Initialization
    # ==============================================================

    def __init__(
        self,
        seed=0,
        board_size=12,
        silent_mode=True,
        grid_lines=False,
    ):
        if board_size < 4:
            raise ValueError("board_size must be at least 4")
        self.board_size = board_size
        self.grid_size = self.board_size**2

        # ==========================================================
        # Game board
        #
        # Keep the same logical board size as the original version.
        # ==========================================================

        self.cell_size = 40

        self.width = self.board_size * self.cell_size
        self.height = self.board_size * self.cell_size

        # ==========================================================
        # UI layout
        # ==============================================================

        self.side_padding = 20
        self.bottom_padding = 20
        # Use an 8x8 board's width as the compact window baseline. Larger
        # boards still get enough room to keep every cell visible.
        self.display_width = max(360, self.width + self.side_padding * 2)
        self.compact_header = self.display_width < 520
        self.header_height = 140 if self.compact_header else 100
        self.board_y = self.header_height
        self.board_x = (self.display_width - self.width) // 2

        self.display_height = self.header_height + self.height + self.bottom_padding

        self.silent_mode = silent_mode

        self.grid_lines = grid_lines

        # ==========================================================
        # Pygame
        # ==============================================================

        self.screen = None

        self.font_title = None
        self.font_label = None
        self.font_value = None
        self.font_small = None
        self.font_overlay = None

        self.sound_eat = None
        self.sound_game_over = None
        self.sound_victory = None

        if not self.silent_mode:
            self._init_ui()

        # ==========================================================
        # Game state
        # ==============================================================

        self.snake = None
        self.non_snake = None

        self.direction = None

        self.score = 0
        self.high_score = 0

        self.food = None

        self.seed_value = seed
        # Keep randomness local to an environment.  Using random.seed() here
        # changes the process-wide RNG, which makes parallel environments
        # affect one another and prevents a training run from being replayed.
        self.rng = random.Random(seed)

        self.reset()

    # ==============================================================
    # UI initialization
    # ==============================================================

    def _init_ui(self):
        pygame.init()

        pygame.display.set_caption("SNAKE AI")

        # SCALED can enlarge a small logical surface to fit a high-resolution
        # desktop. Request the actual window size instead; the demo loop
        # already limits its frame rate. Vsync only applies to SCALED/OpenGL.
        self.screen = pygame.display.set_mode(
            (
                self.display_width,
                self.display_height,
            ),
            pygame.NOFRAME,
        )

        # ----------------------------------------------------------
        # Fonts
        # ----------------------------------------------------------

        self.font_title = pygame.font.SysFont(
            "dingtalkjinbuti",
            38,
            bold=True,
        )

        self.font_label = pygame.font.SysFont(
            "dingtalkjinbuti",
            10,
            bold=True,
        )

        self.font_value = pygame.font.SysFont(
            "dingtalkjinbuti",
            21,
            bold=True,
        )

        self.font_small = pygame.font.SysFont(
            "dingtalkjinbuti",
            13,
        )

        self.font_overlay = pygame.font.SysFont(
            "dingtalkjinbuti",
            42,
            bold=True,
        )

        # ----------------------------------------------------------
        # Optional sound
        # ----------------------------------------------------------

        try:
            mixer.init()

            self.sound_eat = self._load_sound("src/sound/eat.wav")

            self.sound_game_over = self._load_sound("src/sound/game_over.wav")

            self.sound_victory = self._load_sound("src/sound/victory.wav")

        except pygame.error:
            self.sound_eat = None
            self.sound_game_over = None
            self.sound_victory = None

    @staticmethod
    def _load_sound(path):
        try:
            if os.path.exists(path):
                return mixer.Sound(path)
        except pygame.error:
            pass

        return None

    # ==============================================================
    # Game reset
    # ==============================================================

    def reset(self, seed=None):
        """
        Reset the game.

        This keeps the original initial snake layout.
        """

        if seed is not None:
            self.seed_value = seed
            self.rng.seed(seed)
        center = self.board_size // 2

        self.snake = [(center + i, center) for i in range(1, -2, -1)]

        self.non_snake = {
            (row, col)
            for row in range(self.board_size)
            for col in range(self.board_size)
            if (row, col) not in self.snake
        }

        # Original direction
        self.direction = "DOWN"

        self.food = self._generate_food()

        self.score = 0

    # ==============================================================
    # Environment step
    # ==============================================================

    def step(self, action):
        """
        Execute one environment step.

        action:
            0 = UP
            1 = LEFT
            2 = RIGHT
            3 = DOWN

        Returns:
            done, info
        """
        assert (
            self.snake is not None
            and self.non_snake is not None
            and self.direction is not None
        )
        # ----------------------------------------------------------
        # Update direction
        # ----------------------------------------------------------

        self._update_direction(action)
        prev_head = self.snake[0]

        # ----------------------------------------------------------
        # Calculate next position
        # ----------------------------------------------------------

        row, col = self.snake[0]
        dr, dc = self.DIRECTION_OFFSET[self.direction]
        row += dr
        col += dc

        # ----------------------------------------------------------
        # Check food
        # ----------------------------------------------------------

        food_obtained = (row, col) == self.food
        occupied = self.snake if food_obtained else self.snake[:-1]
        collision = (row, col) in occupied or not (
            0 <= row < self.board_size and 0 <= col < self.board_size
        )

        if food_obtained:
            self.score += 10

            if not self.silent_mode:
                self._play_sound(self.sound_eat)

        else:
            # Remove tail.
            if not collision:
                tail = self.snake.pop()
                self.non_snake.add(tail)

        # ----------------------------------------------------------
        # Collision
        # ----------------------------------------------------------

        done = collision

        # ----------------------------------------------------------
        # Move snake
        # ----------------------------------------------------------

        if not done:
            self.snake.insert(0, (row, col))

            self.non_snake.remove((row, col))

        else:
            if not self.silent_mode:
                if len(self.snake) < self.grid_size:
                    self._play_sound(self.sound_game_over)

                else:
                    self._play_sound(self.sound_victory)

        # ----------------------------------------------------------
        # Generate new food
        # ----------------------------------------------------------

        if food_obtained:
            self.food = self._generate_food()

        # ----------------------------------------------------------
        # Keep original info interface
        # ----------------------------------------------------------

        success = len(self.snake) == self.grid_size
        done = done or success
        info = {
            "snake_size": len(self.snake),
            "snake_head_pos": np.array(self.snake[0]),
            "prev_snake_head_pos": np.array(prev_head),
            "food_pos": np.array(self.food),
            "food_obtained": food_obtained,
            "is_success": success,
            "termination_reason": "success"
            if success
            else "collision"
            if collision
            else None,
        }

        return done, info

    # ==============================================================
    # Direction
    # ==============================================================

    def _update_direction(self, action):
        """
        Action mapping:

            0 -> UP
            1 -> LEFT
            2 -> RIGHT
            3 -> DOWN

        Prevent 180-degree turns.
        """

        if action == 0:
            if self.direction != "DOWN":
                self.direction = "UP"

        elif action == 1:
            if self.direction != "RIGHT":
                self.direction = "LEFT"

        elif action == 2:
            if self.direction != "LEFT":
                self.direction = "RIGHT"

        elif action == 3 and self.direction != "UP":
            self.direction = "DOWN"

    # ==============================================================
    # Food
    # ==============================================================

    def _generate_food(self):

        if self.non_snake:
            return self.rng.choice(sorted(self.non_snake))

        # Original fallback.
        return None

    # ==============================================================
    # Drawing utilities
    # ==============================================================

    def _draw_text(
        self,
        text,
        font,
        color,
        center=None,
        topleft=None,
    ):
        surface = font.render(
            text,
            True,
            color,
        )
        if self.screen == None:
            return
        if center is not None:
            rect = surface.get_rect(center=center)

        else:
            rect = surface.get_rect(topleft=topleft)

        self.screen.blit(
            surface,
            rect,
        )

        return rect

    def _rounded_rect(
        self,
        rect,
        color,
        radius=10,
        border=0,
        border_color=None,
    ):
        if self.screen == None:
            return
        pygame.draw.rect(
            self.screen,
            color,
            rect,
            border_radius=radius,
        )

        if border > 0 and border_color is not None:
            pygame.draw.rect(
                self.screen,
                border_color,
                rect,
                width=border,
                border_radius=radius,
            )

    # ==============================================================
    # Glow
    # ==============================================================

    def _draw_glow_circle(
        self,
        center,
        radius,
        color,
    ):
        """
        Draw a simple soft glow.
        """
        if self.screen == None:
            return
        glow_surface = pygame.Surface(
            (
                radius * 4,
                radius * 4,
            ),
            pygame.SRCALPHA,
        )

        cx = glow_surface.get_width() // 2
        cy = glow_surface.get_height() // 2

        for i in range(4, 0, -1):
            current_radius = radius + i * 4

            alpha = 8 + i * 8

            pygame.draw.circle(
                glow_surface,
                (
                    color[0],
                    color[1],
                    color[2],
                    alpha,
                ),
                (cx, cy),
                current_radius,
            )

        self.screen.blit(
            glow_surface,
            (
                center[0] - cx,
                center[1] - cy,
            ),
        )

    # ==============================================================
    # HUD
    # ==============================================================

    def _draw_stat_card(
        self,
        x,
        width,
        label,
        value,
        y=20,
    ):
        card_rect = pygame.Rect(
            x,
            y,
            width,
            58,
        )

        self._rounded_rect(
            card_rect,
            (28, 29, 32),
            radius=8,
            border=1,
            border_color=(64, 65, 68),
        )

        self._draw_text(
            label,
            self.font_label,
            self.LABEL_COLOR if hasattr(self, "LABEL_COLOR") else self.MUTED_TEXT,
            center=(
                card_rect.centerx,
                card_rect.y + 13,
            ),
        )

        self._draw_text(
            str(value),
            self.font_value,
            self.YELLOW,
            center=(
                card_rect.centerx,
                card_rect.y + 38,
            ),
        )

    def _draw_hud(self):
        if self.screen == None:
            return
        if self.compact_header:
            # A dedicated title row leaves the full width for readable cards.
            self._draw_text(
                "DeepSnake",
                self.font_title,
                self.GREEN,
                center=(self.display_width // 2, 33),
            )
            gap = 8
            available = self.display_width - 2 * self.side_padding
            card_width = (available - 2 * gap) // 3
            cards = (
                ("HIGH SCORE", self.high_score),
                ("SCORE", self.score),
                ("LEVEL", 1 + self.score // 50),
            )
            for index, (label, value) in enumerate(cards):
                x = self.side_padding + index * (card_width + gap)
                width = (
                    self.display_width - self.side_padding - x
                    if index == 2
                    else card_width
                )
                self._draw_stat_card(x, width, label, value, y=62)
            return
        # ----------------------------------------------------------
        # SNAKE title
        # ----------------------------------------------------------

        # Soft glow behind title.
        glow_surface = pygame.Surface(
            (170, 60),
            pygame.SRCALPHA,
        )

        glow_font = pygame.font.SysFont(
            "dingtalkjinbuti",
            38,
            bold=True,
        )

        glow_text = glow_font.render(
            "SNAKE",
            True,
            (
                self.GREEN[0],
                self.GREEN[1],
                self.GREEN[2],
                30,
            ),
        )

        glow_surface.blit(
            glow_text,
            (
                15,
                8,
            ),
        )

        self.screen.blit(
            glow_surface,
            (4, 20),
        )

        # Actual title.
        self._draw_text(
            "SNAKE",
            self.font_title,
            self.GREEN,
            topleft=(20, 26),
        )

        # ----------------------------------------------------------
        # Cards
        # ----------------------------------------------------------

        self._draw_stat_card(
            self.display_width - 298,
            104,
            "HIGH SCORE",
            self.high_score,
        )

        self._draw_stat_card(
            self.display_width - 185,
            74,
            "SCORE",
            self.score,
        )

        # Level increases every 50 points.
        level = 1 + self.score // 50

        self._draw_stat_card(
            self.display_width - 102,
            76,
            "LEVEL",
            level,
        )

    # ==============================================================
    # Board
    # ==============================================================

    def draw_board(self):
        if self.screen == None:
            return
        board_rect = pygame.Rect(
            self.board_x,
            self.board_y,
            self.width,
            self.height,
        )

        # Outer subtle shadow.
        shadow_rect = board_rect.inflate(
            8,
            8,
        )

        self._rounded_rect(
            shadow_rect,
            (23, 24, 26),
            radius=7,
        )

        # Black board.
        self._rounded_rect(
            board_rect,
            self.BOARD_COLOR,
            radius=5,
            border=1,
            border_color=self.BOARD_BORDER,
        )

        # Optional grid lines between cells.
        if self.grid_lines:
            for i in range(1, self.board_size):
                x = self.board_x + i * self.cell_size

                pygame.draw.line(
                    self.screen,
                    self.GRID_LINE_COLOR,
                    (x, self.board_y),
                    (x, self.board_y + self.height),
                    1,
                )

                y = self.board_y + i * self.cell_size

                pygame.draw.line(
                    self.screen,
                    self.GRID_LINE_COLOR,
                    (self.board_x, y),
                    (self.board_x + self.width, y),
                    1,
                )

    # ==============================================================
    # Snake rendering
    # ==============================================================

    def draw_snake(self):
        if self.screen == None:
            return

        if not self.snake:
            return

        # ----------------------------------------------------------
        # Body
        # ----------------------------------------------------------

        body = self.snake[1:]

        for index, (r, c) in enumerate(reversed(body)):
            body_index = index + 1

            x = self.board_x + c * self.cell_size

            y = self.board_y + r * self.cell_size

            padding = 5

            rect = pygame.Rect(
                x + padding,
                y + padding,
                self.cell_size - padding * 2,
                self.cell_size - padding * 2,
            )

            ratio = body_index / max(len(body), 1)

            red = int(42 + 38 * ratio)

            green = int(135 + 110 * ratio)

            blue = int(60 + 60 * ratio)

            color = (
                red,
                green,
                blue,
            )

            self._rounded_rect(
                rect,
                color,
                radius=7,
            )

            # Small highlight.
            highlight_rect = pygame.Rect(
                rect.x + 5,
                rect.y + 4,
                max(
                    6,
                    rect.width // 3,
                ),
                3,
            )

            highlight_color = (
                min(
                    255,
                    color[0] + 25,
                ),
                min(
                    255,
                    color[1] + 25,
                ),
                min(
                    255,
                    color[2] + 25,
                ),
            )

            pygame.draw.rect(
                self.screen,
                highlight_color,
                highlight_rect,
                border_radius=2,
            )

        # ----------------------------------------------------------
        # Head
        # ----------------------------------------------------------

        head_r, head_c = self.snake[0]

        head_x = self.board_x + head_c * self.cell_size

        head_y = self.board_y + head_r * self.cell_size

        head_rect = pygame.Rect(
            head_x + 4,
            head_y + 4,
            self.cell_size - 8,
            self.cell_size - 8,
        )

        # Head.
        self._rounded_rect(
            head_rect,
            self.GREEN_BRIGHT,
            radius=8,
        )

        # Head highlight.
        highlight = pygame.Rect(
            head_rect.x + 6,
            head_rect.y + 5,
            11,
            4,
        )

        pygame.draw.rect(
            self.screen,
            (215, 255, 218),
            highlight,
            border_radius=2,
        )

        # ----------------------------------------------------------
        # Eyes
        # ----------------------------------------------------------

        eye_color = (
            15,
            45,
            18,
        )

        if self.direction == "UP":
            eyes = [
                (
                    head_rect.centerx - 7,
                    head_rect.y + 10,
                ),
                (
                    head_rect.centerx + 7,
                    head_rect.y + 10,
                ),
            ]

        elif self.direction == "DOWN":
            eyes = [
                (
                    head_rect.centerx - 7,
                    head_rect.bottom - 10,
                ),
                (
                    head_rect.centerx + 7,
                    head_rect.bottom - 10,
                ),
            ]

        elif self.direction == "LEFT":
            eyes = [
                (
                    head_rect.x + 10,
                    head_rect.centery - 7,
                ),
                (
                    head_rect.x + 10,
                    head_rect.centery + 7,
                ),
            ]

        else:
            eyes = [
                (
                    head_rect.right - 10,
                    head_rect.centery - 7,
                ),
                (
                    head_rect.right - 10,
                    head_rect.centery + 7,
                ),
            ]

        for eye in eyes:
            pygame.draw.circle(
                self.screen,
                eye_color,
                eye,
                3,
            )

    # ==============================================================
    # Food rendering
    # ==============================================================

    def draw_food(self):
        if self.snake == None or self.screen == None or self.food == None:
            return

        if len(self.snake) >= self.grid_size:
            return

        r, c = self.food

        center_x = self.board_x + c * self.cell_size + self.cell_size // 2

        center_y = self.board_y + r * self.cell_size + self.cell_size // 2

        radius = self.cell_size // 4

        # Glow.
        self._draw_glow_circle(
            (
                center_x,
                center_y,
            ),
            radius,
            self.YELLOW,
        )

        # Main food.
        pygame.draw.circle(
            self.screen,
            self.YELLOW,
            (
                center_x,
                center_y,
            ),
            radius,
        )

        # Inner bright part.
        pygame.draw.circle(
            self.screen,
            self.YELLOW_BRIGHT,
            (
                center_x - 2,
                center_y - 2,
            ),
            max(
                3,
                radius // 2,
            ),
        )

        # Small highlight.
        pygame.draw.circle(
            self.screen,
            (255, 245, 190),
            (
                center_x - radius // 3,
                center_y - radius // 3,
            ),
            2,
        )

    # ==============================================================
    # Main render
    # ==============================================================

    def render(self):

        # Important:
        #
        # In training mode:
        #     render() does absolutely nothing.
        #
        # This prevents pygame from slowing down model training.

        if self.silent_mode or self.screen is None:
            return

        # ----------------------------------------------------------
        # Background
        # ----------------------------------------------------------

        self.screen.fill(self.BG_COLOR)

        # ----------------------------------------------------------
        # Main panel
        # ----------------------------------------------------------

        panel_rect = pygame.Rect(
            8,
            8,
            self.display_width - 16,
            self.display_height - 16,
        )

        self._rounded_rect(
            panel_rect,
            self.PANEL_COLOR,
            radius=16,
            border=1,
            border_color=self.PANEL_BORDER,
        )

        # ----------------------------------------------------------
        # Header
        # ----------------------------------------------------------

        header_rect = pygame.Rect(
            12,
            12,
            self.display_width - 24,
            self.header_height - 10,
        )

        self._rounded_rect(
            header_rect,
            self.PANEL_LIGHT,
            radius=13,
        )

        # ----------------------------------------------------------
        # Game
        # ----------------------------------------------------------

        self.draw_board()

        self.draw_food()

        self.draw_snake()

        self._draw_hud()

        # Note: presentation (pygame.display.flip) is done once per
        # frame by the caller (run_human_game), never here.

    # ==============================================================
    # Welcome screen
    # ==============================================================

    def _button_rect(self):
        width = 170
        height = 48

        return pygame.Rect(
            self.display_width // 2 - width // 2,
            self.display_height // 2 + 40,
            width,
            height,
        )

    def _draw_button(
        self,
        text,
    ):

        rect = self._button_rect()

        mouse_pos = pygame.mouse.get_pos()

        hovered = rect.collidepoint(mouse_pos)

        if hovered:
            fill = (
                65,
                175,
                77,
            )
        else:
            fill = (
                48,
                137,
                59,
            )

        self._rounded_rect(
            rect,
            fill,
            radius=10,
            border=1,
            border_color=(
                100,
                215,
                110,
            ),
        )

        self._draw_text(
            text,
            self.font_value,
            (
                240,
                255,
                242,
            ),
            center=rect.center,
        )

        return rect

    def draw_welcome_screen(self):

        if self.silent_mode or self.screen is None:
            return

        self.screen.fill(self.BG_COLOR)

        # Main panel.
        panel_rect = pygame.Rect(
            8,
            8,
            self.display_width - 16,
            self.display_height - 16,
        )

        self._rounded_rect(
            panel_rect,
            self.PANEL_COLOR,
            radius=16,
            border=1,
            border_color=self.PANEL_BORDER,
        )

        # Title.
        self._draw_text(
            "Classic Snake",
            self.font_title,
            self.GREEN,
            center=(
                self.display_width // 2,
                self.display_height // 2 - 80,
            ),
        )

        # Subtitle.
        self._draw_text(
            "Human vs. AI",
            self.font_small,
            self.MUTED_TEXT,
            center=(
                self.display_width // 2,
                self.display_height // 2 - 42,
            ),
        )

        # Start button.
        self._draw_button("START")

    # ==============================================================
    # Game over
    # ==============================================================

    def _draw_overlay(
        self,
        alpha=180,
    ):
        if self.screen == None:
            return
        overlay = pygame.Surface(
            (
                self.display_width,
                self.display_height,
            ),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (
                0,
                0,
                0,
                alpha,
            )
        )

        self.screen.blit(
            overlay,
            (0, 0),
        )

    def draw_game_over_screen(self, button_text="RETRY"):

        if self.silent_mode or self.screen is None or self.snake is None:
            return

        # Draw final game frame.
        self.render()

        # Dark overlay.
        self._draw_overlay(180)

        # Win / game over.
        if len(self.snake) >= self.grid_size:
            title = "VICTORY"
            title_color = self.GREEN

        else:
            title = "GAME OVER"
            title_color = self.RED

        self._draw_text(
            title,
            self.font_overlay,
            title_color,
            center=(
                self.display_width // 2,
                self.display_height // 2 - 70,
            ),
        )

        self._draw_text(
            f"SCORE  {self.score}",
            self.font_value,
            self.TEXT_COLOR,
            center=(
                self.display_width // 2,
                self.display_height // 2 - 15,
            ),
        )

        self._draw_button(button_text)

    # ==============================================================
    # Countdown
    # ==============================================================

    def draw_countdown(
        self,
        number,
    ):

        if self.silent_mode or self.screen is None:
            return

        self.render()

        self._draw_overlay(100)

        self._draw_text(
            str(number),
            self.font_overlay,
            self.GREEN,
            center=(
                self.display_width // 2,
                self.display_height // 2,
            ),
        )

    # ==============================================================
    # Mouse button detection
    # ==============================================================

    def is_mouse_on_button(self):

        if self.screen is None:
            return False

        rect = self._button_rect()

        return rect.collidepoint(pygame.mouse.get_pos())

    # ==============================================================
    # Sound
    # ==============================================================

    @staticmethod
    def _play_sound(sound):

        if sound is None:
            return

        try:
            sound.play()
        except pygame.error:
            pass


# ======================================================================
# Human playable mode
# ======================================================================


def run_human_game():
    seed = random.randint(
        0,
        1_000_000_000,
    )

    game = SnakeGame(seed=seed, board_size=12, silent_mode=False, grid_lines=True)

    clock = pygame.time.Clock()

    game_state = "welcome"

    action = -1

    # --------------------------------------------------------------
    # Game speed
    #
    # 0.15 seconds = approximately 6.67 steps/sec
    #
    # This keeps the original training/game behavior.
    # --------------------------------------------------------------

    update_interval = 0.15

    last_update = time.time()

    countdown_start = None

    running = True

    while running:
        # ==========================================================
        # Event handling
        #
        # IMPORTANT:
        #
        # render() no longer calls pygame.event.get().
        #
        # All events are handled here.
        # ==========================================================

        needs_flip = False

        for event in pygame.event.get():
            # ------------------------------------------------------
            # Quit
            # ------------------------------------------------------

            if event.type == pygame.QUIT:
                running = False

                continue

            # ======================================================
            # Welcome
            # ======================================================

            if game_state == "welcome":
                if event.type == pygame.KEYDOWN:
                    if event.key in (
                        pygame.K_RETURN,
                        pygame.K_SPACE,
                    ):
                        game_state = "countdown"

                        countdown_start = time.time()

                elif event.type == pygame.MOUSEBUTTONDOWN:  # noqa: SIM102
                    if event.button == 1:
                        button = game._button_rect()

                        if button.collidepoint(event.pos):
                            game_state = "countdown"

                            countdown_start = time.time()

            # ======================================================
            # Running
            # ======================================================

            elif game_state == "running":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 0

                    elif event.key == pygame.K_LEFT:
                        action = 1

                    elif event.key == pygame.K_RIGHT:
                        action = 2

                    elif event.key == pygame.K_DOWN:
                        action = 3
                    elif event.key == pygame.K_SPACE:
                        # Pause the game
                        game_state = "paused"

            # ======================================================
            # Game over
            # ======================================================

            elif game_state == "game_over":
                if event.type == pygame.KEYDOWN:
                    if event.key in (
                        pygame.K_RETURN,
                        pygame.K_SPACE,
                    ):
                        game.reset()

                        action = -1

                        game_state = "countdown"

                        countdown_start = time.time()

                elif event.type == pygame.MOUSEBUTTONDOWN:  # noqa: SIM102
                    if event.button == 1:
                        button = game._button_rect()

                        if button.collidepoint(event.pos):
                            game.reset()

                            action = -1

                            game_state = "countdown"

                            countdown_start = time.time()

            elif game_state == "paused" and event.type == pygame.KEYDOWN:  # noqa: SIM102
                if event.key == pygame.K_SPACE:
                    game_state = "running"

        # ==========================================================
        # Welcome screen
        # ==========================================================

        if game_state == "welcome":
            game.draw_welcome_screen()

            needs_flip = True

        # ==========================================================
        # Countdown
        # ==========================================================

        elif game_state == "countdown":
            if countdown_start is None:
                countdown_start = time.time()

            elapsed = time.time() - countdown_start

            if elapsed < 1:
                number = 3

            elif elapsed < 2:
                number = 2

            elif elapsed < 3:
                number = 1

            else:
                game_state = "running"

                action = -1

                last_update = time.time()

                continue

            game.draw_countdown(number)

            needs_flip = True

        # ==========================================================
        # Running
        # ==========================================================

        elif game_state == "running":
            now = time.time()

            if now - last_update >= update_interval:
                done, _ = game.step(action)  # pyright: ignore[reportGeneralTypeIssues]

                last_update = now

                if done:
                    game.high_score = max(
                        game.high_score,
                        game.score,
                    )

                    game_state = "game_over"

                game.render()

                needs_flip = True

        # ==========================================================
        # Game over
        # ==========================================================

        elif game_state == "game_over":
            game.draw_game_over_screen()

            needs_flip = True

        # ==========================================================
        # Present the frame exactly once
        #
        # The draw calls above never flip by themselves; this is the
        # single place the back buffer is presented. Flipping only
        # when something was drawn also keeps the "running" state
        # presenting at the game tick rate instead of 60 FPS.
        # ==========================================================

        if needs_flip:
            pygame.display.flip()

        # ==========================================================
        # Limit UI loop to 60 FPS
        # ==========================================================

        clock.tick(60)

    pygame.quit()


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    run_human_game()
