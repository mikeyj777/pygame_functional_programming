#!/usr/bin/env python3
"""
Functional Programming Adventure Game
---------------------------------------

This game provides a visual analog for functional programming concepts.
The game world is divided into distinct zones, each representing a different
functional programming operation:

  - Map Woods (ZONE_MAP): Represents the 'map' operation.
      * Effect: Every item in your inventory is transformed (e.g., appended with "_mapped").

  - Filter Falls (ZONE_FILTER): Represents the 'filter' operation.
      * Effect: Only items meeting a condition (e.g., containing the substring "pass") remain.

  - Reduce Ruins (ZONE_REDUCE): Represents the 'reduce' operation.
      * Effect: All inventory items are combined into a single item.

  - Compose Cliffs (ZONE_COMPOSE): Represents function composition.
      * Effect: Each item is modified by composing it with a new prefix ("composed_").

The player (a simple circle) moves between zones using the arrow keys.
Press 'A' to add a random item to the inventory, and press 'E' to apply the zone's
functional effect to the inventory.

To run this script, make sure you have Pygame installed:
    pip install pygame
"""

import pygame
import sys
import random
from dataclasses import dataclass, replace
from typing import Tuple

# --------------------------
# 1. Game Settings and Colors
# --------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30

# Define colors for drawing (RGB tuples)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GREEN = (144, 238, 144)  # For Map Woods
LIGHT_BLUE = (173, 216, 230)   # For Filter Falls
LIGHT_RED = (255, 182, 193)    # For Reduce Ruins
ORANGE = (255, 165, 0)         # For Compose Cliffs
PURPLE = (128, 0, 128)         # Player color

# Define zone type constants for clarity
ZONE_MAP = 0         # Map Woods
ZONE_FILTER = 1      # Filter Falls
ZONE_REDUCE = 2      # Reduce Ruins
ZONE_COMPOSE = 3     # Compose Cliffs

# Map zone types to colors for easy drawing
ZONE_COLORS = {
    ZONE_MAP: LIGHT_GREEN,
    ZONE_FILTER: LIGHT_BLUE,
    ZONE_REDUCE: LIGHT_RED,
    ZONE_COMPOSE: ORANGE,
}

# --------------------------
# 2. Data Structures (Immutable)
# --------------------------
@dataclass(frozen=True)
class Player:
    """Represents the player with a position and an inventory."""
    x: int
    y: int
    inventory: Tuple[str, ...]  # A tuple of item names (immutable)

@dataclass(frozen=True)
class Zone:
    """Represents a zone in the world, defined by its type and area."""
    zone_type: int
    rect: pygame.Rect  # The rectangular area of this zone

@dataclass(frozen=True)
class GameState:
    """Encapsulates the entire game state."""
    player: Player
    zones: Tuple[Zone, ...]
    current_zone: int  # The type of zone where the player currently is

# --------------------------
# 3. World and Zone Setup
# --------------------------
def create_zones() -> Tuple[Zone, ...]:
    """
    Create a set of zones for the game world.
    For simplicity, we divide the screen into 4 quadrants:
      - Top-left: Map Woods
      - Top-right: Filter Falls
      - Bottom-left: Reduce Ruins
      - Bottom-right: Compose Cliffs
    """
    zones = []
    half_width = SCREEN_WIDTH // 2
    half_height = SCREEN_HEIGHT // 2
    zones.append(Zone(ZONE_MAP, pygame.Rect(0, 0, half_width, half_height)))
    zones.append(Zone(ZONE_FILTER, pygame.Rect(half_width, 0, half_width, half_height)))
    zones.append(Zone(ZONE_REDUCE, pygame.Rect(0, half_height, half_width, half_height)))
    zones.append(Zone(ZONE_COMPOSE, pygame.Rect(half_width, half_height, half_width, half_height)))
    return tuple(zones)

def get_zone_for_player(zones: Tuple[Zone, ...], player: Player) -> int:
    """
    Determine the current zone type for the player based on their position.
    Returns the zone type (e.g., ZONE_MAP, ZONE_FILTER, etc.).
    """
    for zone in zones:
        if zone.rect.collidepoint(player.x, player.y):
            return zone.zone_type
    return -1  # In case the player is outside all zones

# --------------------------
# 4. Pure Functions for Game Mechanics
# --------------------------
def move_player(player: Player, dx: int, dy: int) -> Player:
    """
    Pure function: moves the player by dx and dy while keeping them within the screen.
    Returns a new Player instance with the updated position.
    """
    new_x = max(0, min(SCREEN_WIDTH, player.x + dx))
    new_y = max(0, min(SCREEN_HEIGHT, player.y + dy))
    return replace(player, x=new_x, y=new_y)

def apply_zone_effects(state: GameState) -> GameState:
    """
    Pure function: applies the functional programming concept associated with the
    current zone to the player's inventory.
    
    Effects:
      - Map Woods (ZONE_MAP): Each inventory item gets transformed (append '_mapped').
      - Filter Falls (ZONE_FILTER): Only items containing "pass" remain.
      - Reduce Ruins (ZONE_REDUCE): All items are combined into a single item.
      - Compose Cliffs (ZONE_COMPOSE): Each item is modified by prepending 'composed_'.
    
    Returns a new GameState with an updated player's inventory.
    """
    player = state.player
    current_zone = state.current_zone
    new_inventory = list(player.inventory)
    
    if current_zone == ZONE_MAP:
        # Map operation: transform every item.
        new_inventory = tuple(item + "_mapped" for item in new_inventory)
    
    elif current_zone == ZONE_FILTER:
        # Filter operation: only keep items that include "pass".
        new_inventory = tuple(item for item in new_inventory if "pass" in item)
    
    elif current_zone == ZONE_REDUCE:
        # Reduce operation: combine all items into one string.
        if new_inventory:
            combined = "reduced(" + "+".join(new_inventory) + ")"
            new_inventory = (combined,)
    
    elif current_zone == ZONE_COMPOSE:
        # Function composition: compose each item with a new prefix.
        new_inventory = tuple("composed_" + item for item in new_inventory)
    
    # Return a new state with the updated player inventory.
    new_player = replace(player, inventory=new_inventory)
    return replace(state, player=new_player)

# --------------------------
# 5. Rendering Functions
# --------------------------
def render_game(screen: pygame.Surface, state: GameState):
    """
    Renders the current game state:
      - Draws each zone with its corresponding color.
      - Draws the player as a circle.
      - Displays UI text showing the current zone and the player's inventory.
    """
    screen.fill(WHITE)
    
    # Draw each zone as a colored rectangle.
    for zone in state.zones:
        pygame.draw.rect(screen, ZONE_COLORS.get(zone.zone_type, BLACK), zone.rect)
        pygame.draw.rect(screen, BLACK, zone.rect, 2)  # Border for clarity
    
    # Draw the player as a purple circle.
    pygame.draw.circle(screen, PURPLE, (state.player.x, state.player.y), 10)
    
    # Display UI information.
    font = pygame.font.SysFont("Arial", 20)
    zone_names = {
        ZONE_MAP: "Map Woods",
        ZONE_FILTER: "Filter Falls",
        ZONE_REDUCE: "Reduce Ruins",
        ZONE_COMPOSE: "Compose Cliffs"
    }
    zone_text = f"Current Zone: {zone_names.get(state.current_zone, 'Unknown')}"
    inv_text = f"Inventory: {', '.join(state.player.inventory) if state.player.inventory else 'Empty'}"
    
    text_surface = font.render(zone_text, True, BLACK)
    screen.blit(text_surface, (10, 10))
    text_surface = font.render(inv_text, True, BLACK)
    screen.blit(text_surface, (10, 40))
    
    pygame.display.flip()

# --------------------------
# 6. Main Game Loop
# --------------------------
def game_loop():
    """
    The main game loop:
      - Initializes Pygame and the game state.
      - Processes user inputs to move the player and interact with the environment.
      - Updates the game state using pure functions.
      - Renders the current state to the screen.
      
    Controls:
      - Arrow keys: Move the player.
      - A: Add a random item to the player's inventory.
      - E: Apply the zone's effect (functional transformation) on the inventory.
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Functional Programming Adventure")
    clock = pygame.time.Clock()
    
    # Initialize player at the center with an empty inventory.
    initial_player = Player(x=SCREEN_WIDTH // 2, y=SCREEN_HEIGHT // 2, inventory=())
    zones = create_zones()
    current_zone = get_zone_for_player(zones, initial_player)
    state = GameState(player=initial_player, zones=zones, current_zone=current_zone)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                # Handle movement.
                if event.key == pygame.K_LEFT:
                    state = replace(state, player=move_player(state.player, dx=-10, dy=0))
                elif event.key == pygame.K_RIGHT:
                    state = replace(state, player=move_player(state.player, dx=10, dy=0))
                elif event.key == pygame.K_UP:
                    state = replace(state, player=move_player(state.player, dx=0, dy=-10))
                elif event.key == pygame.K_DOWN:
                    state = replace(state, player=move_player(state.player, dx=0, dy=10))
                # Add a new random item to inventory.
                elif event.key == pygame.K_a:
                    new_item = f"item{random.randint(1, 100)}"
                    new_inv = list(state.player.inventory) + [new_item]
                    state = replace(state, player=replace(state.player, inventory=tuple(new_inv)))
                # Execute the zone's functional effect on the inventory.
                elif event.key == pygame.K_e:
                    state = apply_zone_effects(state)
        
        # Update current zone based on the player's new position.
        current_zone = get_zone_for_player(state.zones, state.player)
        state = replace(state, current_zone=current_zone)
        
        # Render the current game state.
        render_game(screen, state)
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

# --------------------------
# 7. Entry Point
# --------------------------
if __name__ == '__main__':
    game_loop()
