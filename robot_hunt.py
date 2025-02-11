# =============================================================================
# Import Required Libraries
# =============================================================================

import sys                # For system-level parameters (if needed)
import pygame             # For window creation, input, and event handling
from pygame.locals import *  # Provides constants like DOUBLEBUF and OPENGL
from OpenGL.GL import *   # OpenGL functions for rendering primitives
from OpenGL.GLU import *  # GLU functions for projection and view setup
import random             # For generating random positions and sizes
import numpy as np        # For vector math and numeric operations

# =============================================================================
# Global Constants
# =============================================================================

SCREEN_WIDTH = 800      # Width of the display window (in pixels)
SCREEN_HEIGHT = 600     # Height of the display window (in pixels)

# Define the simulation arena extents (on the X-Z plane)
SIMULATION_AREA = 50

# Number of obstacles to place in the arena
NUM_OBSTACLES = 5

# =============================================================================
# Custom Shape Drawing Function (No GLUT)
# =============================================================================

def draw_cube(size):
    """
    Draw a cube centered at the origin using OpenGL primitives (GL_QUADS).

    Parameters:
        size (float): The edge length of the cube.
    """
    half_size = size / 2.0  # Calculate half the cube size for centering
    glBegin(GL_QUADS)
    # Front face (z positive)
    glVertex3f(-half_size, -half_size,  half_size)
    glVertex3f( half_size, -half_size,  half_size)
    glVertex3f( half_size,  half_size,  half_size)
    glVertex3f(-half_size,  half_size,  half_size)
    
    # Back face (z negative)
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(-half_size,  half_size, -half_size)
    glVertex3f( half_size,  half_size, -half_size)
    glVertex3f( half_size, -half_size, -half_size)
    
    # Top face (y positive)
    glVertex3f(-half_size,  half_size, -half_size)
    glVertex3f(-half_size,  half_size,  half_size)
    glVertex3f( half_size,  half_size,  half_size)
    glVertex3f( half_size,  half_size, -half_size)
    
    # Bottom face (y negative)
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f( half_size, -half_size, -half_size)
    glVertex3f( half_size, -half_size,  half_size)
    glVertex3f(-half_size, -half_size,  half_size)
    
    # Right face (x positive)
    glVertex3f( half_size, -half_size, -half_size)
    glVertex3f( half_size,  half_size, -half_size)
    glVertex3f( half_size,  half_size,  half_size)
    glVertex3f( half_size, -half_size,  half_size)
    
    # Left face (x negative)
    glVertex3f(-half_size, -half_size, -half_size)
    glVertex3f(-half_size, -half_size,  half_size)
    glVertex3f(-half_size,  half_size,  half_size)
    glVertex3f(-half_size,  half_size, -half_size)
    glEnd()

# =============================================================================
# Helper Functions
# =============================================================================

def collides_with_obstacles(pos, obstacles, obj_size):
    """
    Check if a given 2D position on the X-Z plane for an object of a specified size
    collides with any obstacles in the environment.

    Parameters:
        pos (array-like): [x, z] coordinates of the object's center.
        obstacles (list): A list of Obstacle objects.
        obj_size (float): The edge length of the object.

    Returns:
        bool: True if a collision is detected, otherwise False.
    """
    for obs in obstacles:
        # Compute half sizes for both object and obstacle
        half_obs = obs.size / 2.0
        half_obj = obj_size / 2.0
        # Check for overlap using axis-aligned bounding box (AABB) collision detection
        if (abs(pos[0] - obs.position[0]) < (half_obs + half_obj) and
            abs(pos[1] - obs.position[1]) < (half_obs + half_obj)):
            return True
    return False

def line_intersects_box(p1, p2, box_center, box_size):
    """
    Determine if the line segment between points p1 and p2 intersects an
    axis-aligned square (representing an obstacle) on the X-Z plane.

    The function samples points along the line segment and checks whether any of
    these points lie within the obstacle's bounds.

    Parameters:
        p1 (np.array): Starting point [x, z] of the line.
        p2 (np.array): Ending point [x, z] of the line.
        box_center (np.array): [x, z] coordinates of the obstacle's center.
        box_size (float): The edge length of the obstacle.

    Returns:
        bool: True if the line intersects the box, False otherwise.
    """
    half_size = box_size / 2.0
    left   = box_center[0] - half_size
    right  = box_center[0] + half_size
    top    = box_center[1] - half_size   # Smaller z-value
    bottom = box_center[1] + half_size   # Larger z-value

    # Quick rejection: if both endpoints lie completely on one side of the box.
    if ((p1[0] < left and p2[0] < left) or 
        (p1[0] > right and p2[0] > right) or 
        (p1[1] < top and p2[1] < top) or 
        (p1[1] > bottom and p2[1] > bottom)):
        return False

    # Sample points along the line segment.
    steps = 20  # Number of sample points
    for i in range(steps + 1):
        t = i / steps  # Parameter from 0 to 1
        point = p1 + (p2 - p1) * t  # Linear interpolation
        if left <= point[0] <= right and top <= point[1] <= bottom:
            return True
    return False

# =============================================================================
# Classes for Simulation Objects
# =============================================================================

class Robot:
    """
    A robot that uses a simple hiding-and-seeking algorithm to hunt its enemy.
    """
    def __init__(self, position, color):
        """
        Initialize the Robot.

        Parameters:
            position (list or array): [x, z] starting position.
            color (tuple): RGB color values (each between 0 and 1).
        """
        self.position = np.array(position, dtype=float)
        self.color = color
        self.speed = 0.2      # Movement speed per update
        self.size = 2.0       # Cube edge length for the robot

    def update(self, enemy, obstacles):
        """
        Update the robot's position based on whether it sees the enemy.

        If the enemy is visible (no obstacles block the view), the robot moves
        directly toward the enemy. Otherwise, it computes a hiding spot behind
        an obstacle and moves toward that spot.

        Parameters:
            enemy (Robot): The enemy robot.
            obstacles (list): A list of Obstacle objects.
        """
        if self.can_see(enemy, obstacles):
            # If the enemy is visible, head directly toward it.
            target = enemy.position.copy()
        else:
            # Compute a hiding spot behind obstacles.
            target = self.find_hiding_spot(enemy, obstacles)
        
        # Compute a normalized direction vector toward the target.
        direction = target - self.position
        dist = np.linalg.norm(direction)
        if dist > 0.01:  # Avoid division by zero
            direction /= dist
            step = direction * self.speed
            new_position = self.position + step
            # Only update if the move does not collide with obstacles.
            if not collides_with_obstacles(new_position, obstacles, self.size):
                self.position = new_position

    def can_see(self, enemy, obstacles):
        """
        Determine if the enemy is visible (i.e. not obscured by any obstacle).

        Parameters:
            enemy (Robot): The enemy robot.
            obstacles (list): A list of Obstacle objects.

        Returns:
            bool: True if the enemy is visible, otherwise False.
        """
        p1 = self.position
        p2 = enemy.position
        for obs in obstacles:
            if line_intersects_box(p1, p2, obs.position, obs.size):
                return False
        return True

    def find_hiding_spot(self, enemy, obstacles):
        """
        Compute candidate hiding spots behind each obstacle (on the side opposite
        to the enemy) and choose the one that is closest to this robot.

        Parameters:
            enemy (Robot): The enemy robot.
            obstacles (list): A list of Obstacle objects.

        Returns:
            np.array: The [x, z] coordinates of the chosen hiding spot.
        """
        best_spot = None
        best_distance = float('inf')
        for obs in obstacles:
            # Compute the vector from the enemy to the obstacle.
            to_obs = obs.position - enemy.position
            norm = np.linalg.norm(to_obs)
            if norm == 0:
                continue  # Avoid division by zero if enemy overlaps with the obstacle
            to_obs /= norm
            # Offset so that the hiding spot is just outside the obstacle.
            offset = (obs.size / 2 + self.size / 2 + 1.0)
            candidate = obs.position + to_obs * offset
            d = np.linalg.norm(candidate - self.position)
            if d < best_distance:
                best_distance = d
                best_spot = candidate
        if best_spot is None:
            # Fallback: if no hiding spot was computed, stay in place.
            return self.position.copy()
        return best_spot

    def draw(self):
        """
        Render the robot as a colored cube using our custom draw_cube function.
        The cube is translated so that it sits on the ground.
        """
        glColor3fv(self.color)
        glPushMatrix()
        # Translate so that the cube's bottom touches the ground.
        glTranslatef(self.position[0], self.size / 2, self.position[1])
        draw_cube(self.size)
        glPopMatrix()


class Obstacle:
    """
    An obstacle that the robots can use for cover. Rendered as a grey cube.
    """
    def __init__(self, position, size):
        """
        Initialize the Obstacle.

        Parameters:
            position (list or array): [x, z] center position.
            size (float): Cube edge length.
        """
        self.position = np.array(position, dtype=float)
        self.size = size

    def draw(self):
        """
        Render the obstacle as a grey cube using our custom draw_cube function.
        """
        glColor3f(0.5, 0.5, 0.5)  # Grey color
        glPushMatrix()
        glTranslatef(self.position[0], self.size / 2, self.position[1])
        draw_cube(self.size)
        glPopMatrix()


def draw_ground():
    """
    Draw a flat green ground plane covering the simulation arena.
    """
    glColor3f(0, 0.6, 0)  # Green color for the ground
    glBegin(GL_QUADS)
    glVertex3f(-SIMULATION_AREA, 0, -SIMULATION_AREA)
    glVertex3f(-SIMULATION_AREA, 0,  SIMULATION_AREA)
    glVertex3f( SIMULATION_AREA, 0,  SIMULATION_AREA)
    glVertex3f( SIMULATION_AREA, 0, -SIMULATION_AREA)
    glEnd()

# =============================================================================
# Main Simulation Loop
# =============================================================================

def main():
    """
    Set up the simulation environment, initialize robots and obstacles,
    and enter the main loop to update and render the scene.
    """
    # -------------------------------
    # Pygame and OpenGL Initialization
    # -------------------------------
    pygame.init()  # Initialize Pygame
    display = (SCREEN_WIDTH, SCREEN_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    # Enable depth testing so nearer objects occlude farther ones.
    glEnable(GL_DEPTH_TEST)
    
    # Set the background (clear) color (light blue sky).
    glClearColor(0.5, 0.8, 1.0, 1.0)
    
    # Set up a perspective projection.
    gluPerspective(45, (display[0] / display[1]), 0.1, 200.0)
    
    # Set up the camera view.
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Position the camera to view the scene from an elevated angle.
    gluLookAt(0, 60, 60,   # Eye (camera) position.
              0, 0, 0,     # Look-at point.
              0, 1, 0)     # Up vector.
    
    # -------------------------------
    # Create Obstacles
    # -------------------------------
    obstacles = []
    for i in range(NUM_OBSTACLES):
        # Choose random positions (keeping obstacles within bounds).
        x = random.uniform(-SIMULATION_AREA + 10, SIMULATION_AREA - 10)
        z = random.uniform(-SIMULATION_AREA + 10, SIMULATION_AREA - 10)
        size = random.uniform(5, 10)
        obstacles.append(Obstacle([x, z], size))
    
    # -------------------------------
    # Helper Function: Get Random Valid Position
    # -------------------------------
    def get_random_position():
        """
        Generate a random [x, z] position within the simulation area that does not
        collide with any obstacles.
        
        Returns:
            np.array: A valid position.
        """
        while True:
            pos = np.array([
                random.uniform(-SIMULATION_AREA, SIMULATION_AREA),
                random.uniform(-SIMULATION_AREA, SIMULATION_AREA)
            ])
            if not collides_with_obstacles(pos, obstacles, 2.0):
                return pos

    # -------------------------------
    # Create Robots
    # -------------------------------
    # Instantiate two robots with distinct colors.
    robot1 = Robot(get_random_position(), (1, 0, 0))  # Red robot
    robot2 = Robot(get_random_position(), (0, 0, 1))  # Blue robot

    clock = pygame.time.Clock()  # Clock to control the frame rate

    # -------------------------------
    # Main Simulation Loop
    # -------------------------------
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Limit to 60 frames per second

        # Process events (e.g., quit event)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update robot positions (each treats the other as its enemy)
        robot1.update(robot2, obstacles)
        robot2.update(robot1, obstacles)

        # Check for collision between robots ("one touch kills")
        if np.linalg.norm(robot1.position - robot2.position) < robot1.size:
            print("Robot collision! Resetting positions.")
            # Reset robots to new random valid positions.
            robot1.position = get_random_position()
            robot2.position = get_random_position()

        # Clear the screen and the depth buffer.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw the ground, obstacles, and robots.
        draw_ground()
        for obs in obstacles:
            obs.draw()
        robot1.draw()
        robot2.draw()
        
        # Swap the buffers to display the current frame.
        pygame.display.flip()

    # Quit Pygame when the simulation loop ends.
    pygame.quit()

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
