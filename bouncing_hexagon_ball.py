import pygame
import math
from pygame.math import Vector2

# ----- Simulation Constants -----
WIDTH, HEIGHT = 800, 600
FPS = 60

# Ball parameters
BALL_RADIUS = 10
# A coefficient of restitution less than 1 loses some energy at each bounce.
RESTITUTION = 0.9  
# Friction applied during a collision (reduces tangential speed)
WALL_FRICTION = 0.2  
# "Air friction" factor (applied each frame to gradually slow the ball)
AIR_FRICTION_COEFF = 0.1  

# Gravity (in pixels per second^2)
GRAVITY = 1000  

# Hexagon parameters
HEXAGON_RADIUS = 200
HEXAGON_CENTER = Vector2(WIDTH/2, HEIGHT/2)
# The hexagon rotates at a constant angular speed (in radians per second)
ANGULAR_VELOCITY = 1.0  

# ----- Ball Initialization -----
# We use a dictionary for simplicity.
ball = {
    'pos': Vector2(HEXAGON_CENTER.x, HEXAGON_CENTER.y - 100),
    'vel': Vector2(200, 0),
    'radius': BALL_RADIUS
}

# Starting rotation angle for the hexagon (in radians)
hex_angle = 0


def get_hexagon_vertices(center, radius, angle):
    """
    Returns a list of 6 vertices for a regular hexagon centered at 'center'
    with distance 'radius' from the center. The polygon is rotated by 'angle'.
    """
    vertices = []
    for i in range(6):
        theta = angle + i * (2 * math.pi / 6)
        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        vertices.append(Vector2(x, y))
    return vertices


def handle_collision(ball, hex_vertices, hex_center, hex_angular_velocity, restitution, wall_friction):
    """
    Check the ball against each edge of the hexagon.
    If the ball is overlapping an edge, push it out along the wall’s inward normal
    and update its velocity using a collision response that takes into account
    the wall’s motion (due to rotation), energy loss (restitution), and friction.
    Returns True if any collision was processed.
    """
    collided = False

    # Check each edge (each pair of consecutive vertices)
    for i in range(len(hex_vertices)):
        A = hex_vertices[i]
        B = hex_vertices[(i + 1) % len(hex_vertices)]
        edge = B - A

        # --- Find the point on the edge closest to the ball ---
        # Parameterize the edge: closest point = A + t*(B-A) with t clamped to [0,1]
        t = (ball['pos'] - A).dot(edge) / edge.length_squared()
        t = max(0, min(1, t))
        closest_point = A + edge * t

        # --- Check for collision (overlap) ---
        dist = (ball['pos'] - closest_point).length()
        if dist < ball['radius']:
            collided = True
            penetration = ball['radius'] - dist

            # --- Determine the wall's inward normal ---
            # For a polygon with vertices in counterclockwise order, the interior lies to the left
            # of each edge. Thus, an inward normal is given by the left-hand perpendicular.
            wall_normal = Vector2(-edge.y, edge.x)
            if wall_normal.length() != 0:
                wall_normal = wall_normal.normalize()
            else:
                wall_normal = Vector2(0, 0)
            # Ensure the normal points toward the inside of the hexagon.
            if (hex_center - A).dot(wall_normal) < 0:
                wall_normal = -wall_normal

            # --- Reposition the ball out of the wall ---
            ball['pos'] += wall_normal * penetration

            # --- Compute the wall’s velocity at the collision point ---
            # (The wall is moving because the hexagon is rotating.)
            r = closest_point - hex_center
            # For a point at vector r, a counterclockwise rotation gives a velocity:
            wall_vel = Vector2(-r.y, r.x) * hex_angular_velocity

            # --- Compute the ball’s velocity relative to the moving wall ---
            rel_vel = ball['vel'] - wall_vel
            # The component of the relative velocity along the wall’s normal:
            vel_normal = rel_vel.dot(wall_normal)

            # Only reflect if the ball is moving into the wall.
            if vel_normal < 0:
                # Decompose the relative velocity into normal and tangential parts.
                vel_normal_vec = wall_normal * vel_normal
                vel_tangent_vec = rel_vel - vel_normal_vec

                # Reflect the normal component (with energy loss)
                # and reduce the tangential component to simulate friction.
                new_rel_vel = vel_tangent_vec * (1 - wall_friction) - restitution * vel_normal_vec

                # Convert back to the absolute velocity.
                ball['vel'] = new_rel_vel + wall_vel

    return collided


def main():
    global hex_angle

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bouncing Ball in a Spinning Hexagon")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Handle Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Time Step ---
        dt = clock.tick(FPS) / 1000.0  # seconds elapsed since last frame

        # --- Update Ball Physics ---
        # Gravity: accelerate downward.
        ball['vel'].y += GRAVITY * dt
        # Apply a little air friction (damping).
        ball['vel'] *= (1 - AIR_FRICTION_COEFF * dt)
        # Update the ball's position.
        ball['pos'] += ball['vel'] * dt

        # --- Update Hexagon Rotation ---
        hex_angle += ANGULAR_VELOCITY * dt
        hex_vertices = get_hexagon_vertices(HEXAGON_CENTER, HEXAGON_RADIUS, hex_angle)

        # --- Collision Detection & Resolution ---
        # To help avoid the ball “sticking” to a wall when overlapping, we iterate a few times.
        for _ in range(5):
            if not handle_collision(ball, hex_vertices, HEXAGON_CENTER, ANGULAR_VELOCITY, RESTITUTION, WALL_FRICTION):
                break

        # --- Drawing ---
        screen.fill((30, 30, 30))  # dark background

        # Draw the hexagon (as an outline)
        points = [(v.x, v.y) for v in hex_vertices]
        pygame.draw.polygon(screen, (200, 200, 200), points, 3)

        # Draw the ball
        pygame.draw.circle(screen, (255, 0, 0), (int(ball['pos'].x), int(ball['pos'].y)), ball['radius'])

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
