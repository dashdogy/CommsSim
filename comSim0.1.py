import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
AU = 1.496e11  # Distance from Earth to Sun in meters
EARTH_ORBIT_RADIUS = AU
MARS_ORBIT_RADIUS = 1.524 * AU
EARTH_PERIOD = 365.25 * 24 * 3600  # Earth's orbital period in seconds
MARS_PERIOD = 687 * 24 * 3600  # Mars' orbital period in seconds
EARTH_SATELLITE_ALTITUDE = 500e3
MARS_SATELLITE_ALTITUDE = 500e3
EARTH_RADIUS = 6371e3
MARS_RADIUS = 3389.5e3
EARTH_SATELLITE_RADIUS = EARTH_RADIUS + EARTH_SATELLITE_ALTITUDE
MARS_SATELLITE_RADIUS = MARS_RADIUS + MARS_SATELLITE_ALTITUDE
SUN_RADIUS = 6.957e8  # Radius of the Sun in meters

# Initial focus state and simulation speed
focus = None  # None means centered on the Sun
simulation_speed = 1  # Simulation speed multiplier (1x by default)

def satellite_position(orbit_radius, orbital_period, time):
    """Calculate position in a circular orbit."""
    angle = (2 * np.pi * time / orbital_period) % (2 * np.pi)
    x = orbit_radius * np.cos(angle)
    y = orbit_radius * np.sin(angle)
    return np.array([x, y])

def is_obstructed_by_sun(start, end, sun_radius):
    """Check if the communication line is obstructed by the Sun."""
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_dir = line_vec / line_len if line_len != 0 else np.zeros(2)
    
    # Project Sun's position (0, 0) onto the line
    t = -np.dot(start, line_dir) / line_len
    t = max(0, min(1, t))  # Clamp t to the segment
    closest_point = start + t * line_vec
    
    # Distance from Sun's center to the closest point
    dist_to_sun = np.linalg.norm(closest_point)
    
    return dist_to_sun < sun_radius

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# Set initial view limits
initial_zoom = 2 * AU
ax.set_xlim(-initial_zoom, initial_zoom)
ax.set_ylim(-initial_zoom, initial_zoom)

# Zoom control
zoom_factor = 1.1  # Scroll scaling factor

# Static plot elements
theta = np.linspace(0, 2 * np.pi, 1000)

# Planets and satellites
earth, = ax.plot([], [], 'bo', markersize=6, label='Earth')
mars, = ax.plot([], [], 'ro', markersize=6, label='Mars')
earth_satellite, = ax.plot([], [], 'go', markersize=4, label='Earth Satellite')
mars_satellite, = ax.plot([], [], 'mo', markersize=4, label='Mars Satellite')

# Orbital paths
earth_orbit, = ax.plot([], [], 'b--', label='Earth Orbit')
mars_orbit, = ax.plot([], [], 'r--', label='Mars Orbit')
earth_satellite_orbit, = ax.plot([], [], 'g--', label='Earth Satellite Orbit')
mars_satellite_orbit, = ax.plot([], [], 'm--', label='Mars Satellite Orbit')

# Communication line
communication_line, = ax.plot([], [], '-', lw=1.5, label='Communication Link')

# Speed indicator
speed_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black', ha='left')

# Initialization function
def init():
    earth.set_data([], [])
    mars.set_data([], [])
    earth_satellite.set_data([], [])
    mars_satellite.set_data([], [])
    earth_orbit.set_data([], [])
    mars_orbit.set_data([], [])
    communication_line.set_data([], [])
    speed_text.set_text(f"Simulation Speed: {simulation_speed}x")
    return earth, mars, earth_satellite, mars_satellite, earth_orbit, mars_orbit, communication_line, speed_text

# Update function for animation
def update(frame):
    global focus, simulation_speed
    time = frame * 3600 * simulation_speed  # Increment by simulation speed factor
    
    # Planet positions
    earth_pos = satellite_position(EARTH_ORBIT_RADIUS, EARTH_PERIOD, time)
    mars_pos = satellite_position(MARS_ORBIT_RADIUS, MARS_PERIOD, time)
    earth.set_data([earth_pos[0]], [earth_pos[1]])
    mars.set_data([mars_pos[0]], [mars_pos[1]])
    
    # Satellite positions
    earth_sat_pos = satellite_position(EARTH_SATELLITE_RADIUS, EARTH_PERIOD / 24, time)
    mars_sat_pos = satellite_position(MARS_SATELLITE_RADIUS, MARS_PERIOD / 24, time)
    earth_satellite.set_data([earth_sat_pos[0] + earth_pos[0]], [earth_sat_pos[1] + earth_pos[1]])
    mars_satellite.set_data([mars_sat_pos[0] + mars_pos[0]], [mars_sat_pos[1] + mars_pos[1]])
    
    # Orbital paths
    earth_orbit.set_data(EARTH_ORBIT_RADIUS * np.cos(theta), EARTH_ORBIT_RADIUS * np.sin(theta))
    mars_orbit.set_data(MARS_ORBIT_RADIUS * np.cos(theta), MARS_ORBIT_RADIUS * np.sin(theta))
    earth_satellite_orbit.set_data(EARTH_SATELLITE_RADIUS * np.cos(theta) + earth_pos[0], EARTH_SATELLITE_RADIUS * np.sin(theta) + earth_pos[1])
    mars_satellite_orbit.set_data(MARS_SATELLITE_RADIUS * np.cos(theta) + mars_pos[0], MARS_SATELLITE_RADIUS * np.sin(theta) + mars_pos[1])
    
    # Communication line
    start = earth_sat_pos + earth_pos
    end = mars_sat_pos + mars_pos
    communication_line.set_data([start[0], end[0]], [start[1], end[1]])
    
    if is_obstructed_by_sun(start, end, SUN_RADIUS):
        communication_line.set_color('red')
    else:
        communication_line.set_color('yellow')
    
    # Adjust view based on focus
    if focus == 'Earth':
        ax.set_xlim(earth_pos[0] - initial_zoom / 4, earth_pos[0] + initial_zoom / 4)
        ax.set_ylim(earth_pos[1] - initial_zoom / 4, earth_pos[1] + initial_zoom / 4)
    elif focus == 'Mars':
        ax.set_xlim(mars_pos[0] - initial_zoom / 4, mars_pos[0] + initial_zoom / 4)
        ax.set_ylim(mars_pos[1] - initial_zoom / 4, mars_pos[1] + initial_zoom / 4)
    elif focus == 'EarthSat1':
        ax.set_xlim(earth_sat_pos[0] + earth_pos[0] - initial_zoom / 10000, earth_sat_pos[0] + earth_pos[0] + initial_zoom / 10000)
        ax.set_ylim(earth_sat_pos[1] + earth_pos[1] - initial_zoom / 10000, earth_sat_pos[1] + earth_pos[1] + initial_zoom / 10000)
    elif focus == 'MarsSat1':
        ax.set_xlim(mars_sat_pos[0] + mars_pos[0] - initial_zoom / 10000, mars_sat_pos[0] + mars_pos[0] + initial_zoom / 10000)
        ax.set_ylim(mars_sat_pos[1] + mars_pos[1] - initial_zoom / 10000, mars_sat_pos[1] + mars_pos[1] + initial_zoom / 10000)
    elif focus is None:  # Center on Sun
        ax.set_xlim(-initial_zoom, initial_zoom)
        ax.set_ylim(-initial_zoom, initial_zoom)
    
    # Update speed indicator
    speed_text.set_text(f"Simulation Speed: {simulation_speed}x")
    fig.canvas.draw()
    
    return earth, mars, earth_satellite, mars_satellite, earth_orbit, mars_orbit, communication_line, speed_text


# Scroll event handler
def on_scroll(event):
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    x_center = (current_xlim[0] + current_xlim[1]) / 2
    y_center = (current_ylim[0] + current_ylim[1]) / 2
    width = (current_xlim[1] - current_xlim[0]) / 2
    height = (current_ylim[1] - current_ylim[0]) / 2

    if event.button == 'up':  # Zoom in
        width /= zoom_factor
        height /= zoom_factor
    elif event.button == 'down':  # Zoom out
        width *= zoom_factor
        height *= zoom_factor

    ax.set_xlim(x_center - width, x_center + width)
    ax.set_ylim(y_center - height, y_center + height)
    fig.canvas.draw_idle()

# Keyboard event handler
def on_key(event):
    global focus, simulation_speed
    if event.key == 'e':
        focus = 'Earth'  # Focus on Earth
    elif event.key == 'm':
        focus = 'Mars'  # Focus on Mars
    elif event.key == 'g':
        focus = None  # Center on Sun
    elif event.key == '1':
        focus = 'EarthSat1'  # Focus on Earth Satellite
    elif event.key == '2':
        focus = 'MarsSat1' # Focus on Mars Satellite
    elif event.key == 'up':
        simulation_speed = min(simulation_speed * 2, 128)  # Cap speed at 128x
    elif event.key == 'down':
        simulation_speed = max(simulation_speed / 2, 0.0000001)  # Min speed at 0.125x

# Connect the scroll and keyboard events to the handlers
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)

# Animation
frames = int(EARTH_PERIOD / 3600)  # One frame per hour for one Earth year
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

# Display the plot
plt.legend()
plt.title("Satellite Communication Network between Earth and Mars")
plt.xlabel("X Distance (meters)")
plt.ylabel("Y Distance (meters)")
plt.grid(True)
plt.show()
