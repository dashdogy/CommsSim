import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
AU = 1.496e11  # Distance from Earth to Sun in meters
EARTH_SEMI_MAJOR_AXIS = AU
MARS_SEMI_MAJOR_AXIS = 1.524 * AU
EARTH_PERIOD = 365.25 * 24 * 3600  # Earth's orbital period in seconds
MARS_PERIOD = 687 * 24 * 3600  # Mars' orbital period in seconds
SUN_RADIUS = 6.957e8  # Radius of the Sun in meters

EARTH_ECCENTRICITY = 0.0167
MARS_ECCENTRICITY = 0.0934
EARTH_SATELLITE_ALTITUDE = 500e3
MARS_SATELLITE_ALTITUDE = 500e3
EARTH_RADIUS = 6371e3
MARS_RADIUS = 3389.5e3
EARTH_SATELLITE_RADIUS = EARTH_RADIUS + EARTH_SATELLITE_ALTITUDE
MARS_SATELLITE_RADIUS = MARS_RADIUS + MARS_SATELLITE_ALTITUDE

# Keplerian orbit calculation
def kepler_orbit_points(semi_major_axis, eccentricity, num_points=1000):
    true_anomalies = np.linspace(0, 2 * np.pi, num_points)
    radii = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomalies))
    x = radii * np.cos(true_anomalies)
    y = radii * np.sin(true_anomalies)
    return x, y

earth_orbit_x, earth_orbit_y = kepler_orbit_points(EARTH_SEMI_MAJOR_AXIS, EARTH_ECCENTRICITY)
mars_orbit_x, mars_orbit_y = kepler_orbit_points(MARS_SEMI_MAJOR_AXIS, MARS_ECCENTRICITY)

def kepler_position(semi_major_axis, eccentricity, orbital_period, time):
    mean_motion = 2 * np.pi / orbital_period
    mean_anomaly = mean_motion * time
    eccentric_anomaly = mean_anomaly

    # Solve Kepler's Equation iteratively
    for _ in range(100):
        eccentric_anomaly = mean_anomaly + eccentricity * np.sin(eccentric_anomaly)

    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
        np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2)
    )
    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
    x = r * np.cos(true_anomaly)
    y = r * np.sin(true_anomaly)
    return np.array([x, y])

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# Initial zoom and simulation speed
initial_zoom = 2 * AU
simulation_speed = 1
focus = None

# Set plot limits
ax.set_xlim(-initial_zoom, initial_zoom)
ax.set_ylim(-initial_zoom, initial_zoom)

# Orbital paths
ax.plot(earth_orbit_x, earth_orbit_y, 'b--', label='Earth Orbit')
ax.plot(mars_orbit_x, mars_orbit_y, 'r--', label='Mars Orbit')

# Planets and satellites
earth, = ax.plot([], [], 'bo', markersize=6, label='Earth')
mars, = ax.plot([], [], 'ro', markersize=6, label='Mars')
sun, = ax.plot([0], [0], 'yo', markersize=12, label='Sun')
earth_satellite, = ax.plot([], [], 'go', markersize=4, label='Earth Satellite')
mars_satellite, = ax.plot([], [], 'mo', markersize=4, label='Mars Satellite')
communication_line, = ax.plot([], [], '-', lw=1.5, label='Communication Link', color='green')

# Speed indicator
speed_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black', ha='left')

# Initialization function
def init():
    earth.set_data([], [])
    mars.set_data([], [])
    earth_satellite.set_data([], [])
    mars_satellite.set_data([], [])
    communication_line.set_data([], [])
    speed_text.set_text(f"Simulation Speed: {simulation_speed}x")
    return earth, mars, earth_satellite, mars_satellite, communication_line, speed_text
# Initialize the MET text
met_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=10, color='black', ha='left')


def format_time(seconds):
    """Format seconds into Days:HH:MM:SS format."""
    years = int(seconds // 31536000)  # 31536000 seconds in a year
    days = int((seconds % 31536000) // 86400)  # 86400 seconds in a day
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f" {years}yr {days}d {hours:02}:{minutes:02}:{secs:02}"

# Variable to store the current simulation time bad fix but if it works it works
current_time = 0  # Start at 0 seconds

# Function to update the simulation
def update(frame):
    global focus, simulation_speed, current_time
    current_time += 3600 * simulation_speed  # Increment time based on speed (in seconds)

    # Keplerian positions for Earth and Mars based on updated time
    earth_pos = kepler_position(EARTH_SEMI_MAJOR_AXIS, EARTH_ECCENTRICITY, EARTH_PERIOD, current_time)
    mars_pos = kepler_position(MARS_SEMI_MAJOR_AXIS, MARS_ECCENTRICITY, MARS_PERIOD, current_time)
    
    # Ensure positions are sequences
    earth.set_data([earth_pos[0]], [earth_pos[1]])
    mars.set_data([mars_pos[0]], [mars_pos[1]])

    # Satellite positions based on updated time
    earth_sat_angle = (2 * np.pi * current_time / (EARTH_PERIOD / 24)) % (2 * np.pi)
    mars_sat_angle = (2 * np.pi * current_time / (MARS_PERIOD / 24)) % (2 * np.pi)
    earth_sat_pos = earth_pos + np.array([EARTH_SATELLITE_RADIUS * np.cos(earth_sat_angle),
                                          EARTH_SATELLITE_RADIUS * np.sin(earth_sat_angle)])
    mars_sat_pos = mars_pos + np.array([MARS_SATELLITE_RADIUS * np.cos(mars_sat_angle),
                                        MARS_SATELLITE_RADIUS * np.sin(mars_sat_angle)])


    # Update satellite positions
    earth_satellite.set_data([earth_sat_pos[0]], [earth_sat_pos[1]])
    mars_satellite.set_data([mars_sat_pos[0]], [mars_sat_pos[1]])

    # Communication line between satellites
    communication_line.set_data([earth_sat_pos[0], mars_sat_pos[0]], 
                                 [earth_sat_pos[1], mars_sat_pos[1]])

    # Updated focus logic (no more if-elif-else)
    focus_positions = {
    'Earth': {'position': earth_pos, 'zoomRatio': 4},
    'Mars': {'position': mars_pos, 'zoomRatio': 4},
    'EarthSat1': {'position': earth_sat_pos, 'zoomRatio': 10000},
    'MarsSat1': {'position': mars_sat_pos, 'zoomRatio': 10000}
    
    }

    # Get the position for the given focus, defaulting to (0, 0) if not found
    focus_data = focus_positions.get(focus, {'position': (0, 0), 'zoomRatio': 4})
    focus_pos = focus_data['position']
    zoom = focus_data['zoomRatio']

    # Set limits based on the focus position
    ax.set_xlim(focus_pos[0] - initial_zoom / zoom, focus_pos[0] + initial_zoom / zoom)
    ax.set_ylim(focus_pos[1] - initial_zoom / zoom, focus_pos[1] + initial_zoom / zoom)

    # If no specific focus, adjust to the default zoom
    if focus not in focus_positions:
        ax.set_xlim(-initial_zoom, initial_zoom)
        ax.set_ylim(-initial_zoom, initial_zoom)

    
    # Update simulation speed and MET
    speed_text.set_text(f"Simulation Speed: {simulation_speed}x")
    met_text.set_text(f"Mission Elapsed Time: {format_time(current_time)}")
    fig.canvas.draw()
    
    return earth, mars, earth_satellite, mars_satellite, communication_line, speed_text, met_text

# Event handlers
def on_scroll(event):
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    x_center = (current_xlim[0] + current_xlim[1]) / 2
    y_center = (current_ylim[0] + current_ylim[1]) / 2
    width = (current_xlim[1] - current_xlim[0]) / 2
    height = (current_ylim[1] - current_ylim[0]) / 2

    if event.button == 'up':
        width /= 1.1
        height /= 1.1
    elif event.button == 'down':
        width *= 1.1
        height *= 1.1

    ax.set_xlim(x_center - width, x_center + width)
    ax.set_ylim(y_center - height, y_center + height)


def on_key(event):
    global focus, simulation_speed
    if event.key == 'e':
        focus = 'Earth'
    elif event.key == 'm':
        focus = 'Mars'
    elif event.key == 'g':
        focus = None
    elif event.key == '1':
        focus = 'EarthSat1'
    elif event.key == '2':
        focus = 'MarsSat1'
    elif event.key == 'up':
        simulation_speed = min(simulation_speed * 2, 128)
    elif event.key == 'down':
        simulation_speed = max(simulation_speed / 2, 1/128)

fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)

# Animation
frames = int(EARTH_PERIOD / 3600)
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)


plt.title("Comnet Simulation V0.1")
plt.grid(True)
plt.show()
