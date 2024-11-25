import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from astroquery.jplhorizons import Horizons
from datetime import datetime

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
BASE_EARTH_MARKER_SIZE = 6  # Default 4
BASE_MARS_MARKER_SIZE = 6  


# Keplerian orbit calculation
def kepler_orbit_points(semi_major_axis, eccentricity, num_points=1000):
    true_anomalies = np.linspace(0, 2 * np.pi, num_points)
    radii = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomalies))
    x = radii * np.cos(true_anomalies)
    y = radii * np.sin(true_anomalies)
    return x, y


#Orbital Path Calculations
earth_orbit_x, earth_orbit_y = kepler_orbit_points(EARTH_SEMI_MAJOR_AXIS, EARTH_ECCENTRICITY)
mars_orbit_x, mars_orbit_y = kepler_orbit_points(MARS_SEMI_MAJOR_AXIS, MARS_ECCENTRICITY)

# Satellite orbital paths (circular orbits)
theta = np.linspace(0, 2 * np.pi, 1000)
earth_sat_orbit_x = EARTH_SATELLITE_RADIUS * np.cos(theta)
earth_sat_orbit_y = EARTH_SATELLITE_RADIUS * np.sin(theta)
mars_sat_orbit_x = MARS_SATELLITE_RADIUS * np.cos(theta)
mars_sat_orbit_y = MARS_SATELLITE_RADIUS * np.sin(theta)


def kepler_position(semi_major_axis, eccentricity, orbital_period, time, start_anomaly):
    mean_motion = 2 * np.pi / orbital_period
    mean_anomaly = mean_motion * time + start_anomaly  # Add the initial true anomaly (start_anomaly)
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

# Satellite orbital paths
# Earth and Mars orbits are centered at their host planets
earth_sat_path, = ax.plot([], [], 'g--', label='Earth Satellite Orbit', alpha=0.5)
mars_sat_path, = ax.plot([], [], 'm--', label='Mars Satellite Orbit', alpha=0.5)



# Earth's radius circle (outer boundary of Earth)
earth_radius_circle = Circle((0, 0), EARTH_RADIUS, color='blue', alpha=0.3, label="Earth's Outer Radius")
ax.add_artist(earth_radius_circle)

# Mars's radius circle (outer boundary of Mars)
mars_radius_circle = Circle((0, 0), MARS_RADIUS, color='red', alpha=0.3, label="Mars's Outer Radius")
ax.add_artist(mars_radius_circle)

# Planets and satellites
earth, = ax.plot([], [], 'bo', markersize=6, label='Earth')
mars, = ax.plot([], [], 'ro', markersize=6, label='Mars')
sun, = ax.plot([0], [0], 'yo', markersize=12, label='Sun')
earth_satellite, = ax.plot([], [], 'go', markersize=4, label='Earth Satellite')
mars_satellite, = ax.plot([], [], 'mo', markersize=4, label='Mars Satellite')
communication_line, = ax.plot([], [], '-', lw=1.5, label='Communication Link', color='green', alpha=1)

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

def line_intersects_circle(p1, p2, circle_center, circle_radius):
    """
    Check if a line segment (p1 to p2) intersects a circle.
    
    Args:
        p1 (np.array): Start point of the line segment.
        p2 (np.array): End point of the line segment.
        circle_center (np.array): Center of the circle.
        circle_radius (float): Radius of the circle.

    Returns:
        bool: True if the line segment intersects the circle, False otherwise.
    """
    # Vector from p1 to p2
    line_vec = p2 - p1
    # Vector from p1 to the circle center
    p1_to_center = circle_center - p1
    # Project p1_to_center onto the line direction
    t = np.dot(p1_to_center, line_vec) / np.dot(line_vec, line_vec)
    # Clamp t to the line segment (0 ≤ t ≤ 1)
    t = max(0, min(1, t))
    # Closest point on the line segment to the circle center
    closest_point = p1 + t * line_vec
    # Distance from closest point to circle center
    distance_to_center = np.linalg.norm(closest_point - circle_center)
    # Check if the distance is less than the circle's radius
    return distance_to_center <= circle_radius


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

def orbital_period(semi_major_axis, mass_of_planet):
    """Calculate the orbital period using Kepler's Third Law."""
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    return 2 * np.pi * np.sqrt((semi_major_axis**3) / (G * mass_of_planet))

# Create an inset axes for the snapshot
inset_ax = fig.add_axes([0.7, 0.75, 0.25, 0.2])  # [left, bottom, width, height]
inset_ax.set_aspect('equal')
inset_ax.axis('off')  # Turn off the axes by default

starting_time_Earth = 0
starting_time_Mars = 0

#Render Loop
def update(frame):
    global focus, simulation_speed, current_time
    current_time += 3600 * simulation_speed  # Increment time based on speed (in seconds)

    # Keplerian positions for Earth and Mars based on updated time
    earth_pos = kepler_position(EARTH_SEMI_MAJOR_AXIS, EARTH_ECCENTRICITY, EARTH_PERIOD, current_time, starting_time_Earth)
    mars_pos = kepler_position(MARS_SEMI_MAJOR_AXIS, MARS_ECCENTRICITY, MARS_PERIOD, current_time, starting_time_Mars)
    
    # Ensure positions are sequences
    earth.set_data([earth_pos[0]], [earth_pos[1]])
    mars.set_data([mars_pos[0]], [mars_pos[1]])

    # Calculate the orbital periods of the satellites around Earth and Mars using Kepler's Third Law
    earth_sat_semi_major_axis = EARTH_SATELLITE_RADIUS  # Semi-major axis for Earth satellite
    mars_sat_semi_major_axis = MARS_SATELLITE_RADIUS  # Semi-major axis for Mars satellite
    
    # Orbital periods (in seconds)
    earth_sat_period = orbital_period(earth_sat_semi_major_axis, 5.972e24)  # Earth mass
    mars_sat_period = orbital_period(mars_sat_semi_major_axis, 0.64171e24)  # Mars mass

    # Compute angular positions for satellites (angular velocity = 2*pi / period)
    earth_sat_angle = (2 * np.pi * (current_time / earth_sat_period)) % (2 * np.pi)
    mars_sat_angle = (2 * np.pi * (current_time / mars_sat_period)) % (2 * np.pi)

    # Satellite positions based on updated time (earth_sat_angle and mars_sat_angle)
    earth_sat_pos = earth_pos + np.array([EARTH_SATELLITE_RADIUS * np.cos(earth_sat_angle),
                                          EARTH_SATELLITE_RADIUS * np.sin(earth_sat_angle)])
    mars_sat_pos = mars_pos + np.array([MARS_SATELLITE_RADIUS * np.cos(mars_sat_angle),
                                        MARS_SATELLITE_RADIUS * np.sin(mars_sat_angle)])

    # Update satellite orbital paths
    earth_sat_path.set_data(earth_pos[0] + earth_sat_orbit_x, earth_pos[1] + earth_sat_orbit_y)
    mars_sat_path.set_data(mars_pos[0] + mars_sat_orbit_x, mars_pos[1] + mars_sat_orbit_y)

    # Update the radius circles for Earth and Mars
    earth_radius_circle.center = (earth_pos[0], earth_pos[1])
    mars_radius_circle.center = (mars_pos[0], mars_pos[1])

    # Define celestial bodies for line-of-sight obstruction
    celestial_bodies = [
    {'name': 'Sun', 'position': np.array([0, 0]), 'radius': SUN_RADIUS},
    {'name': 'Earth', 'position': earth_pos, 'radius': EARTH_RADIUS},
    {'name': 'Mars', 'position': mars_pos, 'radius': MARS_RADIUS},
]
    # Default communication line color
    communication_line_color = 'green'

    # Check for obstructions
    for body in celestial_bodies:
        if line_intersects_circle(earth_sat_pos, mars_sat_pos, body['position'], body['radius']):
            communication_line_color = 'red'  # Communication obstructed
            break  # No need to check further

    # Update the communication line color and data
    if communication_line_color == 'green':
        communication_line.set_data([earth_sat_pos[0], mars_sat_pos[0]],
                                    [earth_sat_pos[1], mars_sat_pos[1]])
    else:
        communication_line.set_data([], [])  # Clear the line if obstructed
    communication_line.set_color(communication_line_color)
    # Update satellite positions
    earth_satellite.set_data([earth_sat_pos[0]], [earth_sat_pos[1]])
    mars_satellite.set_data([mars_sat_pos[0]], [mars_sat_pos[1]])

    communication_line.set_color(communication_line_color)


    # Communication line between satellites
    communication_line.set_data([earth_sat_pos[0], mars_sat_pos[0]], 
                                 [earth_sat_pos[1], mars_sat_pos[1]])

    # Updated focus logic (no more if-elif-else)
    # focus_positions = {
    #     'Earth': {'position': earth_pos, 'zoomRatio': 4, 'radius': EARTH_RADIUS, 'label': "Earth"},
    #     'Mars': {'position': mars_pos, 'zoomRatio': 4, 'radius': MARS_RADIUS, 'label': "Mars"},
    #     'EarthSat1': {'position': earth_sat_pos, 'zoomRatio': 10000, 'partner': mars_sat_pos,
    #                   'radius': EARTH_RADIUS, 'targetCelestial': mars_pos, 'planet_label': "Earth"},
    #     'MarsSat1': {'position': mars_sat_pos, 'zoomRatio': 10000, 'partner': earth_sat_pos,
    #                  'radius': MARS_RADIUS, 'targetCelestial': earth_pos, 'planet_label': "Mars"}
    # }

    focus_positions = {
            'Earth': {'position': earth_pos, 'zoomRatio': 4, 'radius': EARTH_RADIUS, 'label': "Earth", 'color': 'blue'}, 
            'Mars': {'position': mars_pos, 'zoomRatio': 4, 'radius': MARS_RADIUS, 'label': "Mars", 'color': 'red'},
            'EarthSat1': {'position': earth_sat_pos, 'zoomRatio': 10000, 'partner': 'MarsSat1', 'targetCelestial': 'Mars', 'altitude': EARTH_SATELLITE_ALTITUDE},
            'MarsSat1': {'position': mars_sat_pos, 'zoomRatio': 10000, 'partner': 'EarthSat1', 'targetCelestial': 'Earth', 'altitude': MARS_SATELLITE_ALTITUDE}
        }
    
    # Get the position for the given focus, defaulting to (0, 0) if not found
    focus_data = focus_positions.get(focus, {'position': (0, 0), 'zoomRatio': 4, 
                                             'markerSize': 4, 'celestialMarkerSize': 4})
    focus_pos = focus_data['position']
    zoom = focus_data['zoomRatio']

    # Set limits based on the focus position
    ax.set_xlim(focus_pos[0] - initial_zoom / zoom, focus_pos[0] + initial_zoom / zoom)
    ax.set_ylim(focus_pos[1] - initial_zoom / zoom, focus_pos[1] + initial_zoom / zoom)


    
    # If no specific focus, adjust to the default zoom
    if focus not in focus_positions:
        ax.set_xlim(-initial_zoom, initial_zoom)
        ax.set_ylim(-initial_zoom, initial_zoom)


    if  'targetCelestial' in focus_data and 'partner' in focus_data:
        # Update the inset snapshot

        inset_ax.clear()
        inset_ax.set_aspect('equal')
        inset_ax.set_title(f"View near {focus_data['targetCelestial']}", fontsize=8)

        # Draw the focused planet
        target = focus_data['targetCelestial']
        target_pos = focus_positions[target]['position']
        target_radius = focus_positions[target]['radius']
        target_label = focus_positions[target]['label']
        target_color = focus_positions[target]['color']
        planet_circle = plt.Circle(target_pos, target_radius, color=target_color, alpha=0.3, label=target_label)
        inset_ax.add_artist(planet_circle)

        # Draw the planet's orbital path
        orbital_path = plt.Circle([0, 0], np.linalg.norm(target_pos), color='gray', linestyle='--', fill=False, alpha=0.5)
        inset_ax.add_artist(orbital_path)
            

        # Draw the partner satellite
        partnerSat = focus_data['partner']
        partner_pos = focus_positions[partnerSat]['position']
        partner_altitude = focus_positions[partnerSat]['altitude']
        inset_ax.plot(partner_pos[0], partner_pos[1], 'mo', markersize=6, label='Partner Satellite')

        # Draw communication line
        line_color = 'green' if not line_intersects_circle(
            focus_data['position'], partner_pos, target_pos, target_radius) else 'red'
        inset_ax.plot([focus_data['position'][0], partner_pos[0]],
                        [focus_data['position'][1], partner_pos[1]], color=line_color, linestyle='-', linewidth=1.5)
        # Draw satellite orbit
        satellite_orbit = plt.Circle(target_pos, target_radius + partner_altitude, color='green', linestyle='--', fill=False, alpha=0.5)
        inset_ax.add_artist(satellite_orbit)

        # Draw the focused satellite
        inset_ax.set_xlim(partner_pos[0] - initial_zoom / zoom, partner_pos[0] + initial_zoom / zoom)
        inset_ax.set_ylim(partner_pos[1] - initial_zoom / zoom, partner_pos[1] + initial_zoom / zoom)

    else:
    
        inset_ax.axis('off')
        
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


plt.title("Comnet Simulation V0.2")
plt.grid(True)
plt.show()
