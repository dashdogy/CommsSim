import numpy as np
from scipy.optimize import fsolve
from datetime import datetime

# Orbital parameters for Earth and Mars
orbital_params = {
    'earth': {
        'a': 1.000,     # AU (semi-major axis)
        'e': 0.0167,    # Eccentricity
        'T': 365.25,    # Orbital period in days
        'M0': 357.5291, # Mean anomaly at J2000 in degrees
    },
    'mars': {
        'a': 1.5237,    # AU (semi-major axis)
        'e': 0.0934,    # Eccentricity
        'T': 687,       # Orbital period in days
        'M0': 19.4123,  # Mean anomaly at J2000 in degrees
    }
}

# Function to calculate mean anomaly at a given time (in days from J2000)
def mean_anomaly(M0, T, time):
    n = 2 * np.pi / T  # Mean motion (radians per day)
    M = M0 + n * time  # Mean anomaly in degrees
    return M % 360  # Ensure it's between 0 and 360 degrees

# Kepler's equation solver
def kepler_equation(E, M, e):
    return E - e * np.sin(np.radians(E)) - np.radians(M)

# Function to calculate true anomaly
def true_anomaly(M, e):
    # Solve Kepler's equation for Eccentric anomaly (E)
    E_guess = M  # Initial guess for E
    E_solution = fsolve(kepler_equation, E_guess, args=(M, e))[0]  # Solved Eccentric anomaly in radians
    # Calculate True anomaly
    tan_v2 = np.sqrt(1 + e) * np.tan(E_solution / 2)
    v = 2 * np.arctan(tan_v2)  # True anomaly in radians
    return np.degrees(v)  # Convert back to degrees

# Function to calculate the number of days since J2000
def days_since_j2000(day, month, year):
    # J2000 epoch is 1 January 2000
    j2000 = datetime(2000, 1, 1)
    current_date = datetime(year, month, day)
    delta = current_date - j2000
    return delta.days

# Main function to calculate true anomalies for Earth and Mars on a given date
def calculate_true_anomalies(day, month, year):
    # Calculate the number of days since J2000
    time = days_since_j2000(day, month, year)

    # Calculate mean anomalies for Earth and Mars on the given date
    M_earth = mean_anomaly(orbital_params['earth']['M0'], orbital_params['earth']['T'], time)
    M_mars = mean_anomaly(orbital_params['mars']['M0'], orbital_params['mars']['T'], time)

    # Calculate true anomalies for Earth and Mars
    true_anomaly_earth = true_anomaly(M_earth, orbital_params['earth']['e'])
    true_anomaly_mars = true_anomaly(M_mars, orbital_params['mars']['e'])

    return true_anomaly_earth, true_anomaly_mars

# Example usage: Calculate true anomalies for 1 January 2032
day = 1
month = 1
year = 2032
true_anomaly_earth, true_anomaly_mars = calculate_true_anomalies(day, month, year)

print(f"True Anomaly of Earth on {day}-{month}-{year}: {true_anomaly_earth:.4f} degrees")
print(f"True Anomaly of Mars on {day}-{month}-{year}: {true_anomaly_mars:.4f} degrees")
