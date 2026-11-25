import numpy as np

# Parametry chmury punktów
n = 5000
x_min, x_max = 0, 100
y_min, y_max = 0, 100
z_min, z_max = 0, 100

r = 50                  # Promień koła
x_center = 50           # Środek koła - X
y_center = 50           # Środek koła - Y

# A - powierzchnia płaska pozioma

# Generowanie losowych współrzędnych
x = np.random.uniform(x_min, x_max, n)
y = np.random.uniform(y_min, y_max, n)
z = np.full(n, 0.0)

points = np.column_stack((x, y, z))     # Połączenie współrzędnych XYZ
np.savetxt(                             # Zapis do pliku A XYZ
    r"C:\Users\kacpe\Documents\Obliczenia Inteligentne\chmura_ABC.xyz",
    points,
    fmt="%.6f",
    delimiter=" "
)

# B - powierzchnia płaska pionowa

# Generowanie losowych współrzędnych
x = np.full(n, 0.0)
y = np.random.uniform(y_min+120, y_max+120, n)
z = np.random.uniform(z_min, z_max, n)

points = np.column_stack((x, y, z))     # Połączenie współrzędnych XYZ
with open(r"C:\Users\kacpe\Documents\Obliczenia Inteligentne\chmura_ABC.xyz", "a") as f:
    np.savetxt(
        f,
        points,
        fmt="%.6f",
        delimiter=" "
    )

# C - powierzchnia cylindryczna

# Losowanie współrzędnych biegunowych
theta = np.random.uniform(0, 2 * np.pi, n)
# Przeliczenie na współrzędne kartezjańskie
x = x_center + r * np.cos(theta)
y = y_center + r * np.sin(theta)
z = np.random.uniform(z_min+50, z_max+50, n)

points = np.column_stack((x, y, z))     # Połączenie współrzędnych XYZ
with open(r"C:\Users\kacpe\Documents\Obliczenia Inteligentne\chmura_ABC.xyz", "a") as f:
    np.savetxt(
        f,
        points,
        fmt="%.6f",
        delimiter=" "
    )

print(f"Wygenerowano 3 powierzchnie.")

