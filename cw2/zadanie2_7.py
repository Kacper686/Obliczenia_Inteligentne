
import numpy as np


# --------------------------------------------------
# 1. Wczytywanie chmury punktow XYZRGB
# --------------------------------------------------

def load_cloud(filename):

    points = []

    with open(filename, "r", encoding="utf-8") as file:

        for line in file:

            values = line.split()

            if len(values) < 6:
                continue

            try:
                points.append([float(v) for v in values[:6]])
            except ValueError:
                continue

    if not points:
        raise ValueError("Plik nie zawiera poprawnych punktow XYZRGB.")

    return np.array(points, dtype=float)


# --------------------------------------------------
# 2. Dopasowanie plaszczyzny metoda RANSAC
# --------------------------------------------------

def ransac_plane(points, iterations=1000,
                 threshold=0.01, min_inliers=100,
                 random_state=None):

    """
    Dopasowuje plaszczyzne do chmury punktow XYZ.

    Zwraca:
        coefficients - wspolczynniki plaszczyzny [a,b,c,d]
        inliers      - maska punktow nalezacych do plaszczyzny
    """

    xyz = points[:, :3]

    if len(xyz) < 3:
        raise ValueError("Za malo punktow do dopasowania plaszczyzny.")

    rng = np.random.default_rng(random_state)

    best_inliers = None
    best_count = 0

    for _ in range(iterations):

        # Losowanie trzech roznych punktow
        sample = xyz[rng.choice(len(xyz), 3, replace=False)]

        p1, p2, p3 = sample

        # Wyznaczenie wektora normalnego
        normal = np.cross(p2 - p1, p3 - p1)

        norm = np.linalg.norm(normal)

        # Odrzucenie punktow wspolliniowych
        if norm < 1e-10:
            continue

        normal /= norm

        # Rownanie plaszczyzny
        d = -np.dot(normal, p1)

        # Obliczenie odleglosci punktow od plaszczyzny
        distances = np.abs(xyz @ normal + d)

        inliers = distances < threshold

        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        raise ValueError("Nie znaleziono odpowiedniej plaszczyzny.")

    # Ponowne dopasowanie do znalezionych inliers
    selected = xyz[best_inliers]

    centroid = np.mean(selected, axis=0)

    _, _, vh = np.linalg.svd(
        selected - centroid,
        full_matrices=False
    )

    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    d = -np.dot(normal, centroid)

    # Ponowne wyznaczenie inliers po dopasowaniu
    distances = np.abs(xyz @ normal + d)
    inliers = distances < threshold

    if np.sum(inliers) < min_inliers:
        raise ValueError("Za malo punktow nalezy do dopasowanej plaszczyzny.")

    coefficients = np.append(normal, d)

    return coefficients, inliers


# --------------------------------------------------
# 3. Iteracyjne dopasowanie szesciu plaszczyzn
# --------------------------------------------------

def extract_planes(points, K=6, iterations=1000,
                   threshold=0.01, min_inliers=100):

    remaining = points.copy()

    extracted_planes = []

    for i in range(K):

        print(f"\n{'=' * 45}")
        print(f"ITERACJA {i + 1} / {K}")
        print(f"{'=' * 45}")

        print("Liczba punktow przed dopasowaniem:",
              len(remaining))

        if len(remaining) < min_inliers:
            print("Za malo pozostalych punktow. Koniec ekstrakcji.")
            break

        try:

            # Dopasowanie plaszczyzny do aktualnej chmury
            coefficients, inliers = ransac_plane(
                remaining,
                iterations=iterations,
                threshold=threshold,
                min_inliers=min_inliers,
                random_state=42 + i
            )

            # Wyodrebnienie punktow nalezacych do plaszczyzny
            plane_points = remaining[inliers]

            # Zapis wyekstrahowanej plaszczyzny
            extracted_planes.append(plane_points)

            a, b, c, d = coefficients

            print("\nZnaleziono plaszczyzne.")
            print(f"Liczba punktow plaszczyzny: {len(plane_points)}")

            print("\nWektor normalny:")
            print(f"nx = {a:.5f}")
            print(f"ny = {b:.5f}")
            print(f"nz = {c:.5f}")

            print("\nRownanie plaszczyzny:")
            print(f"{a:.5f}x + {b:.5f}y + "
                  f"{c:.5f}z + {d:.5f} = 0")

            # Usuniecie punktow nalezacych do znalezionej plaszczyzny
            remaining = remaining[~inliers]

            print("\nUsunieto punktow:", len(plane_points))
            print("Liczba pozostalych punktow:", len(remaining))

        except ValueError as error:

            print("Nie znaleziono kolejnej plaszczyzny:", error)
            break

    return extracted_planes, remaining


# --------------------------------------------------
# 4. Eksport wszystkich plaszczyzn do jednego pliku
# --------------------------------------------------

def save_planes(planes, filename):

    if not planes:
        print("Brak plaszczyzn do zapisania.")
        return

    # Polaczenie wszystkich wyekstrahowanych plaszczyzn
    all_planes = np.vstack(planes)

    # Zapis XYZRGB
    np.savetxt(
        filename,
        all_planes,
        fmt="%.6f %.6f %.6f %.0f %.0f %.0f"
    )

    print("\nEksport zakonczony.")
    print("Liczba zapisanych punktow:", len(all_planes))
    print("Plik wynikowy:", filename)


# --------------------------------------------------
# 5. Glowny program
# --------------------------------------------------

input_file = r"C:\Users\kacpe\Documents\Obliczenia Inteligentne\cw2\conferenceRoom_1.txt"
output_file = r"C:\Users\kacpe\Documents\Obliczenia Inteligentne\cw2\conferenceRoom_1_planes.xyz"

# Wczytanie chmury punktow
points = load_cloud(input_file)

print("Wczytano chmure punktow.")
print("Liczba punktow:", len(points))

# Iteracyjne wyodrebnianie szesciu plaszczyzn
planes, remaining = extract_planes(
    points,
    K=6,
    iterations=1000,
    threshold=0.01,
    min_inliers=100
)

# Eksport wszystkich znalezionych plaszczyzn
save_planes(planes, output_file)

print("\nLiczba znalezionych plaszczyzn:", len(planes))
print("Liczba punktow pozostalych:", len(remaining))
