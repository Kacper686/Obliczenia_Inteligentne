
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# --------------------------------------------------
# 1. Wczytywanie chmury punktow XYZ
# --------------------------------------------------

def load_xyz(filename):

    points = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()

            if len(values) < 3:
                continue

            try:
                points.append([
                    float(values[0]),
                    float(values[1]),
                    float(values[2])
                ])
            except ValueError:
                continue

    if not points:
        raise ValueError("Plik nie zawiera poprawnych punktow XYZ.")

    return np.array(points)


# --------------------------------------------------
# 2. Podzial chmury punktow za pomoca K-means
# --------------------------------------------------

def cluster_points(points, k=3):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(points)

    clusters = [points[labels == i] for i in range(k)]

    for i, cluster in enumerate(clusters):
        print(f"Klaster {i}: {len(cluster)} punktow")

    return clusters, labels


# --------------------------------------------------
# 3. Dopasowanie plaszczyzny za pomoca RANSAC
# --------------------------------------------------

def ransac_plane(points, iterations=1000, threshold=0.01, min_inliers=10):

    rng = np.random.default_rng(42)

    best_inliers = None
    best_count = 0

    for _ in range(iterations):

        # Losowanie trzech punktow
        sample = points[rng.choice(len(points), 3, replace=False)]

        p1, p2, p3 = sample

        # Wyznaczenie wektora normalnego
        normal = np.cross(p2 - p1, p3 - p1)

        norm = np.linalg.norm(normal)

        if norm < 1e-10:
            continue

        normal /= norm

        # Rownanie plaszczyzny
        d = -np.dot(normal, p1)

        # Odleglosci punktow od plaszczyzny
        distances = np.abs(points @ normal + d)

        inliers = distances < threshold

        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        raise ValueError("Nie znaleziono odpowiedniej plaszczyzny.")

    # Ponowne dopasowanie do wszystkich inliers
    selected = points[best_inliers]

    centroid = np.mean(selected, axis=0)

    _, _, vh = np.linalg.svd(
        selected - centroid,
        full_matrices=False
    )

    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    d = -np.dot(normal, centroid)

    return np.append(normal, d), best_inliers


# --------------------------------------------------
# 4. Klasyfikacja powierzchni
# --------------------------------------------------

def classify_surface(points, coefficients, inliers,
                     min_inlier_ratio=0.90,
                     max_mean_distance=0.01):

    normal = coefficients[:3]
    d = coefficients[3]

    # Srednia odleglosc wszystkich punktow od plaszczyzny
    distances = np.abs(points @ normal + d)

    mean_distance = np.mean(distances)

    # Procent punktow nalezacych do plaszczyzny
    inlier_ratio = np.sum(inliers) / len(points)

    print(f"Liczba inliers: {np.sum(inliers)}")
    print(f"Liczba outliers: {len(points) - np.sum(inliers)}")
    print(f"Procent inliers: {inlier_ratio * 100:.2f}%")
    print(f"Srednia odleglosc: {mean_distance:.6f}")

    # Sprawdzenie, czy powierzchnia jest plaska
    if (inlier_ratio < min_inlier_ratio or
            mean_distance > max_mean_distance):

        return False, None

    # Okreslenie orientacji plaszczyzny
    if abs(normal[2]) >= 0.95:
        orientation = "POZIOMA"

    elif abs(normal[2]) <= 0.10:
        orientation = "PIONOWA"

    else:
        orientation = "UKOSNA"

    return True, orientation


# --------------------------------------------------
# 5. Glowny program
# --------------------------------------------------

filename = r"C:\Users\kacpe\Documents\Obliczenia Inteligentne\cw1\chmura_ABC.xyz"

# Wczytanie punktow
points = load_xyz(filename)

print("Liczba punktow:", len(points))

# Podzial na trzy klastry
clusters, labels = cluster_points(points, k=3)


# Dopasowanie i klasyfikacja kazdego klastra

for i, cluster in enumerate(clusters):

    print(f"\n{'=' * 40}")
    print(f"KLASTER {i}")
    print(f"{'=' * 40}")

    if len(cluster) < 3:
        print("Za malo punktow do dopasowania plaszczyzny.")
        continue

    try:

        coefficients, inliers = ransac_plane(
            cluster,
            iterations=1000,
            threshold=0.01,
            min_inliers=10
        )

        is_plane, orientation = classify_surface(
            cluster,
            coefficients,
            inliers,
            min_inlier_ratio=0.90,
            max_mean_distance=1
        )

        if is_plane:

            a, b, c, d = coefficients

            print("\nTyp powierzchni: PLASZCZYZNA")
            print("Orientacja:", orientation)

            print("\nWektor normalny:")
            print(f"nx = {a:.5f}")
            print(f"ny = {b:.5f}")
            print(f"nz = {c:.5f}")

            print("\nRownanie plaszczyzny:")
            print(f"{a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")

        else:

            print("\nTyp powierzchni: NIE JEST PLASZCZYZNA")
            print("Wektor normalny: nie wyznaczono.")

    except ValueError as error:

        print("\nNie udalo sie dopasowac plaszczyzny:", error)


# --------------------------------------------------
# 6. Wizualizacja wynikow
# --------------------------------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

colors = ["red", "blue", "green"]

for i, cluster in enumerate(clusters):

    ax.scatter(
        cluster[:, 0],
        cluster[:, 1],
        cluster[:, 2],
        c=colors[i],
        s=2,
        label=f"Klaster {i}"
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Podzial chmury punktow - KMeans (k=3)")
ax.legend()

plt.show()
