import random
import numpy as np
import matplotlib.pyplot as plt

def dp_allocation(H, n, e):
    m = [[0 for _ in range(H+1)] for _ in range(n+1)]

    for k in range(1, n+1):
        for h in range(H+1):
            for h_prime in range(h+1):
                m[k][h] = max(m[k][h], e[k-1][h_prime] + m[k-1][h-h_prime])

    hours = [0] * n
    k = n
    h = H
    while k > 0 and h > 0:
        for h_prime in range(h+1):
            if m[k][h] == e[k-1][h_prime] + m[k-1][h-h_prime]:
                hours[k-1] += h_prime
                h -= h_prime
                k -= 1
                break

    return hours


def allocate_hours(H, n, e):
    hours = [H // n] * n
    for _ in range(H % n):
        best_topic = -1
        best_increase = 0
        for i in range(n):
            if hours[i] < H // n + 1:
                increase = (e[i][hours[i] + 1] - e[i][hours[i]]) / (hours[i] + 1 + 1e-9)
                if increase > best_increase:
                    best_increase = increase
                    best_topic = i
        hours[best_topic] += 1
    return hours


def generate_instances(num_instances, n, H, mark_range, hour_range):
    instances = []
    for _ in range(num_instances):
        e = [[random.randint(*mark_range) for _ in range(H+1)] for _ in range(n)]
        instances.append((n, H, e))
    return instances


def compute_relative_distances(instances):
    dp_distances = []
    greedy_distances = []
    for instance in instances:
        n, H, e = instance
        dp_hours = dp_allocation(H, n, e)
        greedy_hours = allocate_hours(H, n, e)

        dp_dist = [e[i][dp_hours[i]] for i in range(n)]
        greedy_dist = [e[i][greedy_hours[i]] for i in range(n)]

        dp_distances.extend(dp_dist)
        greedy_distances.extend(greedy_dist)

    relative_distances = [(dp_distances[i] - greedy_distances[i]) / dp_distances[i] for i in range(len(dp_distances))]
    return relative_distances


def calculate_statistics(relative_distances):
    mean = np.mean(relative_distances)
    std_dev = np.std(relative_distances)
    median = np.median(relative_distances)
    max_dist = np.max(relative_distances)

    num_outliers = int(0.05 * len(relative_distances))
    outliers = np.partition(relative_distances, -num_outliers)[-num_outliers:]
    outliers_mean = np.mean(outliers)

    return mean, std_dev, median, max_dist, outliers_mean


def sort_by_value(instance):
    n, H, e = instance
    value_densities = [e[i][H] / (H+1) for i in range(n)]
    sorted_indices = np.argsort(value_densities)[::-1]
    sorted_instance = (n, H, [[e[i][j] for j in range(H+1)] for i in sorted_indices])
    return sorted_instance


num_instances = 1000  # Number of instances to generate
n = 5  # Number of topics
H = 10  # Total study hours
mark_range = (1, 10)  # Range of estimated marks for each topic
hour_range = (1, 5)  # Range of study hours for each topic

instances = generate_instances(num_instances, n, H, mark_range, hour_range)

relative_distances = compute_relative_distances(instances)

mean, std_dev, median, max_dist, outliers_mean = calculate_statistics(relative_distances)

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Median:", median)
print("Maximum Relative Distance:", max_dist)
print("Mean of Outliers:", outliers_mean)

# Greedy by Value Density
greedy_by_value_density_instances = sorted(instances, key=lambda x: sort_by_value(x))
greedy_by_value_density_relative_distances = compute_relative_distances(greedy_by_value_density_instances)
greedy_by_value_density_mean, greedy_by_value_density_std_dev, greedy_by_value_density_median, \
    greedy_by_value_density_max_dist, greedy_by_value_density_outliers_mean =\
    calculate_statistics(greedy_by_value_density_relative_distances)

print("Greedy by Value Density - Mean:", greedy_by_value_density_mean)
print("Greedy by Value Density - Standard Deviation:", greedy_by_value_density_std_dev)
print("Greedy by Value Density - Median:", greedy_by_value_density_median)
print("Greedy by Value Density - Maximum Relative Distance:", greedy_by_value_density_max_dist)
print("Greedy by Value Density - Mean of Outliers:", greedy_by_value_density_outliers_mean)

# Plotting Histogram
positive_relative_distances = [d for d in relative_distances if d > 0]

plt.hist(positive_relative_distances, bins=100, edgecolor='black', linewidth=0.8)
plt.xlabel('Difference in Relative Distances')
plt.ylabel('Frequency')
plt.title('Histogram of Relative Distance Differences')
plt.tight_layout()
plt.show()
