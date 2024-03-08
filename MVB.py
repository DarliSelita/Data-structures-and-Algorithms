import random
import time
import numpy as np
import matplotlib.pyplot as plt

# Dynamic programming approach
def dynamic_programming(values, size, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for i in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if size[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - size[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]


def greedy_knapsack_value(values, weights, capacity):
    n = len(values)
    items = list(zip(values, weights))
    items.sort(key=lambda x: x[0], reverse=True)
    max_value = 0
    for v, w in items:
        if w <= capacity:
            max_value += v
            capacity -= w
        if capacity == 0:
            break
    return max_value


# Generate 1000 random cases
random.seed(42)
cases = []
for i in range(1000):
    n = random.randint(10, 50)
    values = [random.randint(1, 40) for j in range(n)]
    size = [random.randint(1, 40) for j in range(n)]
    capacity = random.randint(sum(size) // 2, sum(size))
    cases.append((values, size, capacity))


# Compute maximum value using dynamic programming and greedy algorithm for each case
dp_results = []
greedy_results = []
for i in range(1000):
    values, size, capacity = cases[i]
    dp_result = dynamic_programming(values, size, capacity)
    dp_results.append(dp_result)

    greedy_result = greedy_algorithm_value(values, size, capacity)
    greedy_results.append(greedy_result)


# Compute relative distances between DP and greedy-by-value solutions
rel_distances = []
for i in range(1000):
    dp_result = dp_results[i]
    greedy_result = greedy_results[i]
    rel_distance = (dp_result - greedy_result) / dp_result if dp_result > 0 else 0
    rel_distances.append(rel_distance)


# Plot histogram of relative distances
plt.hist(rel_distances, bins=100, edgecolor='black')
plt.xlabel('Relative distance between DP and greedy-by-value solutions')
plt.ylabel('Frequency')
plt.title('Histogram of relative distances between DP and greedy-by-value solutions')
plt.show()

print("Max relative distance:", max(rel_distances))

# Calculate mean, standard deviation, median, and maximum relative distance for greedy-by-value
mean_value = np.mean(rel_distances)
std_dev_value = np.std(rel_distances)
median_value = np.median(rel_distances)
max_distance_value = np.max(rel_distances)


print(f"Mean: {mean_value:.2f}")
print(f"Standard Deviation: {std_dev_value:.2f}")
print(f"Median: {median_value:.2f}")
print(f"Maximum Relative Distance: {max_distance_value:.2f}")

# Calculate mean of outliers (5% highest relative distances)
outliers_value = sorted(rel_distances)[-int(len(rel_distances) * 0.05):]
mean_outliers_value = np.mean(outliers_value)

print(f"Mean of Outliers (Greedy by value): {mean_outliers_value:.2f}")
