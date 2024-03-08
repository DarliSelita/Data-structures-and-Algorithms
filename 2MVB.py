import numpy as np
import matplotlib.pyplot as plt

def greedy_double_knapsack(capacity1, capacity2, weights, values):
    n = len(weights)
    ratios = [(values[i] / weights[i]) for i in range(n)]
    items = list(range(n))
    items.sort(key=lambda i: -ratios[i])

    knapsack1 = []
    knapsack2 = []

    for i in items:
        if weights[i] <= capacity1:
            knapsack1.append(i)
            capacity1 -= weights[i]
        elif weights[i] <= capacity2:
            knapsack2.append(i)
            capacity2 -= weights[i]

    total_value = sum(values[i] for i in knapsack1) + sum(values[i] for i in knapsack2)
    return knapsack1, knapsack2, total_value

def dynamic_programming_double_knapsack(capacity1, capacity2, weights, values):
    n = len(weights)
    dp = [[0] * (capacity2 + 1) for _ in range(capacity1 + 1)]

    for i in range(n):
        for j in range(capacity1 + 1):
            for k in range(capacity2 + 1):
                if j >= weights[i] and k >= weights[i]:
                    dp[j][k] = max(dp[j][k], dp[j - weights[i]][k - weights[i]] + values[i])

    knapsack1 = []
    knapsack2 = []
    j = capacity1
    k = capacity2

    for i in range(n - 1, -1, -1):
        if j >= weights[i] and k >= weights[i] and dp[j][k] == dp[j - weights[i]][k - weights[i]] + values[i]:
            knapsack1.append(i)
            j -= weights[i]
            k -= weights[i]
        elif k >= weights[i] and dp[j][k] == dp[j][k - weights[i]] + values[i]:
            knapsack2.append(i)
            k -= weights[i]

    total_value = dp[capacity1][capacity2]
    return knapsack1, knapsack2, total_value


# Generate random instances for testing
np.random.seed(42)
num_instances = 100
greedy_distances = []

for _ in range(num_instances):
    capacity1 = np.random.randint(10, 100)
    capacity2 = np.random.randint(10, 100)
    weights = np.random.randint(1, 20, size=50)
    values = np.random.randint(1, 100, size=50)

    knapsack1_greedy, knapsack2_greedy, greedy_value = greedy_double_knapsack(capacity1, capacity2, weights, values)
    knapsack1_dp, knapsack2_dp, dp_value = dynamic_programming_double_knapsack(capacity1, capacity2, weights, values)

    relative_distance = abs(greedy_value - dp_value) / dp_value
    greedy_distances.append(relative_distance)

# Plot histogram
plt.hist(greedy_distances, bins=100, edgecolor='black')
plt.xlabel('Relative Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Relative Distance')
plt.show()

# Calculate statistics
mean = np.mean(greedy_distances)
std_dev = np.std(greedy_distances)
median = np.median(greedy_distances)
max_distance = np.max(greedy_distances)

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Median:", median)
print("Maximum Relative Distance:", max_distance)
