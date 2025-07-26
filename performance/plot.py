import subprocess
import csv
import matplotlib.pyplot as plt

import csv

output_file = "output.csv"

# Dictionary to store data
data = {}

# Open and read the CSV file
with open(output_file, mode='r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    
    # Initialize keys in the dictionary
    for header in csvreader.fieldnames:
        data[header] = []

    # Read each row and append values to corresponding key lists
    for row in csvreader:
        for header in csvreader.fieldnames:
            value = row[header]
            # Convert to float if possible, otherwise keep as string
            try:
                value = float(value)
            except ValueError:
                pass
            data[header].append(value)


def plot_metrics(x, y_metrics, y_labels, title, filename):
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_metrics, y_labels):
        plt.plot(x, y, label=label)
    plt.xlabel("Block Size")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Step 3: Plot graphs
plot_metrics(
    data["BlockSize"],
    [data["gpu__time_duration.avg"], data["gpu__cycles_active.avg"]],
    ["gpu__time_duration.avg", "gpu__cycles_active.avg"],
    "GPU Time Duration and Cycles Active vs Block Size",
    "time.png"
)

plot_metrics(
    data["BlockSize"],
    [data["l1tex__t_sector_hit_rate.pct"], data["lts__t_sector_hit_rate.pct"]],
    ["l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"],
    "L1 and L2 Texture Sector Hit Rate vs Block Size",
    "L1L2.png"
)

plot_metrics(
    data["BlockSize"],
    [data["sm__warps_active.avg.pct_of_peak_sustained_active"],
     data["smsp__sass_average_branch_targets_threads_uniform.pct"],
     data["dram__throughput.max.pct_of_peak_sustained_elapsed"]],
    ["sm__warps_active.avg.pct_of_peak_sustained_active",
     "smsp__sass_average_branch_targets_threads_uniform.pct",
     "dram__throughput.max.pct_of_peak_sustained_elapsed"],
    "Warp Activity, Branch Targets Uniform, and DRAM Throughput vs Block Size",
    "sm_warp_dram.png"
)

print("Graphs have been saved.")