import subprocess
import csv
import matplotlib.pyplot as plt

# Define metrics to be collected
metrics = "gpu__time_duration.avg,gpu__cycles_active.avg,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_targets_threads_uniform.pct,dram__throughput.max.pct_of_peak_sustained_elapsed"

# Define the output CSV file
output_file = "output.csv"

# Header for the CSV file
header = ["BlockSize", "gpu__time_duration.avg", "gpu__cycles_active.avg",
          "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
          "sm__warps_active.avg.pct_of_peak_sustained_active",
          "smsp__sass_average_branch_targets_threads_uniform.pct",
          "dram__throughput.max.pct_of_peak_sustained_elapsed"]

# Data storage for plotting
data = {key: [] for key in header}

# Simulation parameters:
#params = "128 128 100000 0.01 100 1.0 1.0 0.5"
N_GRID_X=128
N_GRID_Y=128
N_PARTICLES=100000
DT=0.01
NSteps=100
Lx=1.0
Ly=1.0
Q_OVER_M = 0.5
params = str(N_GRID_X) + " "\
       + str(N_GRID_Y) + " "\
       + str(N_PARTICLES) + " "\
       + str(DT) + " "\
       + str(NSteps) + " "\
       + str(Lx) + " "\
       + str(Ly) + " "\
       + str(Q_OVER_M) + " "
  
# Open the output CSV file and initialize the writer
with open(output_file, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(header)

    # Iterate over block sizes from 32 to 1024 with a step of 32
    for block_size in range(32, 129, 32):
        try:
            # Run the ncu command
            command = [
                "ncu", "--csv", "--metrics", metrics, "../bin/main", params, str(block_size)
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Running for block size: {block_size}")

            # Parse the stdout from ncu to extract metric values
            lines = result.stdout.splitlines()
            metric_values = [""] * (len(header) - 1)  # Initialize empty metric values list

            # Extract the rows containing metrics and map them to the correct index
            for line in lines:
                # Check if the line contains any of the expected metric names
                for metric_index, metric_name in enumerate(header[1:], start=1):
                    if metric_name in line:
                        fields = line.split(',')
                        # The last field should be the metric value, clean up quotes
                        metric_value = fields[-1].strip('"')
                        metric_values[metric_index - 1] = metric_value
                        break

            # Check if we successfully extracted all the metrics
            if any(metric_values):
                # Write the block size and corresponding metric values to the CSV file
                csvwriter.writerow([block_size] + metric_values)

                # Save data for plotting
                data["BlockSize"].append(block_size)
                for i, metric_name in enumerate(header[1:]):
                    data[metric_name].append(float(metric_values[i]))

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running block size {block_size}: {e}")

print("Completed collecting metrics for all block sizes.")

import subprocess

# Run the script as if you were calling it from the command line
subprocess.run(["python", "plot.py"], check=True)

'''
# Plotting metrics
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

# Plotting the first graph: "gpu__time_duration.avg", "gpu__cycles_active.avg"
plot_metrics(
    data["BlockSize"],
    [data["gpu__time_duration.avg"], data["gpu__cycles_active.avg"]],
    ["gpu__time_duration.avg", "gpu__cycles_active.avg"],
    "GPU Time Duration and Cycles Active vs Block Size",
    "graph_1.png"
)

# Plotting the second graph: "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"
plot_metrics(
    data["BlockSize"],
    [data["l1tex__t_sector_hit_rate.pct"], data["lts__t_sector_hit_rate.pct"]],
    ["l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"],
    "L1 and L2 Texture Sector Hit Rate vs Block Size",
    "graph_2.png"
)

# Plotting the third graph: "sm__warps_active.avg.pct_of_peak_sustained_active",
# "smsp__sass_average_branch_targets_threads_uniform.pct",
# "dram__throughput.max.pct_of_peak_sustained_elapsed"
plot_metrics(
    data["BlockSize"],
    [data["sm__warps_active.avg.pct_of_peak_sustained_active"],
     data["smsp__sass_average_branch_targets_threads_uniform.pct"],
     data["dram__throughput.max.pct_of_peak_sustained_elapsed"]],
    ["sm__warps_active.avg.pct_of_peak_sustained_active",
     "smsp__sass_average_branch_targets_threads_uniform.pct",
     "dram__throughput.max.pct_of_peak_sustained_elapsed"],
    "Warp Activity, Branch Targets Uniform, and DRAM Throughput vs Block Size",
    "graph_3.png"
)

print("Graphs have been saved.")
'''