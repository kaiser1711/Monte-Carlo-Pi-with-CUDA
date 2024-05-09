import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Set the directory where the folders are located
dir_path = "."  # Replace with the actual directory path if needed

# Create an empty list to store the results
results = []

# Loop through each folder starting with "version_"
for folder in [f for f in os.listdir(dir_path) if f.startswith("version_")]:
    # Get the version number from the folder name
    version = folder.split("_")[1]

    # Construct the path to the CUDA file
    cuda_file = os.path.join(dir_path, folder, f"pi_v{version}.cu")

    # Compile the CUDA file with an output file using the version name and compiler optimizations
    output_file = os.path.join(dir_path, folder, f"pi_v{version}")
    compile_cmd = ["nvcc", cuda_file, "-o", output_file, "-O3", "--use_fast_math"]
    subprocess.run(compile_cmd, check=True)

    # Run the compiled executable
    run_cmd = [output_file]
    result = subprocess.run(run_cmd, stdout=subprocess.PIPE, universal_newlines=True)

    # Save the output to a file with the version number
    output_log_file = f"output_v{version}.txt"
    with open(output_log_file, "w") as f:
        f.write(result.stdout)

    # Extract the "Tests per ms" value from the output
    for line in result.stdout.split("\n"):
        if "Tests per ms:" in line:
            tests_per_ms = float(line.split()[-1])
            tests_per_ns = tests_per_ms / 1000000  # Convert to tests per nanosecond
            results.append({"Version": int(version), "Tests per ns": tests_per_ns})

# Create a pandas DataFrame from the results
df = pd.DataFrame(results)

# Sort the DataFrame by version
df = df.sort_values("Version")
print(df)

# Plot the "Tests per ns" data
plt.figure(figsize=(10, 6))
plt.plot(df["Version"], df["Tests per ns"], marker="o", linestyle="-", color="skyblue")
plt.xlabel("Version", fontsize=14)
plt.ylabel("Samples per ns", fontsize=14)
plt.title("Monte Carlo Pi Samples per ns for Each Version", fontsize=16)
plt.xticks(df["Version"], rotation=45, fontsize=12)
plt.yscale("log")
plt.yticks(fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the plot as a JPG file
plt.savefig("performance_chart.jpg", dpi=300, bbox_inches="tight")