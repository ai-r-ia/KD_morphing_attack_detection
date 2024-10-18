import os
import pickle
import pandas as pd

# Define the path to the 'log' directory
log_dir = "/home/ubuntu/volume/mad_kd/KD_morphing_attack_detection/optim_src/logs"  # Adjust this path if needed
print("aeroh")

all_feret_eers = []
if not os.path.exists(log_dir):
    print(f"Directory '{log_dir}' does not exist. Please check the path.")
else:
    # Loop through all subdirectories and files in the log directory
    for root, dirs, files in os.walk(log_dir):
        # print(f"Checking in directory: {root}")  # Debug statement
        for file in files:
            # print(f"Found file: {file}")  # Debug statement
            if file == "eer_testdb_feret_student2.pkl":
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                try:
                    # Open and load the pickle file
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        all_feret_eers.append(data)
                        # Print the content of the file
                        # print(f"Data from {file_path}:")
                        # print(data)  # Adjust this to print specific values if needed
                        # print("=" * 40)  # Separator for readability
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")


# print(all_feret_eers)

consolidated_data = {}

for entry in all_feret_eers:
    for key, value in entry.items():
        if key not in consolidated_data:
            consolidated_data[key] = value
        else:
            for sub_key, sub_value in value.items():
                consolidated_data[key][sub_key] = sub_value

# Print the condensed dictionary
print(consolidated_data)

# RENAME BASELINE

# consolidated_data = {
#     "teacher_lmaubo": {
#         "greedy": 50.92,
#         "lmaubo": 49.19,
#         "mipgan2": 49.39,
#         "mordiff": 50.07,
#     },
#     "teacher_lma": {
#         "greedy": 51.38,
#         "lmaubo": 48.23,
#         "mipgan2": 51.21,
#         "mordiff": 51.45,
#     },
#     "baseline_lmaubo_lma_post_process": {
#         "greedy": 35.84,
#         "lmaubo": 50.79,
#         "mipgan2": 38.25,
#         "mordiff": 51.0,
#     },
#     "student_post_process_llmbd_1.0": {
#         "greedy": 26.98,
#         "lmaubo": 50.95,
#         "mipgan2": 48.81,
#         "mordiff": 40.15,
#     },
#     "teacher_post_process": {
#         "greedy": 42.56,
#         "lmaubo": 50.74,
#         "mipgan2": 43.84,
#         "mordiff": 51.98,
#     },
#     "student_lma_llmbd_1.0": {
#         "greedy": 50.85,
#         "lmaubo": 43.23,
#         "mipgan2": 49.3,
#         "mordiff": 50.86,
#     },
#     "teacher_stylegan": {
#         "greedy": 51.22,
#         "lmaubo": 48.07,
#         "mipgan2": 51.66,
#         "mordiff": 50.88,
#     },
#     "teacher_Morphing_Diffusion_2024": {
#         "greedy": 50.35,
#         "lmaubo": 49.04,
#         "mipgan2": 50.84,
#         "mordiff": 51.18,
#     },
#     "baseline_mipgan2_stylegan_Morphing_Diffusion_2024": {
#         "greedy": 51.4,
#         "lmaubo": 49.15,
#         "mipgan2": 50.32,
#         "mordiff": 50.87,
#     },
#     "student_mipgan2_llmbd_1.0": {
#         "greedy": 46.65,
#         "lmaubo": 50.28,
#         "mipgan2": 12.0,
#         "mordiff": 49.73,
#     },
#     "teacher_mipgan2": {
#         "greedy": 47.23,
#         "lmaubo": 49.97,
#         "mipgan2": 51.2,
#         "mordiff": 47.11,
#     },
#     "student_Morphing_Diffusion_2024_llmbd_1.0": {
#         "greedy": 44.16,
#         "lmaubo": 50.31,
#         "mipgan2": 31.38,
#         "mordiff": 45.94,
#     },
#     "student_stylegan_llmbd_1.0": {
#         "greedy": 50.06,
#         "lmaubo": 48.03,
#         "mipgan2": 48.57,
#         "mordiff": 49.56,
#     },
#     "student_lmaubo_llmbd_1.0": {
#         "greedy": 53.57,
#         "lmaubo": 52.03,
#         "mipgan2": 52.24,
#         "mordiff": 53.19,
#     },
# }
df = pd.DataFrame(consolidated_data).T
print(df)

# Initialize dictionaries for averages
average_teacher_data = {
    key: {"sum": 0.0, "count": 0}
    for key in next(iter(consolidated_data.values())).keys()
}
average_student_data = {
    key: {"sum": 0.0, "count": 0}
    for key in next(iter(consolidated_data.values())).keys()
}
average_baseline_data = {
    key: {"sum": 0.0, "count": 0}
    for key in next(iter(consolidated_data.values())).keys()
}

# Loop through each entry in the consolidated data
for name, entry in consolidated_data.items():
    if name.startswith("teacher"):  # Check if the entry is for teachers
        for key, value in entry.items():
            average_teacher_data[key]["sum"] += value
            average_teacher_data[key]["count"] += 1
    elif name.startswith("student"):  # Check if the entry is for students
        for key, value in entry.items():
            average_student_data[key]["sum"] += value
            average_student_data[key]["count"] += 1
    elif name.startswith("baseline"):  # Check if the entry is for students
        for key, value in entry.items():
            average_baseline_data[key]["sum"] += value
            average_baseline_data[key]["count"] += 1

# Calculate the average for teachers and students
teacher_averages = {
    key: data["sum"] / data["count"]
    for key, data in average_teacher_data.items()
    if data["count"] > 0
}
student_averages = {
    key: data["sum"] / data["count"]
    for key, data in average_student_data.items()
    if data["count"] > 0
}
baseline_averages = {
    key: data["sum"] / data["count"]
    for key, data in average_baseline_data.items()
    if data["count"] > 0
}

# Convert to DataFrame for better display
teacher_averages_df = pd.DataFrame(
    list(teacher_averages.items()), columns=["Metric", "Average (Teachers)"]
)
student_averages_df = pd.DataFrame(
    list(student_averages.items()), columns=["Metric", "Average (Students)"]
)
baseline_averages_df = pd.DataFrame(
    list(baseline_averages.items()), columns=["Metric", "Average (Baseline)"]
)

# Merge both DataFrames for a combined view
averages_combined = pd.merge(
    teacher_averages_df, student_averages_df, on="Metric", how="outer"
)
averages_combined2 = pd.merge(
    averages_combined, baseline_averages_df, on="Metric", how="outer"
)

print(averages_combined2)
