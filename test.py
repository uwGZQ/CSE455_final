import os

# Define the root directory to search in
root_dir = "/data3/cse455/hmdb51_org_256x256q5_rgb_flow"

# Initialize a list to store directories with a different number of images than expected
directories_with_diff_img_count = []

# Define the expected number of images in each "img" directory
expected_img_count = 8

# Walk through the directory
for subdir, dirs, files in os.walk(root_dir):
    if subdir.endswith("/img"):  # Check if the directory ends with "/img"
        img_count = len([file for file in files if file.endswith((".jpg", ".png"))])  # Count only image files
        if img_count != expected_img_count:
            directories_with_diff_img_count.append(subdir)
            print(subdir)

# print(directories_with_diff_img_count)
