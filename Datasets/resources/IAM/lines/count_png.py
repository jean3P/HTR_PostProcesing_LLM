import os

def count_png_and_directories(root_dir):
    png_count = 0
    directory_count = 0

    for root, dirs, files in os.walk(root_dir):
        # Count directories
        directory_count += len(dirs)

        # Count PNG files
        for file in files:
            if file.lower().endswith('.png'):
                png_count += 1

    return png_count, directory_count

# Replace this with the directory you want to search
directory_to_search = './'

png_count, directory_count = count_png_and_directories(directory_to_search)
print(f"Number of PNG files: {png_count}")
print(f"Number of directories (including subdirectories): {directory_count}")

