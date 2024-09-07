import os


def delete_png_files(root_dir):
    png_count = 0
    directory_count = 0

    # Count directories and prepare to delete PNGs in root only
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Count directories
        directory_count += len(dirs)

        # If we're in the root directory, delete PNG files
        if root == root_dir:
            for file in files:
                if file.lower().endswith('.png'):
                    png_count += 1
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            break  # Exit after processing the root directory

    return png_count, directory_count


# Directory to search and delete PNGs from
directory_to_search = './'

png_count, directory_count = delete_png_files(directory_to_search)
print(f"Number of PNG files deleted from root directory: {png_count}")
print(f"Number of directories (including subdirectories): {directory_count}")
