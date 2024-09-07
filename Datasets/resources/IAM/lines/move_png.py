import os
import shutil

def move_png_to_root_and_count(root_dir):
    png_count = 0
    directory_count = 0

    for root, dirs, files in os.walk(root_dir):
        # Count directories
        directory_count += len(dirs)

        # Move PNG files and count them
        for file in files:
            if file.lower().endswith('.png'):
                png_count += 1
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(root_dir, file)
                # Move the file to root directory, ensuring no overwrite
                if not os.path.exists(dest_file_path):
                    shutil.move(src_file_path, dest_file_path)
                else:
                    # Handling name conflict by appending a suffix
                    base, extension = os.path.splitext(dest_file_path)
                    i = 1
                    new_dest_file_path = f"{base}_{i}{extension}"
                    while os.path.exists(new_dest_file_path):
                        i += 1
                        new_dest_file_path = f"{base}_{i}{extension}"
                    shutil.move(src_file_path, new_dest_file_path)

    return png_count, directory_count

# Replace this with the directory you want to search and move PNGs from
directory_to_search = './'

png_count, directory_count = move_png_to_root_and_count(directory_to_search)
print(f"Number of PNG files moved to root directory: {png_count}")
print(f"Number of directories (including subdirectories): {directory_count}")
