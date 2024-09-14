import redis
import subprocess

from constants import Datasets_path, LLMs_path


def run_project_in_conda(project_dir, env_name, script_name):
    # Command to activate the specific conda environment and run the project script
    command = f'conda run --name {env_name} python {project_dir}/{script_name}'

    # Run the command in a new subprocess
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for the subprocess to finish before continuing


def main():
    # Initialize Redis connection
    r = redis.Redis()
    pubsub = r.pubsub()
    pubsub.subscribe("project_channel")

    print("Controller is listening for project completion events...")
    # Start by launching the first project in its own conda environment
    run_project_in_conda(Datasets_path, "Datasets_Conda", "main.py")
    for message in pubsub.listen():
        print(f"Received message: {message}")
        if message['type'] == 'message':  # Check if it's a message
            data = message['data'].decode('utf-8')  # Decode the message from bytes to string
            print(f"Decoded message: {data}")
            if data == 'datasets_done':
                print("Project Dataset completed. Starting Project LLMs...")
                run_project_in_conda(LLMs_path, "trocr_1", "main.py")
            # elif message['data'] == b'project2_done':
            #     print("Project 2 completed. Starting Project 3...")
            #     run_project_in_conda("/path/to/project3", "project3_env", "project3.py")
            # elif message['data'] == b'project3_done':
            #     print("Project 3 completed. All projects finished!")


if __name__ == "__main__":
    # Start the controller to listen for project completions
    main()
