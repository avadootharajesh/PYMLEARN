# Version_Control.py
import subprocess

def run_git_command(command):
    try:
        result = subprocess.run(['git'] + command.split(), capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    print("Simple Git Automation Script")
    print("Commands: init, status, add <file>, commit <msg>, log")
    while True:
        cmd = input("Enter git command: ").strip()
        if cmd == 'exit':
            break
        run_git_command(cmd)
