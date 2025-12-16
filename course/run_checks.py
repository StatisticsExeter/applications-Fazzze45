#!/usr/bin/env python3
import os
import sys
import subprocess

EXPECTED_ENV = "python-exercises"
TEST_FILE = "course/intro/tests/test_db_connection.py"


def get_active_conda_env():
    """
    Detect the active Conda environment using sys.prefix.
    Works cross-platform without calling conda.
    """
    return os.path.basename(sys.prefix)


def run_pytest(test_file):
    """
    Run pytest on the given test file, handling errors cleanly.
    """
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return 1

    try:
        subprocess.run(["pytest", test_file], check=True)
    except FileNotFoundError:
        print("pytest not found in this environment.")
        print("Install it with:")
        print("   conda install pytest")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Pytest failed with return code {e.returncode}")
        return e.returncode
    return 0


def main():
    active_env = get_active_conda_env()
    if active_env == EXPECTED_ENV:
        print(f"Conda environment '{active_env}' is active.")
    else:
        print(f"Expected environment '{EXPECTED_ENV}', but found '{active_env}'.")
        print(f"Please activate the correct environment with:")
        print(f"conda activate {EXPECTED_ENV}")
        sys.exit(1)

    print("Checking database connection via tests...")
    exit_code = run_pytest(TEST_FILE)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
