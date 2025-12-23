import os

# Get the current working directory
cwd = os.getcwd()

# List all files in the current directory
files = os.listdir(cwd)

# Filter for Python files
python_files = [f for f in files if f.endswith('.py')]

# Read and print each Python file
for py_file in python_files:
    print(f"--- File: {py_file} ---\n")
    with open(py_file, 'r', encoding='utf-8') as file:
        code = file.read()
        print(code)
    print("\n" + "-"*40 + "\n")
    