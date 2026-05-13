import os

IGNORE = {".venv", "__pycache__", ".git"}

def print_tree(start_path, prefix=""):
    items = [i for i in os.listdir(start_path) if i not in IGNORE]
    items.sort()

    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = index == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    print("\n📁 CLEAN PROJECT TREE:\n")
    print_tree(".")