import os

tree_file = "tree.out"

ignore = {".git", "__pycache__", ".ipynb_checkpoints"}

tree_out = ""

with open(tree_file, "w", encoding="utf-8") as f:

    def print_tree(path=".", prefix=""):

        items = sorted(os.listdir(path))
        items = [i for i in items if i not in ignore]
        
        for i, item in enumerate(items):
            full_path = os.path.join(path, item)
            connector = "└── " if i == len(items) - 1 else "├── "

            globals()["tree_out"] = globals()["tree_out"] + prefix + connector + item + "\n"
            #print(prefix + connector + item)

            if os.path.isdir(full_path):
                extension = "    " if i == len(items) - 1 else "│   "
                print_tree(full_path, prefix + extension)
    
    print_tree()
    
    f.write(globals()["tree_out"])

print(tree_out)