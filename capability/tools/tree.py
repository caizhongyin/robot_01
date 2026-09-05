from pathlib import Path

def tree_with_pathlib(directory, prefix=''):
    """使用 pathlib 打印目录树"""
    directory = Path(directory)
    items = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))

    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}{'/' if item.is_dir() else ''}")

        if item.is_dir():
            extension = "    " if is_last else "│   "
            tree_with_pathlib(item, prefix + extension)

# 使用示例
tree_with_pathlib('.')