"""
Prepare Multi-Intent Data for AURA LEVEL 2

Creates a mixed dataset with [STORY] and [CODE] tokens, including
"bridge words" that have different meanings in each context.

Bridge words force the model to use CHIRALITY to distinguish meanings:
  - "buffer" -> story: protect, code: memory
  - "branch" -> story: tree, code: git/conditional
  - "loop" -> story: circle, code: iteration
  - "class" -> story: school, code: OOP
  - "root" -> story: plant, code: directory

This is DATA PREPROCESSING only - model architecture unchanged.

Usage:
  python prepare_multi_intent_data.py
"""

import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))


# Python code snippets featuring bridge words
CODE_SNIPPETS = [
    # Buffer examples
    """buffer = []
for item in data:
    buffer.append(item)
    if len(buffer) >= 100:
        process_batch(buffer)
        buffer.clear()""",

    """def read_file_buffered(filename):
    buffer = bytearray()
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)
    return buffer""",

    # Branch examples
    """def check_status(value):
    if value > 0:
        branch = "positive"
    elif value < 0:
        branch = "negative"
    else:
        branch = "zero"
    return branch""",

    """git_branch = subprocess.check_output(
    ["git", "branch", "--show-current"]
).decode().strip()
print(f"Current branch: {git_branch}")""",

    # Loop examples
    """for i in range(10):
    print(f"Loop iteration {i}")
    if i == 5:
        break""",

    """while True:
    user_input = input("Enter command (or 'quit'): ")
    if user_input == 'quit':
        break
    process_command(user_input)
print("Exited loop")""",

    # Class examples
    """class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"

dog = Dog("Buddy")
print(dog.speak())""",

    """class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def process(self, item):
        self.data.append(item)
        return len(self.data)""",

    # Root examples
    """import os

root_dir = os.path.dirname(__file__)
config_path = os.path.join(root_dir, "config.json")
print(f"Root directory: {root_dir}")""",

    """def find_root(node):
    while node.parent is not None:
        node = node.parent
    return node

tree_root = find_root(current_node)""",

    # Mixed examples with multiple bridge words
    """class FileBuffer:
    def __init__(self, root_path):
        self.root = root_path
        self.buffer = []

    def add_file(self, filename):
        full_path = os.path.join(self.root, filename)
        self.buffer.append(full_path)""",

    """def process_branches(tree_root):
    for branch in tree_root.children:
        buffer = []
        for node in branch.nodes:
            buffer.append(node.data)
        yield buffer""",

    # Additional diverse examples
    """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)""",

    """class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()""",

    """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
]


# Story snippets featuring bridge words
STORY_SNIPPETS = [
    # Buffer stories
    """Once upon a time, there was a kind knight who served as a buffer between the kingdom and the dragon.
He protected the villagers from harm and kept them safe.""",

    """The mother bear acted as a buffer for her cubs, standing between them and danger.
She would not let anything hurt her babies.""",

    # Branch stories
    """Lily climbed the tall tree and sat on a sturdy branch.
From there, she could see the whole garden below.""",

    """The river split into two branches, one flowing east and one flowing west.
Tommy didn't know which branch to follow.""",

    # Loop stories
    """The children held hands and formed a big loop in the playground.
They ran in circles, laughing and playing together.""",

    """The roller coaster made a giant loop, turning upside down!
Everyone screamed with excitement.""",

    # Class stories
    """On the first day of class, Emma was nervous.
But her teacher smiled and made everyone feel welcome.""",

    """The class went on a field trip to the zoo.
All the students were excited to see the animals.""",

    # Root stories
    """The old tree had a massive root system underground.
The roots spread far and wide, drinking water from the soil.""",

    """Max pulled the carrot from the ground, but the root was stuck!
He pulled harder and harder until finally it came free.""",

    # Mixed stories with multiple bridge words
    """In class, the teacher drew a tree on the board with many branches.
Each branch had leaves and roots that went deep into the ground.""",

    """The hiking loop took them past a big tree.
They sat in its shade, resting against its strong roots.""",

    # Additional diverse stories
    """The little rabbit was scared of the dark forest.
But his friend the owl told him to be brave.""",

    """Sarah found a magical key that opened a secret door.
Behind it was a room full of toys and games!""",

    """The puppy learned to catch the ball in his mouth.
His owner was so proud and gave him a treat.""",
]


def create_mixed_dataset(
    code_snippets: List[str],
    story_snippets: List[str],
    output_file: Path,
    ratio: float = 0.3,  # 30% code, 70% stories
    verbose: bool = True
):
    """
    Create mixed dataset with [CODE] and [STORY] tokens.

    Args:
        code_snippets: List of Python code examples
        story_snippets: List of story examples
        output_file: Output file path
        ratio: Ratio of code to stories (0.3 = 30% code)
        verbose: Print progress
    """
    if verbose:
        print(f"Creating mixed dataset:")
        print(f"  Code snippets: {len(code_snippets)}")
        print(f"  Story snippets: {len(story_snippets)}")
        print(f"  Code ratio: {ratio:.1%}")
        print()

    # Interleave code and stories based on ratio
    mixed_data = []
    code_idx = 0
    story_idx = 0

    # Calculate how many of each to include
    total_items = len(code_snippets) + len(story_snippets)
    code_every_n = int(1 / ratio) if ratio > 0 else float('inf')

    item_count = 0
    while code_idx < len(code_snippets) or story_idx < len(story_snippets):
        # Add code every N items
        if code_idx < len(code_snippets) and item_count % code_every_n == 0:
            mixed_data.append(f"[CODE] {code_snippets[code_idx]}")
            code_idx += 1
        # Otherwise add story
        elif story_idx < len(story_snippets):
            mixed_data.append(f"[STORY] {story_snippets[story_idx]}")
            story_idx += 1
        # If out of stories, add remaining code
        elif code_idx < len(code_snippets):
            mixed_data.append(f"[CODE] {code_snippets[code_idx]}")
            code_idx += 1

        item_count += 1

    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<|endoftext|>'.join(mixed_data))

    if verbose:
        print(f"[OK] Mixed dataset saved: {output_file}")
        print(f"  Total items: {len(mixed_data)}")
        print(f"  Code items: {code_idx}")
        print(f"  Story items: {story_idx}")
        print()

        # Show examples
        print("Example [CODE] item:")
        print("-" * 70)
        for item in mixed_data:
            if item.startswith('[CODE]'):
                print(item[:200] + "...")
                break
        print("-" * 70)
        print()

        print("Example [STORY] item:")
        print("-" * 70)
        for item in mixed_data:
            if item.startswith('[STORY]'):
                print(item[:200] + "...")
                break
        print("-" * 70)
        print()


def main():
    """Create multi-intent training and validation datasets."""
    print("=" * 70)
    print("PREPARE MULTI-INTENT DATA (AURA LEVEL 2)")
    print("=" * 70)
    print()
    print("Creating mixed [STORY] + [CODE] dataset with bridge words")
    print()
    print("Bridge words (different meaning per context):")
    print("  - buffer: story=protect, code=memory")
    print("  - branch: story=tree, code=git/conditional")
    print("  - loop: story=circle, code=iteration")
    print("  - class: story=school, code=OOP")
    print("  - root: story=plant, code=directory")
    print()
    print("This forces chirality differentiation!")
    print("=" * 70)
    print()

    # Create directories
    data_dir = Path("data/multi_intent")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Split code/stories into train/val (80/20)
    train_ratio = 0.8
    code_train_size = int(len(CODE_SNIPPETS) * train_ratio)
    story_train_size = int(len(STORY_SNIPPETS) * train_ratio)

    code_train = CODE_SNIPPETS[:code_train_size]
    code_val = CODE_SNIPPETS[code_train_size:]
    story_train = STORY_SNIPPETS[:story_train_size]
    story_val = STORY_SNIPPETS[story_train_size:]

    print("CREATING TRAINING DATASET")
    print("=" * 70)
    create_mixed_dataset(
        code_train,
        story_train,
        data_dir / "train.txt",
        ratio=0.3,  # 30% code
        verbose=True
    )

    print("CREATING VALIDATION DATASET")
    print("=" * 70)
    create_mixed_dataset(
        code_val,
        story_val,
        data_dir / "val.txt",
        ratio=0.3,  # 30% code
        verbose=True
    )

    print("=" * 70)
    print("[OK] MULTI-INTENT DATASET READY")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Update train_with_aura.py to use data/multi_intent/")
    print("  2. Train with Aura Level 1 + multi-intent data")
    print("  3. Observe chirality differentiation on bridge words")
    print()
    print("Files created:")
    print(f"  - {data_dir / 'train.txt'}")
    print(f"  - {data_dir / 'val.txt'}")
    print()


if __name__ == "__main__":
    main()
