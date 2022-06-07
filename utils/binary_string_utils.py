import random
from typing import List


def generate_binary_string(
        length: int
) -> str:
    binary_string: List[str] = []
    for i in range(length):
        binary_string.append(f"{random.randint(0, 1)}")
    return "".join(binary_string)


def generate_binary_string_list(
        length: int
) -> List[str]:
    binary_string: List[str] = []
    for i in range(length):
        binary_string.append(f"{random.randint(0, 1)}")
    return binary_string
