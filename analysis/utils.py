from typing import List


def find_longest_common_substring(strs: List[str]):
    if not strs:
        return ""

    shortest_str = min(strs, key=len)
    length = len(shortest_str)

    for sub_len in range(length, 0, -1):
        for start in range(length - sub_len + 1):
            substring = shortest_str[start:start + sub_len]
            if all(substring in s for s in strs):
                return substring
    return ""