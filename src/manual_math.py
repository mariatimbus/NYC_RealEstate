"""
manual_math.py — Implementări pure Python pentru funcții statistice
folosite în proiectul NYC Real Estate.
"""

import math


def mean(arr):
    """Media aritmetică."""
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def percentile(arr, p):
    """Percentila p (0-100) folosind interpolare liniară."""
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    if n == 1:
        return sorted_arr[0]
    idx = (p / 100.0) * (n - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_arr[low]
    frac = idx - low
    return sorted_arr[low] + frac * (sorted_arr[high] - sorted_arr[low])


def argsort(arr):
    """Returnează indicii care sortează crescător lista."""
    return sorted(range(len(arr)), key=lambda i: arr[i])


def argsort_desc(arr):
    """Returnează indicii care sortează descrescător lista."""
    return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)


def unique(arr):
    """Returnează valorile unice (ordonate)."""
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return sorted(result)
