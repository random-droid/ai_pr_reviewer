"""
Sample file with intentional bugs for testing the AI PR reviewer.

To test:
1. Create a new branch: git checkout -b test-pr-review
2. Add this file: git add sample_buggy_code.py
3. Commit: git commit -m "Add sample code for testing"
4. Push: git push origin test-pr-review
5. Create a PR and watch the AI reviewer in action!

Expected issues the reviewer should catch:
- SQL injection vulnerability
- File handle not closed (resource leak)
- Division by zero possibility
- Hardcoded credentials
- Missing null/empty checks
- Potential infinite loop
- Race condition
- Unsafe deserialization
"""

import os
import json
import pickle
import threading


# Issue 1: Hardcoded credentials
DATABASE_PASSWORD = "super_secret_123"
API_KEY = "sk-1234567890abcdef"


def get_user_data(user_id):
    """
    Fetch user data from database.
    Issue: SQL injection vulnerability
    """
    query = f"SELECT * FROM users WHERE id = {user_id}"
    # execute(query)
    return query


def load_config(config_path):
    """
    Load configuration from file.
    Issue: File handle not closed (resource leak)
    """
    f = open(config_path, 'r')
    data = json.load(f)
    # Missing: f.close() or use context manager
    return data


def calculate_average(numbers):
    """
    Calculate average of a list of numbers.
    Issue: Division by zero if list is empty
    """
    total = sum(numbers)
    average = total / len(numbers)
    return average


def process_items(items):
    """
    Process a list of items.
    Issue: No null check, will crash if items is None
    """
    result = []
    for item in items:
        result.append(item.upper())
    return result


def find_element(data, target):
    """
    Find element in nested structure.
    Issue: Potential infinite loop if data has cycles
    """
    visited = []  # Should be a set for O(1) lookup
    queue = [data]

    while queue:
        current = queue.pop(0)
        if current == target:
            return True
        if isinstance(current, dict):
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
        # Missing: cycle detection with visited set

    return False


class Counter:
    """
    Thread-safe counter.
    Issue: Race condition - increment is not atomic
    """

    def __init__(self):
        self.value = 0

    def increment(self):
        # Race condition: read-modify-write is not atomic
        current = self.value
        self.value = current + 1

    def get(self):
        return self.value


def load_user_preferences(data):
    """
    Load user preferences from serialized data.
    Issue: Unsafe pickle deserialization (arbitrary code execution)
    """
    preferences = pickle.loads(data)
    return preferences


def parse_user_input(input_string):
    """
    Parse user input.
    Issue: eval() on user input (code injection)
    """
    result = eval(input_string)
    return result


def get_env_value(key):
    """
    Get environment variable.
    Issue: No default value, will return None silently
    """
    value = os.environ.get(key)
    # Should handle None case or provide default
    return value.strip()  # Will crash if value is None


def write_log(message, filename="app.log"):
    """
    Write message to log file.
    Issue: Path traversal vulnerability
    """
    with open(filename, 'a') as f:
        f.write(message + "\n")
    # filename could be "../../../etc/passwd" etc.


def fetch_url(url):
    """
    Fetch content from URL.
    Issue: No timeout, could hang forever
    Issue: No HTTPS validation
    """
    import urllib.request
    response = urllib.request.urlopen(url)  # No timeout!
    return response.read()


# Issue: Global mutable default argument
def add_item(item, items=[]):
    """
    Add item to list.
    Issue: Mutable default argument (list is shared between calls)
    """
    items.append(item)
    return items


def divide_numbers(a, b):
    """
    Divide two numbers.
    Issue: Generic exception handling hides errors
    """
    try:
        return a / b
    except:  # Bare except - catches everything including KeyboardInterrupt
        return 0


if __name__ == "__main__":
    # Test the functions
    print(get_user_data("1 OR 1=1"))  # SQL injection demo
    print(calculate_average([]))  # Will crash
    print(process_items(None))  # Will crash
