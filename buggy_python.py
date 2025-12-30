# buggy_code.py
import os
import pickle

API_KEY = "sk-1234567890abcdef"  # Hardcoded secret
DB_PASSWORD = "admin123"  # Another secret

# These will trigger security events:
contact_email = "developer@company.com"  # Triggers pii_email
ssn_example = "123-45-6789"  # Triggers pii_ssn

def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection
    return execute_query(query)

def process_file(filename):
    f = open(filename, 'r')  # File handle never closed
    data = f.read()
    return eval(data)  # Unsafe eval

def load_data(user_input):
    return pickle.loads(user_input)  # Insecure deserialization

def render_page(user_content):
    return f"<html><body>{user_content}</body></html>"  # XSS vulnerability

def divide(a, b):
    return a / b  # No zero division check

def get_item(items, index):
    return items[index]  # No bounds check

class UserManager:
    def __init__(self):
        self.users = {}
    
    def add_user(self, name, password):
        self.users[name] = password  # Storing plaintext password
    
    def authenticate(self, name, password):
        if self.users[name] == password:  # KeyError if user doesn't exist
            return True
        return False
