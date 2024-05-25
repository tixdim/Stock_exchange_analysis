def print_red(text):
    print(f"\033[91m{text}\033[0m")

def print_green(text):
    print(f"\033[92m{text}\033[0m")

def print_yellow(text):
    print(f"\033[93m{text}\033[0m")

# Пример использования
print_red("This is red text")
print_green("This is green text")
print_yellow("This is yellow text")