

def load_file(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        return [line.strip().split("\t") for line in f if line.strip()]