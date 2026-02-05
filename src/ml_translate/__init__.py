import logging

# Configure logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())


def main() -> None:
    print("Hello from ml-translate!")
