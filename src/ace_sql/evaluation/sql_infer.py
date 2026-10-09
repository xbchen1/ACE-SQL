"""Generator inference with the shared full-parameter checkpoint."""
from .parsing import parse_sql_response
from .inference import main as run_inference


def main():
    run_inference('generator')


if __name__ == '__main__':
    main()
