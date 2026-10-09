"""Retriever inference with the shared full-parameter checkpoint."""
from .parsing import parse_retriever_response
from .inference import main as run_inference


def main():
    run_inference('retriever')


if __name__ == '__main__':
    main()
