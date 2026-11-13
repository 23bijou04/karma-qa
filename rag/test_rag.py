# test_rag_manual.py
from rag.bootstrap import create_rag


def main():
    rag = create_rag()
    response = rag.answer("what is the current population of Ivory coast?")
    print(response)


if __name__ == "__main__":
    main()