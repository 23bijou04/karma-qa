# test_rag_manual.py
from rag.bootstrap import create_rag


def main():
    rag = create_rag()
    response = rag.answer("Were Scott Derrickson and Ed Wood of the same nationality?")
    print("------------------------------------------------------------------------------------------------------")
    print(response)


if __name__ == "__main__":
    main()