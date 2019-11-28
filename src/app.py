import sys

from pathlib import Path
from application.app_utils import run


def main():
    print('please be sure to provide data in the same form as it was set up')
    print('probabilities with customers names will be provided in excel file')

    unseen_data_path = Path(str(input('Enter path to customers data: ')))
    output_path = Path(str(input('where do you want to save results? Provide path: ')))
    while not unseen_data_path.exists() or output_path.exists():
        if not unseen_data_path.exists():
            print('are you sure that you pass proper customers data path?')
            unseen_data_path = Path(str(input('Enter path to customers data again: ')))
        if output_path.exists():
            print('Unfortunately, file like this already exists')
            output_path = Path(str(input('where do you want to save results? Provide path: ')))

    run(arg_unseen_data_path=unseen_data_path,
        arg_output_path=output_path)


if __name__ == "__main__":
    main()
