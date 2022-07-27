from pathlib import Path
import os
from methods import variance_method, cumulant_method

from utils import (
    add_noise,
    load_image,
)


def main():
    a = 45
    b = 0.015
    s = 42

    x, _ = load_image(
        f"{Path(os.path.abspath(os.path.dirname(__file__)))}/BSDS300/images/test/3096.jpg"
    )
    y = add_noise(x, a, b, s)

    a_var, b_var = variance_method(x, y)
    a_cumulant, b_cumulant = cumulant_method(x, y)

    print("===============================================================")
    print(f"a: {a}")
    print(f"b: {b}")
    print("===============================================================")
    print(f"a variance: {a_var}")
    print(f"b variance: {b_var}")
    print("===============================================================")
    print(f"a cumulant: {a_cumulant}")
    print(f"b cumulant: {b_cumulant}")
    print("===============================================================")


if __name__ == "__main__":
    main()
