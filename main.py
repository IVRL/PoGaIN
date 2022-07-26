from pathlib import Path
import os
from pg_noise import PGNoise

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

    pg_noise = PGNoise(x, y)

    a_var, b_var = pg_noise.variance_method()
    a_cumulant, b_cumulant = pg_noise.cumulant_method()

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
