from implementations import ours_v, ours_c, log_likelihood
from utils import (
    add_noise,
    load_image,
)


def main():
    # Define parameters
    a = 45
    b = 0.015
    s = 42

    # Load noise-free image
    x, _ = load_image("BSDS300/images/test/3096.jpg")

    # Synthesize and add noise to image
    y = add_noise(x, a, b, s)

    # Compute the estimates using our two approaches
    a_v, b_v = ours_v(x, y)
    a_c, b_c = ours_c(x, y)

    # Compute the log-likelihood for the ground truth parameters
    ll = log_likelihood(x, y, a, b, k_max=100)

    # Display the results
    print("===============")
    print("Ground truth:")
    print(f"a={a}")
    print(f"b={b}")
    print("===============")
    print("Log-likelihood:")
    print(f"LL={round(ll, 3)}")
    print("===============")
    print("Ours_v:")
    print(f"a={round(a_v, 5)}")
    print(f"b={round(b_v, 5)}")
    print("===============")
    print("Ours_c:")
    print(f"a={round(a_c, 5)}")
    print(f"b={round(b_c, 5)}")
    print("===============")


if __name__ == "__main__":
    main()
