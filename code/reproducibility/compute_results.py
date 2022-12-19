"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.
"""

import itertools
import os
import sys
from pathlib import Path

import multiprocess as mp
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import implementations as imp
import utils
import w2s.data_bridge as data_bridge


def prepare_csv(file_name):
    csvs_path = os.path.join(os.path.dirname(__file__), "csvs")

    if not os.path.exists(csvs_path):
        os.makedirs(csvs_path)

    path = os.path.join(csvs_path, f"{file_name}.csv")

    if os.path.isfile(path):
        os.remove(path)

    with open(path, "w+") as f:
        f.write("Name,Method,Seed,Real_a,Real_b,Est_a,Est_b\n")

    return path


def combine_csvs():
    csvs_path = os.path.join(os.path.dirname(__file__), "csvs")
    results_path = os.path.join(csvs_path, "results.csv")

    if os.path.isfile(results_path):
        os.remove(results_path)

    files = os.listdir(csvs_path)

    csvs = [read_csv(os.path.join(csvs_path, f)) for f in files]

    df = pd.concat(csvs, ignore_index=True)

    df.to_csv(
        results_path,
        na_rep="NaN",
        columns=df.columns,
        index=False,
    )


def compute_results_real(
    images,
    a_values,
    b_values,
    seeds,
):
    path = prepare_csv("real")

    with open(path, "a+") as f:
        for image in images:
            image_name = image.replace(".", "_")

            for a, b, s in itertools.product(a_values, b_values, seeds):
                f.write(f"{image_name},REAL,{s},{a},{b},{a},{b}\n")


def compute_results(
    method,
    images_path,
    images,
    a_values,
    b_values,
    seeds,
):
    path = prepare_csv(method)

    if method == "var":
        estimator = imp.var
    elif method == "ours":
        estimator = imp.ours
    else:
        raise NotImplementedError

    with open(path, "a+") as f:
        for image in images:
            x, _ = utils.load_image(os.path.join(images_path, image))
            image_name = image.replace(".", "_")

            for a, b, s in itertools.product(a_values, b_values, seeds):
                np.random.seed(s)
                y = utils.add_noise(x, a, b, seed=s)
                est_a, est_b = estimator(x, y)
                if est_b == 0:
                    est_b = 1e-8
                f.write(f"{image_name},{method.upper()},{s},{a},{b},{est_a},{est_b}\n")


def compute_results_foi(
    images_path,
    images,
    a_values,
    b_values,
    seeds,
):
    def compute(a):
        current_a = f"a_{a}"
        data_bridge.set_current_a(current_a)

        with open(path, "a+") as f:
            for b, s in itertools.product(b_values, seeds):
                np.random.seed(s)

                data_bridge.prepare_data(a, b, s)

                os.system(
                    f"matlab -nodisplay -nojvm -nosplash -nodesktop -r \"cd('{w2s_path}'); get_preds('{current_a}');exit\""
                )

                data_bridge.delete_prepared_data()

                name, est_a, est_b = data_bridge.get_results()

                for i in range(len(name)):
                    temp = name[i].replace(".mat", "_jpg")
                    f.write(f"{temp},FOI,{s},{a},{b},{est_a[i]},{est_b[i]}\n")

    path = prepare_csv("foi")
    w2s_path = os.path.join(os.path.dirname(__file__), "w2s")

    data_bridge.set_images(images_path, images)

    with mp.Pool(mp.cpu_count()) as p:
        p.map(compute, a_values)


def compute_results_cnn(images_path, images, a_values, b_values, seeds, dim):
    import tensorflow as tf  # Avoid imports if not needed
    from cnn.model import fix_gpu

    session = fix_gpu()  # Needed to add this to make TF work on my GPU

    model = tf.keras.models.load_model(
        os.path.join(os.path.dirname(__file__), "cnn/model")
    )

    path = prepare_csv("cnn")

    with open(path, "a+") as csv_file:
        for image in images:
            x, shape = utils.load_image(os.path.join(images_path, image))
            image_name = image.replace(".", "_")

            for a, b, s in itertools.product(a_values, b_values, seeds):
                y = utils.add_noise(x, a, b, seed=s)
                y = np.reshape(y, shape)[0 : dim[0], 0 : dim[0]]

                input_var = y[np.newaxis, :, :, np.newaxis]

                input_var = tf.convert_to_tensor(input_var)
                pred = model.predict(input_var)[0]

                csv_file.write(f"{image_name},CNN,{s},{a},{b},{pred[0]},{pred[1]}\n")

    session.close()


def append_csv_extension(name: str) -> str:
    if name[-4:] != ".csv":
        name += ".csv"

    return name


def append_nan(bad_line: list[str]) -> list[str]:
    bad_line.append(np.nan)
    return bad_line


def read_csv(name: str) -> pd.DataFrame:
    name = append_csv_extension(name)
    df = pd.read_csv(
        name,
        header=0,
        comment="#",
        skipinitialspace=True,
        engine="python",
        on_bad_lines=append_nan,
    )
    df.sort_values(by=["Real_b", "Real_a"], ascending=[True, True], inplace=True)
    return df


def compute_ll(image_path, filename: str):
    def helper(df: pd.DataFrame):
        if (
            (df.Est_b > 1e-8)
            and (df.Est_a > 0)
            and (df.LL != df.LL)  # LL != LL is true when LL is nan
        ):
            name: str = df.Name
            seed = df.Seed
            real_a = df.Real_a
            real_b = df.Real_b
            est_a = df.Est_a
            est_b = df.Est_b

            x, _ = utils.load_image(f"{image_path}/{name.replace('_','.')}")
            y = utils.add_noise(x, real_a, real_b, seed=seed)

            ll = imp.log_likelihood(x, y, est_a, est_b)
            df.LL = ll

        return df

    # https://stackoverflow.com/a/66353746
    def parallelize_dataframe(df, func):
        num_processes = mp.cpu_count()
        df_split = np.array_split(df, num_processes)
        with mp.Pool(num_processes) as p:
            df = pd.concat(p.map(func, df_split))
        return df

    def parallelize_function(df):
        return df.apply(helper, axis=1)

    csv_path = os.path.join(os.path.dirname(__file__), "csvs", filename)

    df = read_csv(csv_path)

    if "LL" not in df.columns:
        df["LL"] = np.nan

        df = parallelize_dataframe(df, parallelize_function)
        df.to_csv(
            os.path.join(os.path.dirname(__file__), f"csvs/{filename}.csv"),
            na_rep="NaN",
            columns=df.columns,
            index=False,
        )


if __name__ == "__main__":
    images_path = f"{Path(os.path.abspath(os.path.dirname(__file__))).parent.parent}/BSDS300/images/test"

    num_images = 10
    images = os.listdir(images_path)[:num_images]
    a_values = np.linspace(1, 100, num=25)
    b_values = np.linspace(0.01, 0.15, num=25)
    seeds = np.arange(10)

    seed = 42
    side_length = 321
    dims = (side_length, side_length)

    epochs = 2000
    batch_size = 8

    validation_percentage = 0.1

    filters = (16, 32, 64)
    loss_type = "mean_absolute_percentage_error"

    # ==========================================================================

    # Reduced set of parameters for testing

    # num_images = 3
    # images = os.listdir(images_path)[:num_images]
    # a_values = np.linspace(1, 100, num=10)
    # b_values = np.linspace(0.01, 0.15, num=10)
    # seeds = np.arange(3)

    # seed = 42
    # side_length = 321
    # dims = (side_length, side_length)

    # epochs = 200
    # batch_size = 8

    # validation_percentage = 0.1

    # filters = (16, 32, 64)
    # loss_type = "mean_absolute_percentage_error"

    # ==========================================================================

    print("Computing results real...")
    compute_results_real(
        images,
        a_values,
        b_values,
        seeds,
    )

    # ==========================================================================

    print("Computing results var...")
    compute_results(
        "var",
        images_path,
        images,
        a_values,
        b_values,
        seeds,
    )

    # ==========================================================================

    print("Computing results ours...")
    compute_results(
        "ours",
        images_path,
        images,
        a_values,
        b_values,
        seeds,
    )

    # ==========================================================================

    print("Computing results foi...")
    compute_results_foi(
        images_path,
        images,
        a_values,
        b_values,
        seeds,
    )

    # ==========================================================================

    from cnn.model import model, train_cnn

    model_type = model

    print("Training cnn...")
    train_cnn(
        seed,
        dims,
        epochs,
        batch_size,
        validation_percentage,
        model_type,
        filters,
        loss_type,
    )

    # ==========================================================================

    print("Computing results cnn...")
    compute_results_cnn(
        images_path,
        images,
        a_values,
        b_values,
        seeds,
        dims,
    )

    # ==========================================================================

    print("Computing LL real...")
    compute_ll(images_path, "real")

    # ==========================================================================

    print("Computing LL var...")
    compute_ll(images_path, "var")

    # ==========================================================================

    print("Computing LL ours...")
    compute_ll(images_path, "ours")

    # ==========================================================================

    print("Computing LL foi...")
    compute_ll(images_path, "foi")

    # ==========================================================================

    print("Computing LL cnn...")
    compute_ll(images_path, "cnn")

    # ==========================================================================

    print("Combine csvs...")
    combine_csvs()
