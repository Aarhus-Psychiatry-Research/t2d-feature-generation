from psycop_feature_generation.loaders.raw.load_medications import (
    load as load_medications,
)

if __name__ == "__main__":
    df_antidiabetic_medications = load_medications(
        atc_code="A10",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=True,
        n_rows=1000,
    )

    pass
