from psycop_feature_generation.loaders.raw.load_medications import (
    load as load_medications,
)


def get_antidiabetic_medications():
    df = load_medications(
        atc_code="A10",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=True,
    )

    return df


def get_first_antidiabetic_medication():
    df = get_antidiabetic_medications()

    # Group by person id and sort by timestamp, then get the first row for each person
    df_first_antidiabetic_medication = (
        df.sort_values("timestamp")
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )

    return df_first_antidiabetic_medication


if __name__ == "__main__":
    df = get_first_antidiabetic_medication()

    pass
