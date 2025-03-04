import pandas as pd

def load_data_evaluation():
    # Loading two sheets of the dataframe: Reviews for CRD-FD and Component labels.
    df_reviews = pd.read_excel("data/CRD_components.xlsx", sheet_name="Reviews for CRD-FD") # contains the reviews.
    df_components = pd.read_excel("data/CRD_components.xlsx", sheet_name="Component labels") # contains the components and labels.

    # Load reviews.
    reviews = df_reviews['Comment'].to_numpy()

    # Get component names.
    components_columns = df_components.columns
    attributes_to_remove = [
        "Review_id",
        "Failure comment / Summary",
        "Uncertain data flag",
        "Time-to-failure"
        ]
    only_components = [column for column in components_columns if column not in attributes_to_remove]

    # COMPONENTS_TO_DROP = ["Software", "Charging system", "Connection system", "Audio system"]
    COMPONENTS_TO_DROP = []

    only_components = [_ for _ in only_components if _ not in COMPONENTS_TO_DROP]

    return reviews, only_components, df_components