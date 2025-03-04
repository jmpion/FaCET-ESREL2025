import json
import re

def extract_json_from_string(text):
    """Extracts JSON content from a string.

    Args:
        text (str): Input text from which JSON should be extracted.

    Raises:
        Exception: Raises an exception if there are multiple JSONs found
        in the same text.

    Returns:
        dict or None: A dictionary corresponding to a JSON file if any is found
        in the input. None if no JSON is found.
    """
    # Regular expression to match JSON data
    json_pattern = r'\{[\s\S]*?\}'

    # Find all JSON matches
    json_matches = re.findall(json_pattern, text)

    # Convert matches to JSON objects
    json_objects = [json.loads(match) for match in json_matches]

    # Check if there is exactly one json object found.
    try:
        if len(json_objects) == 1:
            return json_objects[0]
        elif len(json_objects) == 0:
            return None
        else:
            raise Exception(f"Number of json objects extracted from log: {len(json_objects)}")
    except Exception as e:
        print(e)

def extract_components_status(logs):
    fc_hash = {} # maps Review_id to failed components string.
    mfc_hash = {} # maps Review_id to maybe-failed components string.
    count_errors = 0 # counts the number of JSON extraction errors.

    # Fill in the fc_hash using the logs.
    for review_id in logs:
        try: # If one JSON pattern, or no JSON pattern, is found, then fill fc_hash.
            log = logs[review_id]
            json_out = extract_json_from_string(log)
            if "Failed components" in json_out:
                fc_log = json_out["Failed components"]
            else:
                fc_log = None
            if "Maybe failed components" in json_out:
                mfc_log = json_out["Maybe failed components"]
            else:
                mfc_log = None
            fc_hash[review_id] = fc_log
            mfc_hash[review_id] = mfc_log
        except Exception as e: # Otherwise, raise an error, and fill with None.
            fc_hash[review_id] = None
            mfc_hash[review_id] = None
            print(review_id)
            print(e)
            count_errors += 1
    print(f"\nNumber of JSON errors: {count_errors}")
    return fc_hash, mfc_hash, count_errors

def hash_to_label_matrix(fc_hash, mfc_hash, only_components):
    # Get all predicted label vectors.
    # TODO: Include label "Maybe failed" (or equivalently 2;)
    labels_matrix = [] # will contain the predicted labels.
    for review_id in fc_hash:
        fc = fc_hash[review_id]
        mfc = mfc_hash[review_id]
        labels_vec = []
        for component in only_components:
            # Initialize conditions for the component and review.
            fc_is_None = fc is None
            mfc_is_None = mfc is None
            if not fc_is_None:
                is_in_fc = component in fc
            else:
                is_in_fc = False
            if not mfc_is_None:
                is_in_mfc = component in mfc
            else:
                is_in_mfc = False

            # Matching patterns.
            if is_in_fc:
                tmp_label = 1
            elif not is_in_fc and is_in_mfc:
                tmp_label = 2
            else:
                tmp_label = 0

            # Append label.
            labels_vec.append(tmp_label)
        labels_matrix.append(labels_vec)
    return labels_matrix

def get_true_labels_matrix(df_components, only_components, fc_hash):
    # Get reference labels of failure reviews.
    true_labels = df_components.loc[:, only_components].to_numpy()

    # Need to complete with "Not failed" reviews.
    idx_failures = 0 # variable index in the failed reviews.
    true_labels_matrix = [] # final reference labels matrix.
    failure_ids = df_components.loc[:, 'Review_id'].to_numpy() # Review_id of failed reviews.
    for review_id in fc_hash:
        if review_id in failure_ids:
            true_labels_matrix.append(true_labels[idx_failures, :].tolist())
            idx_failures += 1
        else:
            true_labels_matrix.append([0] * len(only_components))
    return true_labels_matrix