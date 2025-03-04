def prompt_template(prompt_choice:str, components_list: list[str], review: str) -> str:
    if prompt_choice=='Good_baseline':
        return f'Given the list of components of a tablet computer: {components_list}, Give me the list of failed components mentioned in the following review: {review}\nProvide the information in JSON format: {{ "Failed components": "...", "Maybe failed components": "...", "Details": "..." }}'
    elif prompt_choice=='Enhanced_1':
        return f'Given the list of components of a tablet computer: {components_list}, Give me the list of failed components mentioned in the following review: {review}\nProvide the information in JSON format: {{ "Failed components": "...", "Maybe failed components": "...", "Details": "..." }}\nIn addition, here are guidelines to follow:\n- If the review specifies that at least one component failed, all components that are not mentioned as failed should be considered as not failed.\n- However, if no component is specified as failed, but we know based on the review that the unit failed, then all components should be considered as maybe failed.'
    elif prompt_choice=='Enhanced_2':
        return f'Given the list of components of a tablet computer: {components_list}, Give me the list of failed components mentioned in the following review: {review}\nProvide the information in JSON format: {{ "Failed components": "...", "Maybe failed components": "...", "Details": "..." }}\nIn addition, here are guidelines to follow:\n- If the review specifies that at least one component failed, all components that are not mentioned as failed should be considered as not failed.\n- However, if no component is specified as failed, but we know based on the review that the unit failed, then all components should be considered as maybe failed.\n - Besides, if no component is mentioned as failed and no failure is reported, all components should be labeled as not failed, thus should not appear in maybe-failed components.'