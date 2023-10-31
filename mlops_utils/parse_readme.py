def parse_readme(readme_path):
    
    entity_name, project_name, run_id = None, None, None

    with open(readme_path, "r") as f:
        content = f.readlines()

    for line in content:
        if "wandb.ai" in line:
            entity_name = line.split("/")[3]
            project_name = line.split("/")[4]
            run_id = line.split("/")[-1].split()[0]

    return entity_name, project_name, run_id
            

def update_readme(run_id, entity, project, readme_path):
    WANDB_URL = "https://wandb.ai/"
    TRIGGER_README_WORDS = ["best model", "model", "URL"]
    
    with open(readme_path, "r") as f:
        content = f.readlines()

    new_url = f"{WANDB_URL}{entity}/{project}/runs/{run_id}\n"
    
    # get URL index in content
    url_idx = None
    best_model_idx = None
    for i, line in enumerate(content):
        if "wandb.ai" in line:
            url_idx = i
        if any([el.lower() in line.lower() for el in TRIGGER_README_WORDS]):
            best_model_idx = i
    
    if url_idx:
        content[url_idx] = new_url
    else:
        content.insert(best_model_idx+1, new_url)

    with open(readme_path, "w") as f:
        f.writelines(content)

if __name__ == "__main__":
    # Function testing
    #update_readme(900, entity="idealworks-ml", project="sample-mlops", readme_path="../README.md")
    #a, b, c = parse_readme("../README.md")
    pass