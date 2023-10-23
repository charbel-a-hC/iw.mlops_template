def parse_readme(readme_path):
    with open(readme_path, "r") as f:
        content = f.readlines()

    for line in content:
        if "wandb.ai" in line:
            entity_name = line.split("/")[3]
            project_name = line.split("/")[4]
            run_id = line.split("/")[-1].split()[0]
            break

    return entity_name, project_name, run_id


def update_readme(run_id, entity, project, readme_path):
    WANDB_URL = "https://wandb.ai/"

    with open(readme_path, "r") as f:
        content = f.readlines()

    new_content = []
    
    new_url = f"{WANDB_URL}{entity}/{project}/runs/{run_id}\n"
    for line in content:
        if "wandb.ai" in line:
            new_content.append(new_url)
        else:
            new_content.append(line)

    with open(readme_path, "w") as f:
        f.writelines(new_content)

if __name__ == "__main__":
    # Function testing
    update_readme(800, entity="idealworks-ml", project="sample-mlops", readme_path="../README.md")