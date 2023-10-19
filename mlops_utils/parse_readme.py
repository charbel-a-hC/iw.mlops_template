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


def update_readme(run_id, readme_path):
    with open(readme_path, "r") as f:
        content = f.readlines()

    new_content = []
    i = 0
    while i < len(content):
        if "wandb.ai" in content[i]:
            new_url = content[i].split("/")[:-1]
            new_url.append(str(run_id))
            content[i] = "/".join(new_url)
        new_content.append(content[i])
        i += 1

    with open(readme_path, "w") as f:
        f.writelines(new_content)
