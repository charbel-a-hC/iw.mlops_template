from mlops_utils.parse_readme import parse_readme

def main():
    entity_name, project_name, run_id = parse_readme("README.md")
    print(run_id)

if __name__ == "__main__":
    main()