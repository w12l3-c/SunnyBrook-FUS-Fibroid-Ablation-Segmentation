import huggingface_hub


# ------------- HuggingFace Repo Git ------------- #
huggingface_hub.create_repo()
huggingface_hub.add_file("README.md")
huggingface_hub.commit("Initial commit")
huggingface_hub.push_to_hub()
