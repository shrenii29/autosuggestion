import kagglehub

# Download latest version
path = kagglehub.dataset_download("disisbig/marathi-wikipedia-articles")

print("Path to dataset files:", path)
