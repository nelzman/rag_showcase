python_local_environment_name := local_dev
make_file_path := $(abspath $(lastword $(MAKEFILE_LIST)))
cwd := $(dir $(make_file_path))

new_environment:
	conda env create --file infrastructure/environment.yml

update_environment:
	conda env update --name gpt_rag --file environment.yml

export_environment:
	conda env export --name gpt_rag > environment.yml

remove_environment:
	conda env remove --name gpt_rag

start_app: 
	python3 -m src.main

build_app: 
	buildozer android debug deploy run

youtube_ipynb_to_pdf:
	jupyter nbconvert --to pdf src/youtube_rag.ipynb

youtube_ipynb_to_html:
	jupyter nbconvert --to html src/youtube_rag.ipynb
