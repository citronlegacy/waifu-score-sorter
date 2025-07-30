# Waifu Image Sorter

This is a command-line tool that uses the [WaifuScorer V3](https://huggingface.co/Eugeoter/waifu-scorer-v3) model on CPU to evaluate the aesthetic quality of anime-style images and automatically organize them into folders based on their scores. GPU is not needed. 

## Features

* Uses the same scoring logic as the official [waifu-scorer-v3 Gradio app](https://huggingface.co/spaces/Eugeoter/waifu-scorer-v3)
* Prompts the user for a directory
* Scores each image using a pretrained aesthetic model
* Sorts images into subfolders `0` through `10` based on their rounded score (higher is better)

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/waifu-image-sorter.git
cd waifu-image-sorter
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script and input the path to a folder containing images:

```bash
python waifu_sorter.py
```

Images will be scored and moved into subdirectories labeled by score bucket:

```
input_folder/
├── 10/
├── 9/
├── 8/
...
├── 0/
```

## Requirements

* Python 3.8+
* PyTorch
* transformers
* pillow
* clip-anytorch
* huggingface-hub
* pytorch-lightning

## License

This project is licensed under the Apache License 2.0.

## Credits

This project uses components of the [waifu-scorer-v3](https://huggingface.co/spaces/Eugeoter/waifu-scorer-v3) by [Eugeoter](https://huggingface.co/Eugeoter), licensed under the Apache License 2.0.

The `utils.py` is copied directly from the original [waifu-scorer](https://github.com/Eugeoter/waifu-scorer) repository.
