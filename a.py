import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import glob
from tqdm import tqdm
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def findFiles(path): return glob.glob(path)

img_paths = findFiles("../task2/data/test/test_img/*.jpg")
result_path = "./test_result.txt"
for _, img_path in tqdm(img_paths):
    for seed in range(5):
        setup_seed(seed=seed)
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        raw_image = Image.open(img_path)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        # generate caption
        k = model.generate({"image": image}, use_nucleus_sampling=True)
        print(f"{Path(img_path).name}#{seed}\t{k}\n")
        with open(result_path, 'a+') as file:
            file.write(f"{Path(img_path).name}#{seed}\t{k}\n")
            