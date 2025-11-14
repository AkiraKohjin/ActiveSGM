
import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler


from src.naruto.cfg_loader import argument_parsing, load_cfg
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step

## oneformer
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import json
import os
import wandb

import torch
import numpy as np
from PIL import Image
import random

def modify_metadata(class_info_file, processor):
    new_metadata = {}
    class_names = []
    thing_ids = []
    num_labels = 0

    # Ensure the file exists
    if not os.path.exists(class_info_file):
        raise FileNotFoundError(f"Error: '{class_info_file}' not found.")

    # Load class info JSON
    with open(class_info_file, "r") as f:
        class_info = json.load(f)

    # Process metadata
    for k, v in class_info.items():
        num_labels += 1
        new_metadata[k] = v["name"]
        class_names.append(v["name"])  # Ensure class names are stored
        if v.get("isthing", False):  # Use .get() to avoid KeyError
            thing_ids.append(int(k))

    new_metadata["num_labels"] = num_labels

    new_metadata['class_names'] = class_names
    new_metadata['thing_ids'] = thing_ids

    # Store in processor metadata
    if not hasattr(processor, "image_processor"):
        raise AttributeError("Error: 'processor' object has no attribute 'image_processor'")

    processor.image_processor.metadata = new_metadata
    processor.image_processor.num_labels = num_labels
    print("Metadata modified successfully.")


class CustomDataset(Dataset):
    def __init__(self, processor, img_save_dir):
        self.processor = processor
        self.img_save_dir = img_save_dir

    def __getitem__(self, idx):
        i = idx*10
        color_file = (f"{self.img_save_dir}/color_{i:04d}.jpg")
        image = Image.open(color_file)  # PIL image

        seman_file = f"{self.img_save_dir}/semantic_map_{i:04d}.npy"
        semantic_map = np.load(seman_file)
        semantic_map[semantic_map < 0] = 255
        semantic_map[semantic_map > 101] = 255
        inputs = processor(images=image, segmentation_maps=semantic_map, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        return idx, inputs

    def __len__(self):
        return 200

class CustomDatasetV2(Dataset):
    def __init__(self, processor, root_dirs):
        self.processor = processor
        self.color_paths = []
        self.seman_paths = []
        for root_dir in root_dirs:
            for idx in range(200):
                i = idx * 10
                self.color_paths.append(f"{root_dir}/color_{i:04d}.jpg")
                self.seman_paths.append(f"{root_dirs}/semantic_map_{i:04d}.npy")
    def __getitem__(self, idx):
        image = Image.open(self.color_paths[idx])  # PIL image
        semantic_map = np.load(self.seman_paths[idx])
        semantic_map[semantic_map < 0] = 255
        semantic_map[semantic_map > 101] = 255
        inputs = processor(images=image, segmentation_maps=semantic_map, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        return idx, inputs

    def get_semantic_gt(self,idx):
        semantic_map = np.load(self.seman_paths[idx])
        semantic_map[semantic_map < 0] = 255
        semantic_map[semantic_map > 101] = 255
        return semantic_map
    def __len__(self):
        return len(self.color_paths)

def map_object_id_to_semlabel(object_ids,id2label):
    '''

    :param object_ids: torch.tensor (H,W) # output from habitat-sim, including class id of each pixel
    :return: semantic labels: torch.tensor (H,W) # output from habitat-sim, including class id of each pixel
    '''
    id2label = torch.tensor(id2label)
    sem_labels = id2label[object_ids.long()]
    return sem_labels


def generate_random_colormap(num_classes):
    """
    Generate a random colormap for a given number of classes.

    Args:
        num_classes (int): Number of unique classes in the mask.

    Returns:
        dict: A dictionary mapping class indices to random RGB colors.
    """
    random.seed(42)  # Set seed for reproducibility
    colormap = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}
    return colormap


def semantic_mask_to_rgb(mask: torch.Tensor, save_path: str):
    """
    Convert a semantic mask tensor (H, W) to an RGB image and save it.

    Args:
        mask (torch.Tensor): A tensor of shape (H, W) with semantic class indices.
        save_path (str): Path to save the RGB image.
    """
    # Get the number of unique classes
    unique_classes = torch.unique(mask).tolist()
    #num_classes = max(unique_classes) + 1  # Assuming classes start from 0

    # Generate a random colormap
    num_classes = 102
    colormap = generate_random_colormap(num_classes)

    # Convert mask to numpy
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Create an RGB image
    H, W = mask_np.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Map each class to its corresponding random color
    for class_id in unique_classes:
        rgb_image[mask_np == class_id] = colormap[class_id]

    # Convert to PIL image and save
    img = Image.fromarray(rgb_image)
    img.save(save_path)


def prepare_dataset_and_info_printer(main_cfg,selected_scene,info_printer,is_train=True):
    info_printer("Modifying configuration...", 0, "Initialization")
    main_cfg.dump(os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
    info_printer.update_total_step(int(per_scene_iter * main_cfg.general.num_iter))
    main_cfg.general.scene = selected_scene
    main_cfg.dirs.cfg_dir = f'configs/{main_cfg.general.dataset}/{selected_scene}/'
    main_cfg.sim.habitat_cfg = f'configs/{main_cfg.general.dataset}/{selected_scene}/habitat.py'
    main_cfg.planner.SLAMData_dir = os.path.join(main_cfg.dirs.data_dir, main_cfg.general.dataset,
                                                 main_cfg.general.scene)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)
    ##########  load in id2label ##############
    ori_dir = f"./data/replica_v1/{main_cfg.general.scene[:-1]}_{main_cfg.general.scene[-1]}/habitat/"
    ori_semantic_info_file = os.path.join(ori_dir, 'info_semantic.json')
    with open(ori_semantic_info_file, 'r') as file:
        scene_id2label = json.load(file)['id_to_label']
    main_cfg.general.semantic_dir = ori_dir
    img_save_dir = f'./data/{main_cfg.general.dataset}/{selected_scene}/finetune/'
    os.makedirs(img_save_dir, exist_ok=True)
    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)
    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')
    current_dataset = CustomDataset(processor, img_save_dir)
    dataloader = DataLoader(current_dataset, batch_size=1, shuffle=is_train)
    info_printer("Prepare optimizer...", 0, "Initialization")

    return dataloader


def eval(gt_semantic_map,pred_semantic_map, class_info_file):
    with open(class_info_file, "r") as f:
        class_info = json.load(f)
    eval_dict = {}
    unique_classes = torch.unique(gt_semantic_map).tolist()

def write_dict_to_file(save_dict,save_file):
    with open(save_file) as f:
        for k,v in save_dict:
            f.write(f'{k}： {v.item():.4f}\n')
            f.close()

if __name__ == "__main__":

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sit-visionlab",
        # Set the wandb project where this run will be logged.
        project="finetune_oneformer",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 5e-5,
            "architecture": "oenformer",
            "dataset": "Replica",
            "epochs": 5000,
        },
    )

    info_printer = InfoPrinter("Finetune Oneformer")
    timer = Timer()
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)

    info_printer("Loading oneformer Ade20K pretrained checkpoint...", 0, "Initialization")
    processor = AutoProcessor.from_pretrained(main_cfg.oneformer["checkpoint"])
    model = AutoModelForUniversalSegmentation.from_pretrained(main_cfg.oneformer["checkpoint"], is_training=True)
    # finetune_ckpt = './data/checkpoint/oneformer/finetune/step_10000'
    # model = AutoModelForUniversalSegmentation.from_pretrained(finetune_ckpt, is_training=True)

    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
    class_info_file = './configs/Replica/office0/class_info_file.json'
    modify_metadata(class_info_file=class_info_file,processor=processor)

    info_printer("Prepare optimizer...", 0, "Initialization")

    # finetune_scenes = ["office0", "office1", "office2", "room0", "room1"]
    # test_scenes = ["room2", "office3", "office4"]
    finetune_scenes = ["office2", "office3", "office4", "room0", "room2",]
    test_scenes = ["office0", "office1", "room1"]

    optimizer = AdamW(model.parameters(), lr=5e-5)

    per_scene_iter = 15
    total_step = 0

    num_training_steps = per_scene_iter*200*len(finetune_scenes)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)

    for num_iter in range(per_scene_iter):

        ##################################################
        ### argument parsing and load configuration
        ##################################################
        for selected_scene in finetune_scenes:
            dataloader = prepare_dataset_and_info_printer(main_cfg,selected_scene,info_printer,is_train=True)

            step = 0

            for idx, batch in dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}

                # forward pass
                outputs = model(**batch)

                # backward pass + optimize
                loss = outputs.loss
                info_printer("Training...",step, f"Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()

                run.log({"total_step": total_step, "loss": loss.item()})

                step += 1
                total_step += 1

            save_ckpt_dir = f'./data/checkpoint/oneformer/finetune/step_{total_step}'
            os.makedirs(save_ckpt_dir, exist_ok=True)
            model.save_pretrained(save_ckpt_dir)

        with torch.no_grad():
            for evaluate_scene in test_scenes:

                dataloader = prepare_dataset_and_info_printer(main_cfg, evaluate_scene, info_printer, is_train=False)
                scene_acc = 0
                step = 0
                for idx, batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)

                    gt_idx = idx.item() * 10
                    gt_save_dir = f'./data/{main_cfg.general.dataset}/{evaluate_scene}/finetune/'
                    gt_seman_file = f"{gt_save_dir}/semantic_map_{gt_idx:04d}.npy"
                    gt_semantic_map = np.load(gt_seman_file)
                    gt_semantic_map = torch.from_numpy(gt_semantic_map).to(device)
                    gt_semantic_map[gt_semantic_map<0]=0
                    gt_semantic_map[gt_semantic_map>101]=0

                    img_size = gt_semantic_map.shape
                    semantic_segmentation = \
                    processor.post_process_semantic_segmentation(outputs, target_sizes=[img_size])[0]

                    valid_pixels = img_size[0]*img_size[1] - (gt_semantic_map==0).sum()
                    acc = (semantic_segmentation==gt_semantic_map).sum()/valid_pixels

                    info_printer("Evaluating...", step, f"Acc: {acc.item():.4f}")
                    run.log({"total_step": total_step, "Acc": acc.item()})

                    step += 1
                    scene_acc += acc

                scene_acc /= step
                info_printer("Evaluating...", step, f"Scene Acc: {scene_acc.item():.4f}")


    #############################
    # print out the test semantic mask
    #############################

    model.eval()
    with torch.no_grad():
        for evaluate_scene in test_scenes:

            dataloader = prepare_dataset_and_info_printer(main_cfg, evaluate_scene, info_printer, is_train=False)
            scene_acc = 0
            step = 0
            for idx, batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                gt_idx = idx.item() * 10
                gt_save_dir = f'./data/{main_cfg.general.dataset}/{evaluate_scene}/finetune/'
                gt_seman_file = f"{gt_save_dir}/semantic_map_{gt_idx:04d}.npy"
                gt_semantic_map = np.load(gt_seman_file)
                gt_semantic_map = torch.from_numpy(gt_semantic_map).to(device)
                gt_semantic_map[gt_semantic_map < 0] = 0
                gt_semantic_map[gt_semantic_map > 101] = 0

                img_size = gt_semantic_map.shape
                semantic_segmentation = \
                    processor.post_process_semantic_segmentation(outputs, target_sizes=[img_size])[0]

                semantic_segmentation[semantic_segmentation<0]=0
                semantic_segmentation[semantic_segmentation>101]=0

                save_dir = f'./data/checkpoint/oneformer/finetune/eval/{evaluate_scene}/'
                os.makedirs(save_dir,exist_ok=True)
                semantic_mask_to_rgb(semantic_segmentation, f"{save_dir}/semantic_rgb_{gt_idx:04d}.png")
                valid_pixels = img_size[0] * img_size[1] - (gt_semantic_map == 0).sum()
                acc = (semantic_segmentation==gt_semantic_map).sum()/valid_pixels
                info_printer("Evaluating...", step, f"Acc: {acc.item():.4f}")

                step += 1
                scene_acc += acc
            scene_acc /= step
            info_printer("Evaluating...", step, f"Scene Acc: {scene_acc.item():.4f}")

    run.finish()







