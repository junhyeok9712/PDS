"""Move some basic utils in distill.py in VL-Distill here"""
import os
import numpy as np
import torch

import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import joblib
from PIL import Image
from huggingface_hub import login
from diffusers import StableUnCLIPImg2ImgPipeline

__all__ = [
    "nearest_neighbor",
    "load_or_process_file",
]


def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """
    Find the nearest neighbors for a batch of embeddings.
    """
    nearest_neighbors = []
    
    
    similarities = cosine_similarity(query_embeddings, database_embeddings)

    most_similar_indices = np.argmax(similarities, axis=1)

    nearest_neighbors = [sentences[i] for i in most_similar_indices]
        
    return nearest_neighbors



def load_or_process_file(file_type, process_func, args, data_source):
    """
    Load the processed file if it exists, otherwise process the data source and create the file.

    Args:
    file_type: The type of the file (e.g., 'train', 'test').
    process_func: The function to process the data source.
    args: The arguments required by the process function and to build the filename.
    data_source: The source data to be processed.

    Returns:
    The loaded data from the file.
    """
    if 'img' in file_type:
        filename = f'{args.embed_path}/{args.dataset}_{args.image_encoder}_{file_type}_embed.npz'
    elif 'text' in file_type:
        filename = f'{args.embed_path}/{args.dataset}_{args.text_encoder}_{file_type}_embed.npz'

    if not os.path.exists(filename):
        print(f'Creating {filename}')
        process_func(args, data_source)
    else:
        print(f'Loading {filename}')
    
    return np.load(filename)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kmeans_clustering(img_embeds, txt_embeds, args):
    
    if not os.path.exists(f'data/center/{args.image_encoder}-{args.text_encoder}'):
        os.makedirs(f'data/center/{args.image_encoder}-{args.text_encoder}', exist_ok=True)
        
    img_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_img_kmeans_centers_{args.num_pairs}.pkl'
    txt_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_text_kmeans_centers_{args.num_pairs}.pkl'
    
        
    if not os.path.exists(img_center_path) or not os.path.exists(txt_center_path):    
        
        if args.normalize_embedding:
            image_embeds_norm = normalize(img_embeds, axis=1)
            text_embeds_norm = normalize(txt_embeds, axis=1)

            kmeans_img = MiniBatchKMeans(n_clusters=args.num_pairs, random_state=42, batch_size=10000, n_init=20)
            img_labels = kmeans_img.fit_predict(image_embeds_norm)
            
            kmeans_txt = MiniBatchKMeans(n_clusters=args.num_pairs, random_state=42, batch_size=10000, n_init=20)
            txt_labels = kmeans_txt.fit_predict(text_embeds_norm)
            
            img_centers = np.zeros((args.num_pairs, image_embeds_norm.shape[1]), dtype=np.float32)
            txt_centers = np.zeros((args.num_pairs, text_embeds_norm.shape[1]), dtype=np.float32)
            
            for k in range(args.num_pairs):
                img_idx = np.where(img_labels == k)[0]
                txt_idx = np.where(txt_labels == k)[0]
                if len(img_idx) > 0:
                    img_centers[k] = img_embeds[img_idx].mean(axis=0)
                if len(txt_idx) > 0:
                    txt_centers[k] = txt_embeds[txt_idx].mean(axis=0)
        
        else:
            kmeans_img = MiniBatchKMeans(n_clusters=args.num_pairs, random_state=42, batch_size=10000, n_init=20)
            img_labels = kmeans_img.fit_predict(img_embeds)
            img_centers = kmeans_img.cluster_centers_

            kmeans_txt = MiniBatchKMeans(n_clusters=args.num_pairs, random_state=42, batch_size=10000, n_init=20)
            txt_labels = kmeans_txt.fit_predict(txt_embeds)
            txt_centers = kmeans_txt.cluster_centers_


        df = pd.DataFrame({'index': np.arange(len(img_labels)), 'img_cluster': img_labels, 'txt_cluster': txt_labels})

        count_table = df.groupby(['img_cluster', 'txt_cluster']).size().unstack(fill_value=0)
        cost_matrix = -count_table.values
        img_idxs, txt_idxs = linear_sum_assignment(cost_matrix)
        
        matched_img_centers, matched_txt_centers = match_and_sort_centers(img_labels, txt_labels, img_idxs, txt_idxs, 
                                                                          img_centers, txt_centers, img_embeds, txt_embeds, 
                                                                          args) 
        
        joblib.dump(matched_img_centers, img_center_path)
        joblib.dump(matched_txt_centers, txt_center_path)
        
        print(f"Cluster centers are saved")          
            
    return joblib.load(img_center_path), joblib.load(txt_center_path)


def match_and_sort_centers(img_labels, txt_labels, img_idxs, txt_idxs, img_centers, txt_centers, img_embeds, txt_embeds, args):


    matched_img_centers = []
    matched_txt_centers = []

    for i, j in zip(img_idxs, txt_idxs):
        mask = (img_labels == i) & (txt_labels == j)
        num_matched = np.sum(mask)
        print(f"Matched Image cluster {i} with Text cluster {j} -> {num_matched} samples")

        if num_matched == 0 and args.num_pairs < 300:
            matched_img_centers.append(img_centers[i])
            matched_txt_centers.append(txt_centers[j])

        else:
            matched_img_centers.append(img_embeds[mask].mean(axis=0))
            matched_txt_centers.append(txt_embeds[mask].mean(axis=0))

    matched_img_centers = np.stack(matched_img_centers, axis=0)
    matched_txt_centers = np.stack(matched_txt_centers, axis=0)

    norm_img = matched_img_centers / np.linalg.norm(matched_img_centers, axis=1, keepdims=True)
    norm_txt = matched_txt_centers / np.linalg.norm(matched_txt_centers, axis=1, keepdims=True)
    sims_np = np.sum(norm_img * norm_txt, axis=1)
    sorted_indices = np.argsort(sims_np)[::-1]

    return matched_img_centers[sorted_indices], matched_txt_centers[sorted_indices]



def load_rep_embed(args, embed_type='text', get_origin=False):
    

    img_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_img_kmeans_centers_{args.num_pairs}.pkl'
    txt_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_text_kmeans_centers_{args.num_pairs}.pkl'

    
    if embed_type == 'text':
        if not os.path.exists(txt_center_path):
            raise FileNotFoundError(f"Text centers file not found: {txt_center_path}")
        return joblib.load(txt_center_path)
    
    elif embed_type == 'image':
        if not os.path.exists(img_center_path):
            raise FileNotFoundError(f"Image centers file not found: {img_center_path}")
        return joblib.load(img_center_path)



def remove_low_sim_pairs(img_embeds, txt_embeds, sim, remove_ratio=0.1):

    assert len(img_embeds) == len(sim)
    assert 0 <= remove_ratio < 1

    num_to_remove = int(len(sim) * remove_ratio)
    if num_to_remove == 0:
        return img_embeds, txt_embeds

    sorted_indices = np.argsort(sim)
    remove_indices = sorted_indices[:num_to_remove]

    keep_mask = torch.ones(len(sim), dtype=torch.bool)
    keep_mask[remove_indices] = False

    return img_embeds[keep_mask], txt_embeds[keep_mask]


def compute_self_sim(img_embeds, txt_embeds, args, prune=False):
    

    assert len(img_embeds) == len(txt_embeds), "List lengths must match"

    norm_img = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    norm_txt = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    sims_np = np.sum(norm_img * norm_txt, axis=1) 
    
    return sims_np 






def generate_syn_img(img_emdeds, sentence_list, img_path, args):
    if sentence_list is not None:
        assert len(img_emdeds) == len(sentence_list), "Image and text embeddings must have the same length"
    
    decoder_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(args.device)

    if sentence_list is None:
        sentence_list = [""]*len(img_emdeds)  # Default empty prompt if none provided
    
    os.makedirs(f'{img_path}', exist_ok=True)
    for idx, (img_emded, txt_promt) in enumerate(zip(img_emdeds, sentence_list)):
        save_path = f'{img_path}/{idx}.png'
        img_emded = torch.tensor(img_emded, dtype=torch.float16).to(args.device)
        
        # Image generation using Unclip
        negative_prompt= "text, watermark" 
        decoder_output = decoder_pipe(prompt=txt_promt, negative_prompt=negative_prompt, \
                                      image_embeds=img_emded.unsqueeze(0), num_inference_steps=args.infer_num_steps, \
                                      guidance_scale=args.guidance_scale, noise_level=args.noise_level)

        
        img_generated = decoder_output.images[0]
            
        # Resize and save
        img_resized = img_generated.resize((args.image_size, args.image_size), resample=Image.LANCZOS)  #Image.NEAREST, Image.BILINEAR, Image.BICUBIC
        img_resized.save(save_path)
        
