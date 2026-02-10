from collections import defaultdict
import os
import argparse

import numpy as np
import torch
import datetime
from PIL import Image

from data import get_dataset, imgprocess, imgprocess_train, textprocess, textprocess_train, collate_fn_train, collate_fn_test
from src.epoch import evaluate_synset_with_similarity
from src.networks import CLIPModel_full
from src.vl_distill_utils import nearest_neighbor, load_or_process_file, kmeans_clustering, \
                                 compute_self_sim, remove_low_sim_pairs, generate_syn_img, load_rep_embed, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('False', 'false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_timestamp(prefix: str="", suffix: str="") -> str:
    tmstamp = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
    return prefix + tmstamp + suffix


def formatting_result_content(val_result):
    return "{img_r1:9.2f} | {img_r5:9.2f} | {img_r10:9.2f} | {txt_r1:9.2f} | {txt_r5:9.2f} | {txt_r10:9.2f} | {r_mean:9.2f}".format(
        **val_result
    )

def formatting_result_content_clean(val_result):
    return "{img_r1} {img_r5} {img_r10} {txt_r1} {txt_r5} {txt_r10} {r_mean}".format(
        **val_result
    )


def main(args):
    set_seed(42)
    
    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    args.clip_model_id = "openai/clip-vit-large-patch14"
    
    lr_img = args.lr_img
    lr_txt = args.lr_txt

    img_path = f'data/syn_data/{args.dataset}/{args.image_encoder}-{args.text_encoder}/{args.num_pairs}'
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(args.embed_path, exist_ok=True)
    
    trainloader, testloader, train_dataset, test_dataset, transform_test = get_dataset(args, [collate_fn_train, collate_fn_test, collate_fn_test])
    

    if args.mode=="distill":

        train_sentences = train_dataset.get_all_captions()  
        
        train_txt = load_or_process_file('train_text', textprocess_train, args, train_sentences)
        train_txt_embed = torch.from_numpy(train_txt['train_embed']).cpu()
        print("The shape of train_caption_embed: {}".format(train_txt_embed.shape))
        train_img = load_or_process_file('train_img', imgprocess_train, args, trainloader)
        
        sim = compute_self_sim(train_img['train_embed'], train_txt['train_embed'], args)
        train_img_prune, train_txt_prune = remove_low_sim_pairs(train_img['train_embed'], train_txt['train_embed'], sim, args.rm_ratio)

        img_rep_embed, txt_rep_embed = kmeans_clustering(train_img_prune, train_txt_prune, args)
        sentence_list = nearest_neighbor(train_sentences, txt_rep_embed, train_txt_embed)
                
        ''' generate the synthetic data '''
        generate_syn_img(img_rep_embed, sentence_list, img_path, args)
            

        
    elif args.mode=="eval":

        test_txt = load_or_process_file('test_text', textprocess, args, testloader)
        test_txt_embed = torch.from_numpy(test_txt['test_embed']).cpu()

        print("The shape of text_test_embed: {}".format(test_txt_embed.shape))
        
        img_rep_embed = load_rep_embed(args, embed_type='image')
        txt_rep_embed = load_rep_embed(args, embed_type='text')

        png_files = sorted([f for f in os.listdir(img_path) if f.endswith(".png")], key=lambda x: int(x.split(".")[0]))
        img_syn = [transform_test(Image.open(os.path.join(img_path, f)).convert("RGB")) for f in png_files]
        img_syn = torch.stack(img_syn)  

        similarity = None            
        for eval_img_encoder in ['nf_resnet50', 'vit' ]: 
             # aggregated results of multiple evaluations
            args.image_encoder = eval_img_encoder
            print('Evaluation\nimage_model = %s, text_model_train = %s'%(eval_img_encoder, args.text_encoder))
            
            multi_eval_aggr_result = defaultdict(list) 
            for it_eval in range(args.num_eval):
                net_eval = CLIPModel_full(args)  
                _, _, best_val_result = evaluate_synset_with_similarity(
                    it_eval, net_eval, img_syn, txt_rep_embed, lr_img, lr_txt,
                    similarity, testloader, args, test_txt_embed)

                
                for k, v in best_val_result.items():
                    multi_eval_aggr_result[k].append(v)

            mean_results = {k: np.mean(v) for k, v in multi_eval_aggr_result.items()}
            std_results = {k: np.std(v) for k, v in multi_eval_aggr_result.items()}
            
            print(formatting_result_content(mean_results))
            print(formatting_result_content(std_results))
            print(formatting_result_content_clean({k: "%.2f$\\pm$%.2f"%(mean_results[k],std_results[k]) for k in std_results}))
                

        print(f'dataset: {args.dataset}, num_pairs: {args.num_pairs}, img_encod : {args.image_encoder}, txt_encod: {args.text_encoder}')
    

    
    print("-----------------------------------------------------------------------------------------")
                
    torch.cuda.empty_cache()


if __name__ == '__main__':
    
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--dataset', type=str, default='flickr', choices=["flickr", "coco"], help='dataset') # flickr, coco
    parser.add_argument('--data_root', type=str, default='./datasets/Flickr30k/', help='dataset path')
    parser.add_argument('--embed_path', type=str, default='./data/embed/', help='embedding path')
    parser.add_argument('--lr_img', type=float, default=0.2, help='learning rate for updating network parameters')
    parser.add_argument('--lr_txt', type=float, default=0.2, help='learning rate for updating network parameters')
    parser.add_argument('--loss_type', default='InfoNCE', type=str, choices=["KL", "BCE", "BalanceBCE", "WBCE", "NCE", "InfoNCE", "MSE","CWCL"])
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--no_aug', action="store_true", default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--text_pretrained', type=str2bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=str2bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=str2bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=str2bool, default=False, help='image_trainable') 
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--normalize_embedding', type=str2bool, default=False, help='normalize before clustering') 
    parser.add_argument('--rm_ratio', type=float, default=0.2, help='pruning ratio')
    parser.add_argument("--infer_num_steps", default = 100,  type=int, help='diffusion sampling steps')
    parser.add_argument("--guidance_scale", default = 5,  type=int, help='classifier-free guidance strength')
    parser.add_argument("--noise_level", default = 10,  type=int, help='sampling noise level')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--image_encoder', type=str, default='clip',  help='image encoder') 
    parser.add_argument('--text_encoder', type=str, default='clip', help='text encoder')
    parser.add_argument('--mode', type=str, default="distill", choices=["distill", "eval"], help='mode of the script')
    parser.add_argument('--num_pairs', type=int, default=100, help='number of distilled pairs')
    
        
    args = parser.parse_args()

    main(args)