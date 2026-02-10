import os
import json
from torch.utils.data import Dataset
from PIL import Image
import re

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class coco_train(Dataset):
    def __init__(self, transform, data_root, max_words=30, prompt=''):        
        '''
        data_root (string): Root directory of data (e.g. coco/images/)
        '''        
        filename = 'coco_karpathy_train.json'
        
        self.annotation = json.load(open(os.path.join(data_root,filename),'r'))
        self.transform = transform
        self.image_root = data_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        if self.transform is not None:
            image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    def get_all_captions(self):
        captions = []
        for ann in self.annotation:
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)
            captions.append(caption)
        return captions

    
class coco_caption_eval(Dataset):
    def __init__(self, transform, data_root, split):  
        '''
        data_root (string): Root directory of data (e.g. coco/images/)
        split (string): val or test
        '''
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        
        self.annotation = json.load(open(os.path.join(data_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = data_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        if self.transform is not None:
            image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   
    
    
class coco_retrieval_eval(Dataset):
    def __init__(self, transform, data_root, split, max_words=30):  
        '''
        data_root (string): Root directory of data (e.g. coco/images/)
        split (string): val or test
        '''
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        self.annotation = json.load(open(os.path.join(data_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = data_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:    
            image = self.transform(image)  

        return image, index