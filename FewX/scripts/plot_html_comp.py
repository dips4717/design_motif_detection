import PIL
from moka import *
from tqdm import tqdm
import numpy
import cv2 

def vis_fn(gt, finetune, fewx):
    return dict(
        # GT = gt.show(),
        # Finetune =  finetune.show(),
        # Fewshot =  fewx.show()  
        
        # GT = PIL.Image.open(gt),
        # Finetune = PIL.Image.open(finetune),
        # Fewshot = PIL.Image.open(fewx)
        GT = gt,
        finetune = finetune, 
        Fewshot = fewx
    )

def visualize(samples, web_dir='', show=False, refresh=False):
    # split = 'train' if 'train' in conf.test_dataset else 'val'
        
    # if conf.model_epoch is None:
    html = HTML('Comparison_Plots', 'finetunevsfewx', base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
    # else:
    #     html = HTML(f'/ECCV_retrieval@{conf.exp_name}_epoch_{conf.model_epoch}', conf.expname, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))

    html.add_table().add([vis_fn(*_) for _ in tqdm(samples)])
    html.save()

    # domain = conf.domain if hasattr(conf, 'domain') else None
    # if show: html.show(domain)


txtfile = '/home/dipu/deepdiscover/codes/detection/fewshot_finetuning/Paisley_test.txt'
imfns = open(txtfile, 'r').readlines()
imfns = [x.strip() for x in imfns ]

#gtfolder = '/home/dipu/deepdiscover/codes/detection/fewshot_finetuning/Groundtruth_Plots/test/' 
gtfolder =            '/home/dipu/deepdiscover/codes/detection/fewshot_finetuning/Groudtruth_Plots/test/'
finetunefolder = '/home/dipu/deepdiscover/codes/detection/fewshot_finetuning/Predicted/Paisley/test_thr0.2/'
fewshotfolder  = '/home/dipu/deepdiscover/codes/detection/FewX/output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2/Predicted/test_thr0.2/'

samples = []

for imfn in imfns:
    # gtim = PIL.Image.open(f'{gtfolder}{imfn}.png')
    # ftim = PIL.Image.open(f'{fewshotfolder}{imfn}.png')
    # fewim = PIL.Image.open(f'{fewshotfolder}{imfn}.png')
    
    gtim = cv2.imread(f'{gtfolder}{imfn}.png')
    ftim = cv2.imread(f'{finetunefolder}{imfn}.png')
    fewim =cv2.imread(f'{fewshotfolder}{imfn}.png')
    
    # gtim = numpy.asarray(gtim)
    # ftim = numpy.asarray(ftim)
    # fewim = numpy.asarray(fewim)
    # gtim =f'{gtfolder}{imfn}.png'
    # ftim = f'{fewshotfolder}{imfn}.png'
    # fewim = f'{fewshotfolder}{imfn}.png'
    
    
    samples.append([gtim, ftim, fewim])

visualize(samples, web_dir = '', show=False )