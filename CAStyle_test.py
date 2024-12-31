import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import MultiLevelAE,SingleLevelAE
import time
import os
import math
from feature_transformer import mean_shift


def test_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

trans = test_transform(512)


def main():
    torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)
    parser = argparse.ArgumentParser(description='CAStyle by Pytorch')
    parser.add_argument('--content',        '-c',   type=str,default = "H:\Project_python\Style_transfer\art_transfer\data\content",       help='Content image path e.g. content.jpg')
    parser.add_argument('--style',          '-s',   type=str,default = "H:\Project_python\Style_transfer\art_transfer\data\style",         help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name',    '-o',   type=str,                           help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha',          '-a',   type=float, default=1,            help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu',            '-g',   type=int, default=0,                help='GPU ID(negative value indicate CPU)')
    parser.add_argument('--model_state_path',       type=str, default='model_state',    help='save directory for result and loss')
    parser.add_argument('--model_layer', type=str, default='m',                         help='save directory for result and loss')
    parser.add_argument('--output', type=str, default='output', help='save directory for result and loss')
    parser.add_argument('--rounds', type=int, default=5, help='save directory for result and loss')
    args = parser.parse_args()


    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
    start = time.time()
    if not os.path.exists(args.output):
            os.makedirs(args.output)

    # set model
    #model = SingleLevelAE(3,args.model_state_path)
    if args.model_layer == "s":
        model = SingleLevelRT(4,args.model_state_path)
    if args.model_layer == "m":
        model = MultiLevelAE(args.model_state_path)

    model = model.to(device)

    content_folder 		= args.content
    style_folder 		= args.style
    content_lists 		= os.listdir(content_folder)
    style_lists 		= os.listdir(style_folder)
    pic_num = 0


    alpha_folder='%f' %args.alpha
    if not os.path.exists(os.path.join(args.output,alpha_folder)):
                os.makedirs(os.path.join(args.output,alpha_folder))
    for content_name in content_lists:
        for style_name in style_lists:
            content_path 	= os.path.join(content_folder, content_name)
            style_path 		= os.path.join(style_folder, style_name)
            c = Image.open(content_path).convert('RGB')
            s = Image.open(style_path).convert('RGB')
            c_tensor = trans(c).unsqueeze(0).to(device)
            s_tensor = trans(s).unsqueeze(0).to(device)

            _, _, ch, cw = c_tensor.shape
            ch_f = ch / 32
            ch_new = math.floor(ch_f) * 32
            cw_f = cw / 32
            cw_new = math.floor(cw_f) * 32
            c_tensor = c_tensor[:, :, 0:ch_new, 0:cw_new]
            aa = 1 / (args.rounds)

            with torch.no_grad():
                for r in range(args.rounds):
                    out = model(c_tensor, s_tensor, args.alpha)
                    c_tensor = aa * out + (1 - aa) * c_tensor
            out = out #

            #if args.output_name is None:
            c_name = os.path.splitext(os.path.basename(content_path))[0]
            s_name = os.path.splitext(os.path.basename(style_path))[0]
            args.output_name = f'{c_name}_{s_name}'

            print("style_strength:",args.alpha,"content:",c_name,",style:",s_name)

            if not os.path.exists(os.path.join(args.output,alpha_folder, c_name)):
                os.makedirs(os.path.join(args.output,alpha_folder, c_name))

            end = time.time()
            print("CAStyle:%.2fs"%(end-start))
            save_image(out, os.path.join(args.output,alpha_folder, c_name,f'{args.output_name}.jpg'), nrow=1)
            
   
    

if __name__ == '__main__':
    main()
