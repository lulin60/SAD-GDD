from cgi import test
import os
import torch
import torch.nn as nn
from PIL import Image
import argparse
from utils_train_val_ad import validate
from data.deepfake import Deepfake as deepfake_dataset
import glob
from models.flow_model import  load_flow_model
from config import get_config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nw = 8  # number of workers

print('Using {} dataloader workers every process'.format(nw))

def parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch-size-test', type=int,help="test batch size for single GPU")
    parser.add_argument('--model_name',type=str, default= 'swin_ad2stream')
    parser.add_argument("--ad", type=str, default='False')
    parser.add_argument("--ad_dim", type=int, default=128)
    parser.add_argument("--coupling_layers", type=int, default=6)
    parser.add_argument("--feats_l", type=int, default=2)
    parser.add_argument('--ckpt_name',type=str, default= None)
    parser.add_argument('--test_data',type=str, default= None)

    parser.add_argument("--gid_flag", type=str, default='')


    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(args,config):
    csv_dir = None

    if args.test_data == 'default':
        csv_dir = 'default' #csv_dir,default

    elif args.test_data == 'ff_32f_raw':
        csv_dir = os.path.join('./datasets','raw_test_dlib2_32frames.csv')

  
    elif args.test_data == 'ff_32f_c23':
        csv_dir = 'ff_32f_c23'


    img_size = 224
    cross_at = True
    rgb = True
    fre = True
    test_type = 'test'
    locate = True
    shuffle = True #True #False

    print(args.test_dist, args.dist_type, args.dist_level,args.gid_flag,'****') 

    test_dataset = deepfake_dataset(phase=test_type, csv_dir=csv_dir, resize=(img_size,img_size))
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size_test,
                                            collate_fn=lambda x:test_dataset.collate_fn(x,test_type),
                                             shuffle=shuffle,pin_memory=True,num_workers=nw)
    
    # two stream encoder
    name = '_last' #
    ckpt_path_encoder = glob.glob('./output/weight/{}/encoder[0-9]*{}.pth'.format(args.ckpt_name, name))[0]
    print('model_name:',args.model_name)

    encoder,encoder_projector_rgb,encoder_projector_fre,decoders_rgb,decoders_fre,classifier_mix = None,None,None,None,None,None
    if args.model_name == 'swinb':
        from models.swin_model_base import swin_base_patch4_window7_224_in22k as create_model
        swinb_M = create_model(num_classes=2)
        encoder = swinb_M
        encoder =torch.nn.DataParallel(encoder.to(device))
        encoder.load_state_dict(torch.load(ckpt_path_encoder,map_location = device),strict=True)

    elif args.model_name == 'swin_ad2stream':
        from models.build_model import build_encoder, Projector_MLP, Classifier_Mix,Two_Stream
        ckpt_path_classifier_mix = glob.glob('./output/weight/{}/classifier_mix*{}.pth'.format(args.ckpt_name,name))[0]
        if locate:
            ckpt_path_encoder_projector_rgb = glob.glob('./output/weight/{}/encoder_projector_rgb*{}.pth'.format(args.ckpt_name,name))[0]
            ckpt_path_encoder_projector_fre = glob.glob('./output/weight/{}/encoder_projector_fre*{}.pth'.format(args.ckpt_name,name))[0]
            ckpt_path_decoders_rgb = glob.glob('./output/weight/{}/decoders_rgb*{}.pth'.format(args.ckpt_name,name))[0]
            ckpt_path_decoders_fre = glob.glob('./output/weight/{}/decoders_fre*{}.pth'.format(args.ckpt_name,name))[0]
        print(ckpt_path_encoder,ckpt_path_classifier_mix,'*****')

        encoder = build_encoder(config)
        fre_flag = 'high'
        encoder = Two_Stream(encoder,cross_at = cross_at,rgb = rgb, fre = fre, fre_flag = fre_flag ) 
        classifier_mix = Classifier_Mix(in_dim=512, inner_dim=512, out_dim=512)

        encoder = encoder.to(device)  
        classifier_mix = classifier_mix.to(device)

        out_dim=config.AD_DIM
        encoder_projector_rgb = Projector_MLP(in_dim=512, inner_dim=512, out_dim=out_dim, num_layers=3)
        encoder_projector_fre = Projector_MLP(in_dim=512, inner_dim=512, out_dim=out_dim, num_layers=3)
        decoders_rgb = [load_flow_model(config, feat_dim) for feat_dim in [out_dim]]
        decoders_fre = [load_flow_model(config, feat_dim) for feat_dim in [out_dim]]

        encoder =torch.nn.DataParallel(encoder.to(device))
        classifier_mix =torch.nn.DataParallel(classifier_mix.to(device))
    
        encoder_projector_rgb =torch.nn.DataParallel(encoder_projector_rgb.to(device))
        encoder_projector_fre =torch.nn.DataParallel(encoder_projector_fre.to(device))
        decoders_rgb =[torch.nn.DataParallel(decoders_rgb[l].to(device)) for l in range(args.feats_l)]
        decoders_fre =[torch.nn.DataParallel(decoders_fre[l].to(device)) for l in range(args.feats_l)]

        encoder.load_state_dict(torch.load(ckpt_path_encoder,map_location = device),strict=True)
        classifier_mix.load_state_dict(torch.load(ckpt_path_classifier_mix ,map_location = device),strict=True)
        if locate:
            encoder_projector_rgb.load_state_dict(torch.load(ckpt_path_encoder_projector_rgb ,map_location = device),strict=True)
            encoder_projector_fre.load_state_dict(torch.load(ckpt_path_encoder_projector_fre ,map_location = device),strict=True)
            decoders_rgb[0].load_state_dict(torch.load(ckpt_path_decoders_rgb,map_location = device),strict=True)
            decoders_fre[0].load_state_dict(torch.load(ckpt_path_decoders_fre ,map_location = device),strict=True)

    epoch = 'test'
    if args.test_flag == 'image':
        loss_auc_dict_test = validate(config, encoder, test_loader, epoch, distributed=False, data_flag = 'val',layers=None,\
                                        encoder_projector_rgb=encoder_projector_rgb,encoder_projector_fre=encoder_projector_fre,\
                                        decoders_rgb=decoders_rgb, decoders_fre=decoders_fre,\
                                        classifier_mix= classifier_mix)
                            
        print('auc:{:.4f}'.format(loss_auc_dict_test['auc']))

    elif args.test_flag == 'video':
        video_auc, video_acc = validate(config, encoder, test_loader, epoch, distributed=False, data_flag = 'val',layers=None,\
                                        encoder_projector_rgb=encoder_projector_rgb,encoder_projector_fre=encoder_projector_fre,\
                                        decoders_rgb=decoders_rgb, decoders_fre=decoders_fre,\
                                        classifier_mix= classifier_mix,
                                        test_flag = 'video', img_list_name=test_dataset.img_list)

        print('csv_dir, video auc:{:.4f}, video_acc:{:.4f}'.format(video_auc,video_acc.item()))

if __name__ == '__main__':
    args, config = parse_option()

    main(args,config)