import os
import time
import argparse
import datetime
import numpy as np

import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from config import get_config
from models.build_model import build_encoder, Projector_MLP, Classifier_Mix,Two_Stream
from data import build_loader
from logger import create_logger
from utils_train_val_ad import train_one_epoch, validate
from utils import load_checkpoint

from models.flow_model import load_flow_model

try:
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size-train', type=int, help="batch size for single GPU")
    parser.add_argument('--batch-size-test', type=int,help="test batch size for single GPU")

    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    

    parser.add_argument("--save_name", type=str, default='')
    parser.add_argument("--model_name", type=str, default='') #swin_ad2stream  swinb

    parser.add_argument("--ad", type=str, default='False')
    parser.add_argument("--ad_dim", type=int, default=128)
    parser.add_argument("--ad_alpha", type=float, default=0.5)
    parser.add_argument("--coupling_layers", type=int, default=6)
    parser.add_argument("--feats_l", type=int, default=2)
    
    parser.add_argument("--ce_alpha", type=float, default=1)
    parser.add_argument("--contrast_alpha", type=float, default=0.1)
    parser.add_argument("--bgspp_lambda", type=float, default=0.1)

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(config):
    dataset_train,data_loader_val,data_loader_test =  build_loader(config)

    config.defrost()
    config.TRAINING_IMAGES = len(dataset_train)
    config.freeze()

    pg_encoder,pg_encoder_projector_rgb,pg_encoder_projector_fre,pg_flow_rgb,pg_flow_fre,pg_classifier_mix = None,None,None,None,None,None
    encoder,encoder_projector_rgb,encoder_projector_fre,decoders_rgb,decoders_fre,classifier_mix = None,None,None,None,None,None
    find_unused_parameters = True # default True
    
    print('name', args.model_name, config.MODEL_NAME,'MODEL_NAME***')
    if config.MODEL_NAME=='swinb':
        from models.swin_model_base import swin_base_patch4_window7_224_in22k as create_model,pretrain
        weights = './models/pretrain/swin_base_patch4_window7_224_22k.pth'
        swinb_M = create_model(num_classes=2)
        pretrain(swinb_M, weights, device='cpu')
        encoder = swinb_M.cuda()
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=find_unused_parameters)
        pg_encoder = [p for p in encoder.parameters() if p.requires_grad]

    elif config.MODEL_NAME == 'swin_ad2stream':

        encoder = build_encoder(config)
        path ='./models/pretrained/swin_base_patch4_window7_224_22k.pth' 
        _ = load_checkpoint(config, path, encoder)

        # encoder two stream
        fre_flag = "high" # high low middle
        encoder = Two_Stream(encoder, cross_at = True, rgb = True, fre = True, fre_flag = fre_flag ) 

        encoder.cuda()
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=find_unused_parameters)
        pg_encoder = [p for p in encoder.parameters() if p.requires_grad]

        # encodrer projector
        out_dim=config.AD_DIM
        encoder_projector_rgb = Projector_MLP(in_dim=512, inner_dim=512, out_dim=out_dim, num_layers=3)
        encoder_projector_rgb.cuda()
        encoder_projector_rgb = torch.nn.parallel.DistributedDataParallel(encoder_projector_rgb, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=find_unused_parameters)
        pg_encoder_projector_rgb = [p for p in encoder_projector_rgb.parameters() if p.requires_grad]
        
        encoder_projector_fre = Projector_MLP(in_dim=512, inner_dim=512, out_dim=out_dim, num_layers=3)
        encoder_projector_fre.cuda()
        encoder_projector_fre = torch.nn.parallel.DistributedDataParallel(encoder_projector_fre, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=find_unused_parameters)
        pg_encoder_projector_fre = [p for p in encoder_projector_fre.parameters() if p.requires_grad]

        # Normalizing Flows rgb
        decoders_rgb = [load_flow_model(config, feat_dim) for feat_dim in [out_dim]]
        decoders_rgb = [decoder.cuda() for decoder in decoders_rgb]
        decoders_rgb=[torch.nn.parallel.DistributedDataParallel(decoders_rgb[l], device_ids=[config.LOCAL_RANK], broadcast_buffers=False,\
                    find_unused_parameters=find_unused_parameters) for l in range(config.FEATS_L)]
        pg_flow_rgb = list(decoders_rgb[0].parameters())
        for l in range(1, config.FEATS_L):
            pg_flow_rgb += list(decoders_rgb[l].parameters())

        # Normalizing Flowsfre
        decoders_fre = [load_flow_model(config, feat_dim) for feat_dim in [out_dim]]
        decoders_fre = [decoder.cuda() for decoder in decoders_fre]
        decoders_fre=[torch.nn.parallel.DistributedDataParallel(decoders_fre[l], device_ids=[config.LOCAL_RANK], broadcast_buffers=False,\
                    find_unused_parameters=find_unused_parameters) for l in range(config.FEATS_L)]
        pg_flow_fre = list(decoders_fre[0].parameters())
        for l in range(1, config.FEATS_L):
            pg_flow_fre += list(decoders_fre[l].parameters())

        #mix classifier        
        classifier_mix = Classifier_Mix(in_dim=512, inner_dim=512, out_dim=512)
        classifier_mix.cuda()
        classifier_mix = torch.nn.parallel.DistributedDataParallel(classifier_mix, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=find_unused_parameters)
        pg_classifier_mix = [p for p in classifier_mix.parameters() if p.requires_grad]

    lr = 5e-5 # default old 5.0 default
    lrf = 0.01 
    import torch.optim.lr_scheduler as lr_scheduler
    model_params = pg_encoder + pg_encoder_projector_rgb + pg_encoder_projector_fre + pg_flow_rgb + pg_flow_fre + pg_classifier_mix

    optimizer = optim.AdamW(model_params, lr=lr, weight_decay=5E-2)
    lf = lambda x: ((1 + math.cos(x * math.pi / config.TRAIN.EPOCHS)) / 2) * (1 - lrf) + lrf  # cosine
    lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    val_auc_best=0
    for epoch in range(config.TRAIN.EPOCHS):
        dataset_train,data_loader_train = build_loader(config, train=True, epoch=epoch)

        data_loader_train.sampler.set_epoch(epoch)
        loss_auc_dict_train = train_one_epoch(config, encoder, data_loader_train, model_params,optimizer, epoch,\
                                                encoder_projector_rgb=encoder_projector_rgb, encoder_projector_fre=encoder_projector_fre,
                                                decoders_rgb=decoders_rgb, decoders_fre=decoders_fre,\
                                                classifier_mix=classifier_mix)
        lr_scheduler.step()

        loss_auc_dict_val = validate(config, encoder, data_loader_val, epoch,
                                        encoder_projector_rgb=encoder_projector_rgb, encoder_projector_fre=encoder_projector_fre,
                                        decoders_rgb=decoders_rgb, decoders_fre=decoders_fre,\
                                        classifier_mix=classifier_mix)
        loss_auc_dict_test = validate(config, encoder, data_loader_test, epoch, \
                                        encoder_projector_rgb=encoder_projector_rgb, encoder_projector_fre=encoder_projector_fre,
                                        decoders_rgb=decoders_rgb, decoders_fre=decoders_fre,\
                                        classifier_mix=classifier_mix)
        #'''
        loss_auc_dict_test_dfdc = {}
        loss_auc_dict_test_dfdc['auc'] = 0
        loss_auc_dict_test_dfdc['auc_ad'] = 0
        loss_auc_dict_test_dfdc['loss_ce'] = 0
        #'''
        if dist.get_rank()==0:
            print('train epoch:{}, train_auc:{:.3f}, loss_ce:{:.3f}'.format(
                                                                    epoch,
                                                                    loss_auc_dict_train['auc'],
                                                                    loss_auc_dict_train['loss_ce'])
                                                                    )
            print('val_ff++ epoch:{}, val_auc:{:.3f}, val_auc_ad:{:.3f}, val_ce:{:.3f}'.format(epoch,
                                                                loss_auc_dict_val['auc'],
                                                                loss_auc_dict_val['auc_ad'],
                                                                loss_auc_dict_val['loss_ce'])
                                                                )
                    
            print('test_ff++ epoch:{}, test_auc:{:.3f}, test_auc_ad:{:.3f}, test_ce:{:.3f}'.format(epoch,
                                                                loss_auc_dict_test['auc'],
                                                                loss_auc_dict_test['auc_ad'],
                                                                loss_auc_dict_test['loss_ce'])
                                                                )

        if dist.get_rank() == 0 :
            model_save_dir = './'+config.OUTPUT + '/'+'weight'+ '/'+config.SAVE_NAME 
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            # two_stream encoder
            os.system('rm -rf {}/encoder*_last.pth'.format(model_save_dir))
            torch.save(encoder.state_dict(), "{}/encoder{}_last.pth".format(model_save_dir,epoch)) 
            torch.save(encoder.state_dict(), "{}/encoder{}.pth".format(model_save_dir,epoch)) 

            if config.MODEL_NAME == 'swin_ad2stream':
                # save last
                os.system('rm -rf {}/classifier_mix*_last.pth'.format(model_save_dir))
                torch.save(classifier_mix.state_dict(), "{}/classifier_mix{}_last.pth".format(model_save_dir,epoch)) 
                os.system('rm -rf {}/encoder_projector_rgb*_last.pth'.format(model_save_dir))
                torch.save(encoder_projector_rgb.state_dict(), "{}/encoder_projector_rgb{}_last.pth".format(model_save_dir,epoch)) 
                os.system('rm -rf {}/encoder_projector_fre*_last.pth'.format(model_save_dir))
                torch.save(encoder_projector_fre.state_dict(), "{}/encoder_projector_fre{}_last.pth".format(model_save_dir,epoch)) 
                os.system('rm -rf {}/decoders_rgb*_last.pth'.format(model_save_dir))
                torch.save(decoders_rgb[0].state_dict(), "{}/decoders_rgb{}_last.pth".format(model_save_dir,epoch)) 
                os.system('rm -rf {}/decoders_fre*_last.pth'.format(model_save_dir))
                torch.save(decoders_fre[0].state_dict(), "{}/decoders_fre{}_last.pth".format(model_save_dir,epoch)) 

                # save all
                torch.save(classifier_mix.state_dict(), "{}/classifier_mix{}.pth".format(model_save_dir,epoch)) 
                torch.save(encoder_projector_rgb.state_dict(), "{}/encoder_projector_rgb{}.pth".format(model_save_dir,epoch)) 
                torch.save(encoder_projector_fre.state_dict(), "{}/encoder_projector_fre{}.pth".format(model_save_dir,epoch)) 
                torch.save(decoders_rgb[0].state_dict(), "{}/decoders_rgb{}.pth".format(model_save_dir,epoch)) 
                torch.save(decoders_fre[0].state_dict(), "{}/decoders_fre{}.pth".format(model_save_dir,epoch)) 

            if loss_auc_dict_val['auc'] > val_auc_best:
                val_auc_best = loss_auc_dict_val['auc']
                os.system('rm -rf {}/encoder*_auc_best.pth'.format(model_save_dir))
                torch.save(encoder.state_dict(), "{}/encoder{}_auc_best.pth".format(model_save_dir,epoch))   
                if config.MODEL_NAME == 'swin_ad2stream':  
                    os.system('rm -rf {}/classifier_mix*_auc_best.pth'.format(model_save_dir))
                    torch.save(classifier_mix.state_dict(), "{}/classifier_mix{}_auc_best.pth".format(model_save_dir,epoch))     
                    torch.save(encoder_projector_rgb.state_dict(), "{}/encoder_projector_rgb{}_auc_best.pth".format(model_save_dir,epoch)) 
                    os.system('rm -rf {}/encoder_projector_fre*_auc_best.pth'.format(model_save_dir))
                    torch.save(encoder_projector_fre.state_dict(), "{}/encoder_projector_fre{}_auc_best.pth".format(model_save_dir,epoch)) 
                    os.system('rm -rf {}/decoders_rgb*_auc_best.pth'.format(model_save_dir))
                    torch.save(decoders_rgb[0].state_dict(), "{}/decoders_rgb{}_auc_best.pth".format(model_save_dir,epoch)) 
                    os.system('rm -rf {}/decoders_fre*_best.pth'.format(model_save_dir))
                    torch.save(decoders_fre[0].state_dict(), "{}/decoders_fre{}_auc_best.pth".format(model_save_dir,epoch)) 


if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if config.LOCAL_RANK == 0:
        name = 'ceAlpha{}_bgspp{}_ct{}_posebeta{}_lr5_Clipgrad1.0_rgbdctHf_AD{}_Clayers{}_dim{}'.format(\
                            config.CE_ALPHA, config.BGSPP_LAMBDA, config.CONTRAST_ALPHA, \
                            config.MODEL.FLOW.POS_BETA,
                            config.AD, 
                            config.MODEL.FLOW.COUPLING_LAYERS, config.AD_DIM,)
        config.defrost()
        config.SAVE_NAME = name
        config.freeze()

    
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = dist.get_rank()
    np.random.seed(seed)
    #'''
    torch.cuda.manual_seed_all(seed) # 
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #'''
    cudnn.benchmark = False

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    
    main(config)
