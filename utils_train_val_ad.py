
import sys
import time
import datetime
import numpy as np
from tqdm import tqdm
from  sklearn.metrics import roc_auc_score as AUC,average_precision_score,roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch
import torchvision
import torch.nn.functional as F
from timm.utils import AverageMeter
from utils import reduce_tensor, gather_tensor
from utils_ad import get_logp, MetricRecorder,  convert_to_anomaly_scores,convert_to_image_loglogits
from losses.loss import SupConLoss,get_logp_boundary, calculate_bg_spp_loss, normal_fl_weighting, abnormal_fl_weighting
from models.flow_model import positionalencoding2d, activation
import os
import matplotlib.pyplot as plt
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
log_theta = torch.nn.LogSigmoid()


def video_metric(img_list_name, pred_sigmoid_list):
    assert len(img_list_name)==len(pred_sigmoid_list)

    video_pred_dict = {}
    for i in range(len(img_list_name)):
        if '/FF++/' in img_list_name[i]:
            video_name = img_list_name[i].rsplit('/',1)[0]
            
        if video_name not in video_pred_dict.keys():
            video_pred_dict[video_name]=[]
        video_pred_dict[video_name].append(pred_sigmoid_list[i])
    
    for item in video_pred_dict.keys():
        video_pred_dict[item]=np.average(np.array(video_pred_dict[item]))
    
    label_list=[]
    for item in video_pred_dict.keys():
        if ('real' in item) or ('original_sequences' in item):
            label_list.append(0)
        else:
            label_list.append(1)

    video_auc = AUC(label_list,list(video_pred_dict.values()))
    video_pred_dict_class = [1 if x > 0.5 else 0 for x in list(video_pred_dict.values()) ] 
    video_acc =np.equal(np.array(video_pred_dict_class), np.array(label_list)).mean()
    
    pred_fake_count = np.sum(np.array(label_list)*np.array(video_pred_dict_class))
    pred_real_count = np.sum((1-np.array(label_list))*(1-np.array(video_pred_dict_class)))
    fake_acc = pred_fake_count/np.sum(np.array(label_list))
    real_acc = pred_real_count/np.sum(1-np.array(label_list))

    print('GT real:{},fake:{}'.format(len(label_list)-sum(np.array(label_list)),\
                                        sum(np.array(label_list))))

    video_ap = average_precision_score(label_list,list(video_pred_dict.values()))
    fpr, tpr, thresholds = roc_curve(label_list,list(video_pred_dict.values()))

    video_eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print('video all auc:{}, ap:{:.4f} video_acc:{:.4f}, eer:{:.4f}, real_acc:{:.4f}, fake_acc:{:.4f},'.format(\
                  video_auc, video_ap, video_acc, video_eer, real_acc, fake_acc))

    return video_auc, video_acc


def ad_flow_loss(config,logps,m_b):
    if config.MODEL.FLOW.Focal_WEIGHTING:
        logps_detach = logps.detach()
        normal_logps = logps_detach[m_b == 0]
        anomaly_logps = logps_detach[m_b == 1]
        nor_weights = normal_fl_weighting(normal_logps)
        ano_weights = abnormal_fl_weighting(anomaly_logps)
        weights = nor_weights.new_zeros(logps_detach.shape)
        weights[m_b == 0] = nor_weights
        weights[m_b == 1] = ano_weights
        loss_ml = -log_theta(logps[m_b == 0]) * nor_weights 
        loss_ml = torch.mean(loss_ml)
    else:
        loss_ml = -log_theta(logps[m_b == 0])
        loss_ml = torch.mean(loss_ml)
    boundaries = get_logp_boundary(logps, m_b, pos_beta=config.MODEL.FLOW.POS_BETA, margin_tau=config.MODEL.FLOW.MARGIN_TAU, normalizer=config.MODEL.FLOW.NORMALIZER) # paper

    if config.MODEL.FLOW.Focal_WEIGHTING:
        loss_bn, loss_ba = calculate_bg_spp_loss(logps, m_b, boundaries, config.MODEL.FLOW.NORMALIZER, weights=weights) 

    else:
        loss_bn, loss_ba = calculate_bg_spp_loss(logps, m_b, boundaries, config.MODEL.FLOW.NORMALIZER)  
    loss_ad = loss_bn + loss_ba


    return loss_ad, loss_ml


def train_one_epoch(config, model, data_loader, model_params,optimizer, epoch, distributed=True,\
                        encoder_projector_rgb=None, encoder_projector_fre=None,\
                        decoders_rgb=None,decoders_fre=None,\
                        classifier_mix=None):
    data_loader = tqdm(data_loader, file=sys.stdout)
    num_steps = len(data_loader)
    model.train()
    optimizer.zero_grad()
   
    loss_meter = AverageMeter()
    loss_meter_ce =  AverageMeter()
    loss_meter_ad = AverageMeter()
    loss_meter_ml = AverageMeter()
    loss_meter_bn = AverageMeter()
    loss_meter_ba = AverageMeter()
    acc_meter =  AverageMeter()

    contrast_alpha = config.CONTRAST_ALPHA
    flag_contrast = True if contrast_alpha>1e-4 else False
    ml_alpha = 1 

    pred_classifer_list = []
    labels_list = []

    for idx, data in enumerate(data_loader):

        imgs,labels,file_index = data['img'],data['label'],data['file_index']
        hard_mask = data['hard_mask']
        logps_list_rgb = [list() for _ in range(config.FEATS_L)]
        logps_list_fre = [list() for _ in range(config.FEATS_L)]

        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        hard_mask = hard_mask.cuda(non_blocking=True).to(torch.float) 

        loss_ce = torch.tensor(0).cuda(non_blocking=True)
        loss_contrast_rgb = torch.tensor(0).cuda(non_blocking=True)
        loss_contrast_fre = torch.tensor(0).cuda(non_blocking=True)
        loss_contrast = torch.tensor(0).cuda(non_blocking=True)

        loss = torch.tensor(0).cuda(non_blocking=True)
        loss_ad = torch.tensor(0).cuda(non_blocking=True)
        loss_bn = torch.tensor([0]).cuda(non_blocking=True)
        loss_ba = torch.tensor([0]).cuda(non_blocking=True)

        pred_sigmoid = None
        pred_classes = None

        if config.MODEL_NAME == "swinb":
            pred = model(imgs)
            loss_ce = F.cross_entropy(pred, labels)
            pred_classes = (torch.max(pred, dim=1)[1])#classifier out as out 
            pred_sigmoid = torch.nn.functional.softmax(pred,dim=1)[:,1]

            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if config.MODEL_NAME == "swin_ad2stream" and config.AD:
            alpha_ce = config.CE_ALPHA if epoch < 1 else 1 
            encoder_x = model(imgs)

            for l in range(config.FEATS_L):
                fea_rgbfre, rgb, fre = encoder_x
                #e = encoder_x

                if epoch==0:
                    rgb = rgb[labels.float()==0]
                    fre = fre[labels.float()==0]
                    hard_mask = hard_mask[labels.float()==0]
                    
                rgb = encoder_projector_rgb(rgb)
                bs,hw,c = rgb.size() 
                rgb = rgb.permute(0,2,1)
                rgb = rgb.reshape(bs,c,int(hw**0.5),int(hw**0.5))
                bs, dim, h, w = rgb.size()
                rgb = rgb.permute(0, 2, 3, 1).reshape(-1, dim)

                fre = encoder_projector_fre(fre)
                fre = fre.permute(0,2,1).reshape(bs,c,int(hw**0.5),int(hw**0.5)).permute(0, 2, 3, 1).reshape(-1, dim)

                mask = F.interpolate(hard_mask[:,None,],size=(h,w),mode = 'nearest').squeeze(1)
                mask = mask.reshape(-1)
                m_b = mask

                pos_embed = positionalencoding2d(config.MODEL.FLOW.POS_EMBED_DIM, h, w).cuda(non_blocking=True).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, config.MODEL.FLOW.POS_EMBED_DIM)
                decoder_rgb = decoders_rgb[l]
                decoder_fre = decoders_fre[l]

                if config.MODEL.FLOW.FLOW_ARCH == 'flow_model':
                    z_rgb, log_jac_det_rgb = decoder_rgb(rgb)  
                    z_fre, log_jac_det_fre = decoder_fre(fre)  
                else:
                    z_rgb, log_jac_det_rgb = decoder_rgb(rgb, [pos_embed, ])
                    z_fre, log_jac_det_fre = decoder_fre(fre, [pos_embed, ])

                # first epoch only training normal samples
                if epoch == 0:
                #if epoch < 0:
                    pred = classifier_mix(fea_rgbfre)
                    pred_classes = (torch.max(pred, dim=1)[1]) #classifier out as out 
                    pred_sigmoid = torch.nn.functional.softmax(pred,dim=1)[:,1]
                    loss_ce = F.cross_entropy(pred, labels)
                    
                    #2ad:rgb+fre
                    logps_rgb = get_logp(dim, z_rgb, log_jac_det_rgb) 
                    logps_rgb = logps_rgb / dim      # likelihood per dim
                    loss_ml_rgb = -log_theta(logps_rgb).mean()

                    logps_fre = get_logp(dim, z_fre, log_jac_det_fre) 
                    logps_fre = logps_fre / dim      # likelihood per dim
                    loss_ml_fre = -log_theta(logps_fre).mean()

                    if flag_contrast==True:
                        z_rgb_fea = z_rgb.view(bs,int(hw**0.5),int(hw**0.5),c).permute(0,3,1,2).reshape(bs,c,hw).permute(0,2,1)
                        z_fre_fea = z_fre.view(bs,int(hw**0.5),int(hw**0.5),c).permute(0,3,1,2).reshape(bs,c,hw).permute(0,2,1)
                        if labels[labels.float()==0].shape[0]>1:
                            loss_contrast_rgb = SupConLoss(z_rgb_fea)
                            loss_contrast_fre = SupConLoss(z_fre_fea)

                    if model.module.swin_rgb and model.module.swin_fre:
                        loss_ml = loss_ml_rgb + loss_ml_fre
                        loss_contrast = loss_contrast_rgb +  loss_contrast_fre

                    elif model.module.swin_rgb  and not model.module.swin_fre:
                        loss_ml = loss_ml_rgb
                        loss_contrast = loss_contrast_rgb 

                    elif not model.module.swin_rgb and model.module.swin_fre:
                        loss_ml = loss_ml_fre
                        loss_contrast = loss_contrast_fre

                    #All loss
                    #loss = loss_ce
                    #loss = alpha_ce*loss_ce 
                    loss = alpha_ce*loss_ce + loss_ml * ml_alpha + loss_contrast*contrast_alpha
                    #print(flag_contrast, loss_ce.data.cpu(), loss_ml.data.cpu(), loss_contrast.data.cpu()*contrast_alpha, '*****')

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    logps_rgb = get_logp(dim, z_rgb, log_jac_det_rgb) 

                    logps_rgb = logps_rgb / dim     
                    logps_bhw_rgb = logps_rgb.reshape(bs,h,w)
                    logps_list_rgb[l].append(logps_bhw_rgb)

                    logps_fre = get_logp(dim, z_fre, log_jac_det_fre) 
                    logps_fre = logps_fre / dim      
                    logps_bhw_fre = logps_fre.reshape(bs,h,w)
                    logps_list_fre[l].append(logps_bhw_fre)

                    pred = classifier_mix(fea_rgbfre)

                    pred_classes = (torch.max(pred, dim=1)[1])
                    pred_sigmoid = torch.nn.functional.softmax(pred,dim=1)[:,1]
                    loss_ce = F.cross_entropy(pred, labels)

                    loss_ad_rgb, loss_ml_rgb= ad_flow_loss(config, logps_rgb, m_b)
                    loss_ad_fre , loss_ml_fre= ad_flow_loss(config,logps_fre, m_b)

                   
                    # loss contrast
                    if flag_contrast==True:                    
                        z_rgb_fea = z_rgb.view(bs,int(hw**0.5),int(hw**0.5),c).permute(0,3,1,2).reshape(bs,c,hw).permute(0,2,1)
                        z_fre_fea = z_fre.view(bs,int(hw**0.5),int(hw**0.5),c).permute(0,3,1,2).reshape(bs,c,hw).permute(0,2,1)
                        if labels[labels.float()==0].shape[0]>1:
                            loss_contrast_rgb = SupConLoss(z_rgb_fea[labels.float()==0])
                            loss_contrast_fre = SupConLoss(z_fre_fea[labels.float()==0])
                    
                    if model.module.swin_rgb and model.module.swin_fre:
                        loss_ml = loss_ml_rgb + loss_ml_fre
                        loss_ad = loss_ad_rgb  + loss_ad_fre
                        loss_contrast = (loss_contrast_rgb +  loss_contrast_fre)*contrast_alpha

                    elif model.module.swin_rgb  and not model.module.swin_fre:
                        loss_ml = loss_ml_rgb
                        loss_ad = loss_ad_rgb 
                        loss_contrast = loss_contrast_rgb * contrast_alpha

                    elif not model.module.swin_rgb and model.module.swin_fre:
                        loss_ml = loss_ml_fre
                        loss_ad = loss_ad_fre
                        loss_contrast = loss_contrast_fre*contrast_alpha

                    loss_ad = config.BGSPP_LAMBDA * loss_ad
                    loss =  alpha_ce*loss_ce + loss_ml*ml_alpha + loss_ad + loss_contrast

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)

                    optimizer.step()
                    torch.distributed.barrier()

        torch.cuda.synchronize()

        if distributed:
            loss_meter_ce.update(reduce_tensor(loss_ce.data).item(), labels.size(0))
            loss_meter.update(reduce_tensor(loss.data).item(), labels.size(0))
            
            if config.AD:
                loss_meter_ad.update(reduce_tensor(loss_ad.data).item(),labels.size(0))
                loss_meter_ml.update(reduce_tensor(loss_ml.data).item(),labels.size(0))
                loss_meter_ba.update(reduce_tensor(loss_ba.data).item(),labels.size(0))
                loss_meter_bn.update(reduce_tensor(loss_bn.data).item(),labels.size(0))

            labels_g= gather_tensor(labels)
            labels_list.extend(labels_g.cpu().numpy().tolist())

            pred_sigmoid_g = gather_tensor(pred_sigmoid)
            pred_classifer_list.extend(pred_sigmoid_g.cpu().numpy().tolist())
        else:
            loss_meter.update(loss.data.item(), labels.size(0))
            loss_meter_ce.update(loss_ce.data.item(),labels.size(0))
            
            labels_list.extend(labels.cpu().numpy().tolist())
            pred_classifer_list.extend(pred_sigmoid.detach().cpu().numpy().tolist())

        
        acc= torch.eq(pred_classes, labels).float().mean()
        acc_meter.update(acc.item(),labels.size(0))

        data_loader.desc = "[train epoch {}] loss:{:.3f}, loss_ce:{:.3f}, loss_meter_ml:{:.3f}, loss_sup:{:.3f}, acc:{:.3f}".format(
                                                                epoch,
                                                                loss_meter.avg, 
                                                                loss_meter_ce.avg,
                                                                loss_meter_ml.avg,
                                                                loss_meter_ad.avg,
                                                                acc_meter.avg,
                                                                )

        del loss 


    labels_list[0]=0
    auc = AUC(labels_list,pred_classifer_list)
    loss_auc_dict={
        'loss_ce':loss_meter_ce.avg,
        'loss':loss_meter.avg,
        'loss_ad':loss_meter_ad.avg,
        'loss_ml':loss_meter_ml.avg,
        'loss_bn':loss_meter_bn.avg,
        'loss_ba':loss_meter_ba.avg,
        'auc':auc,
        'acc':acc_meter.avg
        }
    return loss_auc_dict

@torch.no_grad()
def validate(config,model, dataloader, epoch, distributed=True,\
                encoder_projector_rgb=None,encoder_projector_fre=None,\
                decoders_rgb=None, decoders_fre=None,\
                classifier_mix= None,
                test_flag = 'image', img_list_name=False,ff_fake_type=None):
    

    loss_meter_ce = AverageMeter()
    loss_meter_ad = AverageMeter()
    acc_meter =  AverageMeter()

    labels_list = []
    pred_classifer_list = []
    pred_classifer_list_ad_rgb = []
    pred_classifer_list_ad_rgb_real = []
    pred_classifer_list_ad_rgb_fake = []
    
    pred_classifer_list_ad_rgb_real_loglist = []
    pred_classifer_list_ad_rgb_fake_loglist = []


    file_names_list = []
    total_loss, loss_count = 0.0, 0

    model.eval()
    if config.MODEL_NAME == 'swin_ad2stream':
        classifier_mix.eval()
        encoder_projector_rgb.eval()
        encoder_projector_fre.eval()
        for l in range(config.FEATS_L):
            decoders_rgb[l].eval()
            decoders_fre[l].eval()

    for idx, data in enumerate(dataloader):
        imgs,labels,file_index = data['img'],data['label'],data['file_index']
        hard_mask = data['hard_mask']
        file_names_list.extend(file_index)

        logps_list_rgb = [list() for _ in range(config.FEATS_L)]
        logps_list_fre = [list() for _ in range(config.FEATS_L)]
        with torch.no_grad():
            imgs = imgs.cuda(non_blocking=True)
            labels =  labels.cuda(non_blocking=True)
            hard_mask = hard_mask.cuda(non_blocking=True).to(torch.float) 

            loss_ad = torch.tensor(0).cuda(non_blocking=True)
            loss = torch.tensor(0).cuda(non_blocking=True)
            loss_ce = torch.tensor(0).cuda(non_blocking=True)

            if config.AD==False:
                if config.MODEL_NAME == "swinb": 
                    pred = model(imgs)
                    loss_ce = F.cross_entropy(pred, labels)
                    pred_classes = (torch.max(pred, dim=1)[1])
                    pred_sigmoid = torch.nn.functional.softmax(pred,dim=1)[:,1]
                    loss = loss_ce

            if config.AD:
                fea_rgbfre,rgb,fre = model(imgs)
                for l in range(config.FEATS_L):
                    e = rgb
                    e = encoder_projector_rgb(e)

                    bs,hw,c = e.size()
                    e = e.permute(0,2,1)
                    e = e.reshape(bs,c,int(hw**0.5),int(hw**0.5))
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)

                    pos_embed = positionalencoding2d(config.MODEL.FLOW.POS_EMBED_DIM, h, w).cuda(non_blocking=True).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, config.MODEL.FLOW.POS_EMBED_DIM)
                    decoder = decoders_rgb[l]

                    if config.MODEL.FLOW.FLOW_ARCH == 'flow_model':
                        z, log_jac_det = decoder(e)  
                    else:
                        z, log_jac_det = decoder(e, [pos_embed, ])

                    logps = get_logp(dim, z, log_jac_det) 
                    logps = logps / dim      
                    loss_ad = -log_theta(logps).mean()
                    total_loss += loss_ad
                    loss_count += 1

                    logps_list_rgb[l].append(logps.reshape(bs, h, w))
                    logps_list_fre[l].append(logps.reshape(bs, h, w))

                    pred = classifier_mix(fea_rgbfre)
                    pred_classes = (torch.max(pred, dim=1)[1])
                    pred_sigmoid = torch.nn.functional.softmax(pred,dim=1)[:,1]

                    loss_ce = F.cross_entropy(pred, labels)

                loss_ad = total_loss / loss_count
                
            if distributed:
                loss_meter_ce.update(reduce_tensor(loss_ce.data).item(), labels.size(0))
            
                labels_g= gather_tensor(labels)
                labels_list.extend(labels_g.cpu().numpy().tolist())
                pred_sigmoid_g = gather_tensor(pred_sigmoid)
                pred_classifer_list.extend(pred_sigmoid_g.detach().cpu().numpy().tolist())

                if config.AD:
                    loss_meter_ad.update(reduce_tensor(loss_ad.data).item(), labels.size(0))
                    scores_rgb = convert_to_anomaly_scores(config, logps_list_rgb)
                    img_scores_rgb,_ = scores_rgb.reshape(scores_rgb.shape[0], -1).max(dim=-1) 
                    img_scores_rgb_g = gather_tensor(img_scores_rgb)
                    pred_classifer_list_ad_rgb.extend(img_scores_rgb_g.detach().cpu().numpy().tolist())

            else:
                loss_meter_ce.update(loss.item(), labels.size(0))
                labels_list.extend(labels.cpu().numpy().tolist())
                pred_classifer_list.extend(pred_sigmoid.detach().cpu().numpy().tolist())

                if config.AD:
                    scores_rgb = convert_to_anomaly_scores(config, logps_list_rgb)
                    img_scores_rgb,_ = scores_rgb.reshape(scores_rgb.shape[0], -1).max(dim=-1) 
                    pred_classifer_list_ad_rgb.extend(img_scores_rgb.detach().cpu().numpy().tolist())
                    pred_classifer_list_ad_rgb_real.extend(img_scores_rgb[labels==0].detach().cpu().numpy().tolist())
                    pred_classifer_list_ad_rgb_fake.extend(img_scores_rgb[labels==1].detach().cpu().numpy().tolist())

                    loglist_rgb_img = convert_to_image_loglogits(config, logps_list_rgb)
                    loglist_rgb_img= loglist_rgb_img.reshape(loglist_rgb_img.shape[0], -1).mean(dim=-1) 

                    pred_classifer_list_ad_rgb_real_loglist.extend(loglist_rgb_img[labels==0].detach().cpu().numpy().tolist())
                    pred_classifer_list_ad_rgb_fake_loglist.extend(loglist_rgb_img[labels==1].detach().cpu().numpy().tolist())


            acc= torch.eq(pred_classes, labels).float().mean()
            acc_meter.update(acc.item(),labels.size(0))

        del loss 

    labels_list[0]=0
    auc_ad = 0
    if config.MODEL_NAME == 'swin_ad2stream':
        auc_ad = AUC(labels_list,pred_classifer_list_ad_rgb) if config.AD else 0
 
    auc = AUC(labels_list, pred_classifer_list)

    loss_auc_dict={
        'loss_ce': loss_meter_ce.avg,
        'auc': auc,
        'auc_ad': auc_ad,
        'acc': acc_meter.avg,
    }

    if test_flag == 'image':
        return loss_auc_dict
    
    elif test_flag == 'video': 
        print('image level auc:','{:.4f}'.format(loss_auc_dict['auc']))
        print('image level acc:','{:.4f}'.format(loss_auc_dict['acc']))
        print('image level auc_ad:','{:.4f}'.format(loss_auc_dict['auc_ad']))

        video_auc , video_acc = video_metric(img_list_name, pred_classifer_list )
        return video_auc,video_acc


