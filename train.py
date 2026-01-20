import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option().parse_args()

if (not opt.use_region_prior) and (opt.prior_label_dir is not None or opt.prior_mode != "gate"):
    print("[warn] prior_mode/prior_label_dir is set but --use_region_prior is False; region prior will NOT be used.")

def _get_prior_alpha(model, prior_mode: str):
    base = model.module if hasattr(model, "module") else model
    if prior_mode == "film":
        alpha_param = getattr(base, "region_film").alpha
    elif prior_mode == "attn":
        alpha_param = getattr(base, "region_attn").alpha
    else:
        alpha_param = getattr(base, "structure_gate").alpha
    alpha_raw = float(alpha_param.detach().cpu().item())
    alpha_sigmoid = float(torch.sigmoid(alpha_param.detach()).cpu().item())
    return alpha_raw, alpha_sigmoid

def _get_prior_alpha2(model, prior_mode: str):
    base = model.module if hasattr(model, "module") else model
    if prior_mode == "film":
        alpha_param = getattr(base, "region_film2").alpha
    elif prior_mode == "attn":
        alpha_param = getattr(base, "region_attn2").alpha
    else:
        alpha_param = getattr(base, "structure_gate2").alpha
    alpha_raw = float(alpha_param.detach().cpu().item())
    alpha_sigmoid = float(torch.sigmoid(alpha_param.detach()).cpu().item())
    return alpha_raw, alpha_sigmoid

def _get_prior_policy_stats(model):
    base = model.module if hasattr(model, "module") else model
    policy = getattr(base, "region_policy", None)
    if policy is None:
        return None
    stats = {}
    for key in ["last_gamma_mean", "last_gamma_std", "last_beta_mean", "last_beta_std"]:
        if hasattr(policy, key):
            stats[key] = getattr(policy, key)
    return stats or None

def _get_prior_film_stats(model):
    base = model.module if hasattr(model, "module") else model
    film = getattr(base, "region_film", None)
    if film is None:
        return None
    stats = {}
    for key in [
        "last_a",
        "last_gamma_dev_mean",
        "last_gamma_dev_std",
        "last_beta_used_mean",
        "last_beta_used_std",
        "last_delta_ratio",
    ]:
        if hasattr(film, key):
            stats[key] = getattr(film, key)
    return stats or None

def _get_prior_film2_stats(model):
    base = model.module if hasattr(model, "module") else model
    film = getattr(base, "region_film2", None)
    if film is None:
        return None
    stats = {}
    for key in ["last_a", "last_delta_ratio"]:
        if hasattr(film, key):
            stats[key] = getattr(film, key)
    return stats or None

def _get_prior_attn_stats(model):
    base = model.module if hasattr(model, "module") else model
    attn = getattr(base, "region_attn", None)
    if attn is None:
        return None
    stats = {}
    for key in ["last_a", "last_delta_ratio", "mask_bias_scale"]:
        if hasattr(attn, key):
            val = getattr(attn, key)
            if torch.is_tensor(val):
                stats[key] = float(val.detach().cpu().item())
            else:
                stats[key] = val
    return stats or None

def _get_prior_attn2_stats(model):
    base = model.module if hasattr(model, "module") else model
    attn = getattr(base, "region_attn2", None)
    if attn is None:
        return None
    stats = {}
    for key in ["last_a", "last_delta_ratio", "mask_bias_scale"]:
        if hasattr(attn, key):
            val = getattr(attn, key)
            if torch.is_tensor(val):
                stats[key] = float(val.detach().cpu().item())
            else:
                stats[key] = val
    return stats or None

def seed_torch():
    # seed = random.randint(1, 1000000)
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        if opt.use_region_prior:
            if len(batch) != 5:
                raise RuntimeError(
                    f"--use_region_prior True expects dataset to return 5 items "
                    f"(im1, im2, index_map, path1, path2), got {len(batch)}. "
                    f"Dataset={opt.dataset}, prior_label_dir={opt.prior_label_dir}, prior_mode={opt.prior_mode}"
                )
            im1, im2, index_map, path1, path2 = batch
        else:
            im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        if opt.use_region_prior:
            index_map = index_map.cuda(non_blocking=True)
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            if opt.use_region_prior:
                output_rgb = model(im1 ** gamma, index_map=index_map, prior_mode=opt.prior_mode)
            else:
                output_rgb = model(im1 ** gamma)
        else:
            if opt.use_region_prior:
                output_rgb = model(im1, index_map=index_map, prior_mode=opt.prior_mode)
            else:
                output_rgb = model(im1)
            
        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(
            opt.data_train_lol_v1,
            size=opt.cropSize,
            label_dir=opt.prior_label_dir,
            use_prior=opt.use_region_prior,
            max_regions=opt.max_regions,
        )
        test_set = get_eval_set(opt.data_val_lol_v1)
        
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)

    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(
            opt.data_train_lolv2_real,
            size=opt.cropSize,
            label_dir=opt.prior_label_dir,
            use_prior=opt.use_region_prior,
            max_regions=opt.max_regions,
        )
        test_set = get_eval_set(opt.data_val_lolv2_real)
        
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(
            opt.data_train_lolv2_syn,
            size=opt.cropSize,
            label_dir=opt.prior_label_dir,
            use_prior=opt.use_region_prior,
            max_regions=opt.max_regions,
        )
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise Exception("should choose a dataset")
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet(use_wtconv=opt.use_wtconv).cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    # Apply attn init params from options for reproducible tuning (only for fresh training).
    base = model.module if hasattr(model, "module") else model
    if opt.use_region_prior and opt.prior_mode == "attn":
        # Always apply mb clamp settings (these are plain attributes, not saved in state_dict).
        if hasattr(base, "region_attn"):
            base.region_attn.mask_bias_scale_max = None if float(opt.attn_mask_bias_scale1_max) < 0 else float(opt.attn_mask_bias_scale1_max)
        if hasattr(base, "region_attn2"):
            base.region_attn2.mask_bias_scale_max = None if float(opt.attn_mask_bias_scale2_max) < 0 else float(opt.attn_mask_bias_scale2_max)

        # Apply init params only for fresh training (avoid overwriting a resumed checkpoint).
        if opt.start_epoch == 0:
            if hasattr(base, "region_attn"):
                base.region_attn.alpha.data.fill_(float(opt.attn_alpha1_init))
                base.region_attn.mask_bias_scale.data.fill_(float(opt.attn_mask_bias_scale1_init))
            if hasattr(base, "region_attn2"):
                base.region_attn2.alpha.data.fill_(float(opt.attn_alpha2_init))
                base.region_attn2.mask_bias_scale.data.fill_(float(opt.attn_mask_bias_scale2_init))
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 

    if not os.path.exists("./results/training"):
        os.makedirs("./results/training",exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_tag = str(opt.dataset).replace("/", "_").replace("\\", "_").replace(" ", "_")
    log_path = f"./results/training/metrics_{dataset_tag}_{now}.md"
    with open(log_path, "w") as f:
        f.write(f"dataset: {opt.dataset}\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write(f"use_region_prior: {opt.use_region_prior}\n")
        f.write(f"prior_mode: {opt.prior_mode}\n")
        f.write(f"use_wtconv: {opt.use_wtconv}\n")
        f.write(f"prior_label_dir: {opt.prior_label_dir}\n")
        f.write(f"max_regions: {opt.max_regions}\n")
        f.write(f"attn_alpha1_init: {opt.attn_alpha1_init}\n")
        f.write(f"attn_alpha2_init: {opt.attn_alpha2_init}\n")
        f.write(f"attn_mask_bias_scale1_init: {opt.attn_mask_bias_scale1_init}\n")
        f.write(f"attn_mask_bias_scale2_init: {opt.attn_mask_bias_scale2_init}\n")
        f.write(f"attn_mask_bias_scale1_max: {opt.attn_mask_bias_scale1_max}\n")
        f.write(f"attn_mask_bias_scale2_max: {opt.attn_mask_bias_scale2_max}\n")
        # f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        # f.write("|----------------------|----------------------|----------------------|----------------------|\n")
        f.write("| Epoch | PSNR | SSIM | LPIPS | mode | alpha1 | a1 | d1 | eff1 | mb1 | alpha2 | a2 | d2 | eff2 | mb2 |\n")
        f.write("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")


    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        if opt.use_region_prior:
            alpha_raw, alpha_sigmoid = _get_prior_alpha(model, opt.prior_mode)
            alpha2_raw, alpha2_sigmoid = _get_prior_alpha2(model, opt.prior_mode)
            msg = (f"[prior] mode={opt.prior_mode}"
                   f" alpha1={alpha_raw:.6f} a1={alpha_sigmoid:.6f}"
                   f" | alpha2={alpha2_raw:.6f} a2={alpha2_sigmoid:.6f}")
            if opt.prior_mode == "film":
                stats = _get_prior_film_stats(model)
                if stats:
                    msg += (f" | film1:delta_ratio={stats.get('last_delta_ratio', 0.0):.6f}"
                            f" gamma1_used-1(std)={stats.get('last_gamma_dev_std', 0.0):.6f}"
                            f" beta1_used(std)={stats.get('last_beta_used_std', 0.0):.6f}")
                else:
                    stats = _get_prior_policy_stats(model)
                    if stats:
                        msg += (f" | gamma_raw(mean,std)=({stats.get('last_gamma_mean', 0.0):.6f},{stats.get('last_gamma_std', 0.0):.6f})"
                                f" beta_raw(mean,std)=({stats.get('last_beta_mean', 0.0):.6f},{stats.get('last_beta_std', 0.0):.6f})")
                stats2 = _get_prior_film2_stats(model)
                if stats2:
                    msg += f" | film2:delta_ratio={stats2.get('last_delta_ratio', 0.0):.6f}"
            elif opt.prior_mode == "attn":
                stats = _get_prior_attn_stats(model)
                if stats:
                    a1 = alpha_sigmoid
                    d1 = float(stats.get("last_delta_ratio", 0.0))
                    msg += (f" | attn1:delta_ratio={d1:.6f} eff1={a1*d1:.6f} mb1={float(stats.get('mask_bias_scale', 0.0)):.3f}")
                stats2 = _get_prior_attn2_stats(model)
                if stats2:
                    a2 = alpha2_sigmoid
                    d2 = float(stats2.get("last_delta_ratio", 0.0))
                    msg += f" | attn2:delta_ratio={d2:.6f} eff2={a2*d2:.6f} mb2={float(stats2.get('mask_bias_scale', 0.0)):.3f}"
            print(msg)
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'
            is_lol_v1 = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)

            # with open(log_path, "a") as f:
            #     f.write(f"| {epoch} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")
            mode_s = "-" if not opt.use_region_prior else opt.prior_mode

            alpha1_s = a1_s = d1_s = eff1_s = mb1_s = "-"
            alpha2_s = a2_s = d2_s = eff2_s = mb2_s = "-"

            if opt.use_region_prior and opt.prior_mode == "attn":
                stats1 = _get_prior_attn_stats(model) or {}
                stats2 = _get_prior_attn2_stats(model) or {}
                alpha_raw, alpha_sigmoid = _get_prior_alpha(model, opt.prior_mode)
                alpha2_raw, alpha2_sigmoid = _get_prior_alpha2(model, opt.prior_mode)

                d1 = float(stats1.get("last_delta_ratio", 0.0) or 0.0)
                d2 = float(stats2.get("last_delta_ratio", 0.0) or 0.0)
                mb1 = float(stats1.get("mask_bias_scale", 0.0) or 0.0)
                mb2 = float(stats2.get("mask_bias_scale", 0.0) or 0.0)

                alpha1_s = f"{alpha_raw:.6f}"
                a1_s     = f"{alpha_sigmoid:.6f}"
                d1_s     = f"{d1:.6f}"
                eff1_s   = f"{(alpha_sigmoid*d1):.6f}"
                mb1_s    = f"{mb1:.3f}"

                alpha2_s = f"{alpha2_raw:.6f}"
                a2_s     = f"{alpha2_sigmoid:.6f}"
                d2_s     = f"{d2:.6f}"
                eff2_s   = f"{(alpha2_sigmoid*d2):.6f}"
                mb2_s    = f"{mb2:.3f}"

            with open(log_path, "a") as f:
                f.write(
                    f"| {epoch} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} | {mode_s} | "
                    f"{alpha1_s} | {a1_s} | {d1_s} | {eff1_s} | {mb1_s} | "
                    f"{alpha2_s} | {a2_s} | {d2_s} | {eff2_s} | {mb2_s} |\n"
                )

        torch.cuda.empty_cache()
