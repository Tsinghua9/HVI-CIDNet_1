import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,unpaired=False,alpha=1.0,gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            output = model(input**gamma) 
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
    eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
    eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
    eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
    eval_parser.add_argument('--fivek', action='store_true', help='output FiveK dataset')
    eval_parser.add_argument('--SID', action='store_true', help='output SID dataset')

    eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
    eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
    eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

    eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
    eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--gamma', type=float, default=1.0)
    eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')
    eval_parser.add_argument('--weights', type=str, default=None, help='override weight path')

    # model structure options (must match training)
    eval_parser.add_argument('--fe_type', type=str, default='legacy', choices=['legacy', 'dual_gate'])
    eval_parser.add_argument('--lca_type', type=str, default='cab', choices=['cab', 'diem', 'waveformer'])
    eval_parser.add_argument('--use_wtconv_i', type=_str2bool, default=True)
    eval_parser.add_argument('--use_dwconv_hv', type=_str2bool, default=False)
    eval_parser.add_argument('--pre_lca_film', type=_str2bool, default=False)
    eval_parser.add_argument('--pre_lca_film_scale', type=float, default=0.1)
    eval_parser.add_argument('--pre_lca_film_bias', type=float, default=0.1)
    eval_parser.add_argument('--pre_lca_film_alpha', type=float, default=-2.197225)
    eval_parser.add_argument('--pre_lca_film_branches', type=str, default='i', choices=['i', 'hv', 'both'])
    eval_parser.add_argument('--pre_lca_film_layers', type=str, default='12', choices=['12', 'all'])
    eval_parser.add_argument('--pre_lca_film_depth_decay', type=float, default=0.7)
    eval_parser.add_argument('--glib_on_i', type=_str2bool, default=True)
    eval_parser.add_argument('--glib_on_hv', type=_str2bool, default=False)
    eval_parser.add_argument('--attn_alpha1_init', type=float, default=-2.197225)
    eval_parser.add_argument('--attn_alpha2_init', type=float, default=-2.197225)
    eval_parser.add_argument('--attn_mask_bias_scale1_init', type=float, default=1.0)
    eval_parser.add_argument('--attn_mask_bias_scale2_init', type=float, default=1.0)
    eval_parser.add_argument('--attn_mask_bias_scale1_max', type=float, default=1.0)
    eval_parser.add_argument('--attn_mask_bias_scale2_max', type=float, default=0.65)
    eval_parser.add_argument('--max_regions', type=int, default=32)

    ep = eval_parser.parse_args()


    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = '/home/zqh/code/HVI-CIDNet_1/weights/LOLv1/attn/epoch_640.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
        if alpha is None:
            alpha = ep.alpha
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.SID:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/Sony_total_dark/eval/short"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SID/'
        weight_path = './weights/SID/epoch_150.pth'
        
    elif ep.fivek:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/FiveK/test/input"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/fivek/'
        weight_path = './weights/fivek.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/custome/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights

    if ep.weights is not None:
        weight_path = ep.weights
        
    eval_net = CIDNet(
        fe_type=ep.fe_type,
        lca_type=ep.lca_type,
        use_wtconv_i=ep.use_wtconv_i,
        use_dwconv_hv=ep.use_dwconv_hv,
        pre_lca_film=ep.pre_lca_film,
        pre_lca_film_scale=ep.pre_lca_film_scale,
        pre_lca_film_bias=ep.pre_lca_film_bias,
        pre_lca_film_alpha=ep.pre_lca_film_alpha,
        pre_lca_film_branches=ep.pre_lca_film_branches,
        pre_lca_film_layers=ep.pre_lca_film_layers,
        pre_lca_film_depth_decay=ep.pre_lca_film_depth_decay,
        glib_on_i=ep.glib_on_i,
        glib_on_hv=ep.glib_on_hv,
        attn_alpha1_init=ep.attn_alpha1_init,
        attn_alpha2_init=ep.attn_alpha2_init,
        attn_mask_bias_scale1_init=ep.attn_mask_bias_scale1_init,
        attn_mask_bias_scale2_init=ep.attn_mask_bias_scale2_init,
        attn_mask_bias_scale1_max=ep.attn_mask_bias_scale1_max,
        attn_mask_bias_scale2_max=ep.attn_mask_bias_scale2_max,
        max_regions=ep.max_regions,
    ).cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)
