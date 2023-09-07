import argparse, os, torch
import random
from datetime import datetime
import numpy as np
from utils.data import load_data, build_collate_fn
from utils.data_sub_absa import load_data_sub_absa, build_collate_fn_sub_absa
from trainer.trainer import Trainer
from trainer.trainer_sub_absa import Trainer_sub_absa
from models.setpred4RE import SetPred4RE
from models.setpred4RE_sub_absa import SetPred4RE_sub_absa
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pdb import set_trace as stop


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='')
    parser.add_argument('--output_path', default='log/')
    parser.add_argument('--bert_directory', type=str, default="./bert_base_uncased/")
    parser.add_argument('--model_name', type=str, default="Set-Prediction-Networks")
    parser.add_argument('--num_generated_triples', type=int, default=10)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    parser.add_argument('--na_rel_coef', type=float, default=1)
    parser.add_argument('--rel_loss_weight', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--decoder_lr', type=float, default=2e-5)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--n_best_size', type=int, default=5)
    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--max_span_length', type=int, default=10)
    #  add
    parser.add_argument('--kind', type=str, default='coqe') # absa or coqe or tuple-logits or sub-absa
    parser.add_argument('--kl_loss_weight', type=float, default=100.0, help='the weight of kl loss to calculate distillation')
    parser.add_argument('--pt_file', type=str, help='the path of transfer learning')
    args = parser.parse_args()
    # args.output_path = os.path.join(args.output_path, datetime.today().strftime("%Y-%m-%d-%H-%M-%S"))
    args.output_path = os.path.join(args.output_path, args.model_name +"_"+ str(args.random_seed))

    tokenizer = AutoTokenizer.from_pretrained(args.bert_directory)
    args.tokenizer = tokenizer

    if os.path.exists(args.output_path):
        print("该文件夹已存在，请勿重复创建！")
    else:
        os.makedirs(args.output_path)

        
    with open(os.path.join(args.output_path, str(args.kind) +"_params.txt"),"w") as fw:
        print("Start time is {}".format(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")), file=fw)
        for arg in vars(args):
            print(arg, ":",  getattr(args, arg)) # show directly
            print(arg, ":",  getattr(args, arg), file=fw) # write to params_txt

    set_seed(args.random_seed)

    if args.kind == 'sub-absa': # 做<sub,obj,asp>三元组抽取任务
        collate_sub_fn = build_collate_fn_sub_absa(args)
        data = {
            'train': DataLoader(load_data_sub_absa(args, 'camera_train_three_combined_A'), args.batch_size, True, collate_fn=collate_sub_fn),
            'dev': DataLoader(load_data_sub_absa(args, 'camera_dev_three_combined_A'), args.batch_size, False, collate_fn=collate_sub_fn),
        }
        model = SetPred4RE_sub_absa(args, 2).to(args.device)
        trainer = Trainer_sub_absa(model, data, args)
        trainer.train_model()


    if args.kind =="main": # 将<sub,obj,asp>向<asp,op,type>迁移之后的结果，再向五元组迁移
        collate_fn = build_collate_fn(args)
        data = {
            'train': DataLoader(load_data(args, 'train'), args.batch_size, True, collate_fn=collate_fn),
            'dev': DataLoader(load_data(args, 'dev'), args.batch_size, False, collate_fn=collate_fn),
            'test': DataLoader(load_data(args, 'test'), args.batch_size, False, collate_fn=collate_fn),
        }
        model = SetPred4RE(args, 5).to(args.device)
        model.load_state_dict(torch.load(args.pt_file), strict=False) # 加载第一阶段训练好的.pt文件
        trainer = Trainer(model, data, args)
        trainer.train_model()

