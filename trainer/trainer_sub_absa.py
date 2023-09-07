import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
import json
import os
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold, formulate_gold_absa, formulate_gold_sub_absa
from utils.metric import metric, num_metric, overlap_metric, proportional_metric, binary_metric
from utils.metric_sub_absa import metric_sub_absa, proportional_metric_sub_absa, binary_metric_sub_absa 
from datetime import datetime
from pdb import set_trace as stop


class Trainer_sub_absa(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        self.tokenizer = self.args.tokenizer

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        
    def create_path(self, cur_path):
        if os.path.exists(cur_path):
            print("该文件夹已存在，请勿重复创建！")
        else:
            os.makedirs(cur_path)

    def train_model(self):
        best_f1 = 0
        no_improvement_count = 0
        if self.args.kind.lower() == 'coqe':
            path = os.path.join(self.args.output_path, 'ckpt-coqe-tuple')
            self.create_path(path)
        elif self.args.kind.lower() == 'absa':
            path = os.path.join(self.args.output_path, 'ckpt-absa-tuple')
            self.create_path(path)
        elif self.args.kind.lower() == 'tuple-logits':
            path = os.path.join(self.args.output_path, 'ckpt-tuple-logits'+ str(self.args.kl_loss_weight))
            self.create_path(path)
        elif self.args.kind.lower() == 'sub-absa':
            path = os.path.join(self.args.output_path, 'ckpt-sub-absa')
            self.create_path(path)           
        elif self.args.kind.lower() == 'absa-to-sub':
            path = os.path.join(self.args.output_path, 'ckpt-absa-to-sub')
            self.create_path(path)     

        train_loader = self.data['train']
        # result = self.eval_model(self.data.test_loader)
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            for batch_id, (input_ids, targets, _) in enumerate(tqdm(train_loader, desc=f'training on epoch {epoch}')):
                attention_mask = (input_ids != self.args.tokenizer.pad_token_id).long()
                loss, _ = self.model(input_ids, attention_mask, targets)
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print("     Instance: %d; loss: %.4f" % (batch_id * self.args.batch_size, avg_loss.avg), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            # Validation
            print("=== Epoch %d Validation ===" % epoch)
            result = self.eval_model(self.data['dev'], process='dev')
            f1 = result['f1']
            # Test
            if f1 > best_f1:
                print("Achieving Best Result on Validation Set.", flush=True)
                # torch.save({'state_dict': self.model.state_dict()}, self.args.generated_param_directory + " %s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.dataset_name, epoch, result['f1']))
                if self.args.kind.lower() == 'coqe':
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-coqe-tuple', 'best.pt'), 'wb'))
                elif  self.args.kind.lower() == 'absa':
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-absa-tuple', 'best.pt'), 'wb'))
                elif  self.args.kind.lower() == 'tuple-logits':
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-tuple-logits'+ str(self.args.kl_loss_weight), 'best.pt'), 'wb'))
                elif self.args.kind.lower() == 'sub-absa':
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-sub-absa', 'best.pt'), 'wb'))
                elif self.args.kind.lower() == 'absa-to-sub':
                    torch.save(self.model.state_dict(), open(os.path.join(self.args.output_path, 'ckpt-absa-to-sub', 'best.pt'), 'wb'))

                best_f1 = f1
                best_result_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count >= 8:
                print("No improvement in F1 for 8 consecutive epochs. Early stopping...")
                break
            # if f1 <= 0.3 and epoch >= 10:
            #     break
            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on validation set is %f achieving at epoch %d." % (best_f1, best_result_epoch), flush=True)
        self.wirte_value(os.path.join(self.args.output_path, 'ckpt-sub-absa-result'), best_f1)

        if self.args.kind.lower() == 'coqe':
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-coqe-tuple', 'best.pt'), 'rb')))
            file_name = "coqe_tuple_pred_evaluation"
        elif self.args.kind.lower() == 'absa':
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-absa-tuple', 'best.pt'), 'rb')))
            file_name = "absa_tuple_pred_evaluation"
        elif self.args.kind.lower() == 'tuple-logits':
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-tuple-logits'+ str(self.args.kl_loss_weight), 'best.pt'), 'rb')))
            file_name = "coqe_tuple_logits_pred_evaluation"
        elif self.args.kind.lower() == 'sub-absa':
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-sub-absa', 'best.pt'), 'rb')))
            file_name = "sub_absa_pred_evaluation"
        elif self.args.kind.lower() == 'absa-to-sub':
            self.model.load_state_dict(torch.load(open(os.path.join(self.args.output_path, 'ckpt-absa-to-sub', 'best.pt'), 'rb')))
            file_name = "absa_to_sub_pred_evaluation"
               
        # Test
        # print("=== Final Test ===", flush=True)
        # result = self.eval_model(self.data['test'], 'test')
        # result = self.eval_model(self.data['test'], process='test')
        # # test_f1 = result['f1'] 

        # with open(os.path.join(self.args.output_path, file_name),"a") as f:
        #     print("================   Final Result   ====================", file=f)
        #     print("-------------- Exact Result --------------------------", file=f)
        #     print(result[0], file=f)
        #     print("-------------- Proportional Result --------------------------", file=f)
        #     print(result[1], file=f)
        #     print("-------------- Binary Result --------------------------", file=f)
        #     print(result[2], file=f)
        #     print("+++++++++++++++++++++++++++++++++++++", file=f)
        #     print("End time is {}".format(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")), file=f)


    def eval_model(self, eval_loader, process):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold = {}, {}
        pred_texts = {}

        def get_text(input_ids, start_index, end_index):
            tokenizer = self.args.tokenizer
            text = tokenizer.decode(input_ids[start_index: end_index])
            return text.strip()

        whole_input_ids = []
        with torch.no_grad():
            batch_size = self.args.batch_size
            for batch_id, (input_ids, target, info) in enumerate(tqdm(eval_loader, f'evaluation')): # info 为每个bsz内的索引序号
                # stop()
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                whole_input_ids += input_ids.tolist()
                gold.update(formulate_gold_sub_absa(target, info))
                # print(target)
                gen_triples = self.model.gen_triples_sub_absa(input_ids, attention_mask, info)
                prediction.update(gen_triples)
    
        if process=='dev':
            print("run dev", process)
            return metric_sub_absa(prediction, gold)

    def wirte_value(self, file_path, best_f1):
        with open(file_path, 'a') as f:
            print("================   Best Result   ====================", file=f)
            print("-------------- Exact Result --------------------------", file=f)
            print(best_f1, file=f)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
