import time
import gc

import os
import sys
import numpy as np
import random

import logging
import logging.config
import argparse

import torch
from basic.utils import *
from basic.data import Data,make_data,InputFeatures

import warnings
warnings.filterwarnings('ignore')

import sys
if not sys.warnoptions:
    warnings.simplefilter('ignore')

# model select

#from model import opinionMining
#from model_noAspLabel import opinionMining
#from model_Pol_Asp import opinionMining
#from model_Pol_Asp_opi import opinionMining


# new ablation models
from model import opinionMining
#from model_noAttn import opinionMining
#from model_noOpinion import opinionMining
#from model_noAspLabel_withAttn import opinionMining


import warnings
warnings.filterwarnings("ignore")

print("mainLayer1Total")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed_num = 57
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def evaluate(test_set, args, model, name, output_file_path, otherOutput_path=""):
    pred_pol_results = []
    gold_pol_results = []

    pred_asp_results = []
    gold_asp_results = []

    pred_opi_results = []
    gold_opi_results = []

    dev_start = time.time()

    # set model in eval model
    model.eval()
    batch_size = args.batchSize
    test_num = len(test_set)
    total_batch = test_num // batch_size + 1
    for step in range(total_batch):
        start = step * batch_size
        end = (step + 1) * batch_size
        if end > test_num:
            end = test_num

        # make batch input data
        batch = test_set[start:end]
        if len(batch) == 0:
            continue
        all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length, char_recover, char_mask, asp_labels, opi_labels, polaritys = make_data(
            batch, args.ifgpu)

        asp_tag_seq, opi_tag_seq, polar_preds, _, _, _ = model.forward(all_input_ids, input_length, input_recover,
                                                                       all_input_mask, all_char_ids,
                                                                       char_length, char_recover, char_mask, )

        # get real label
        pred_pol_label, gold_pol_label = recover_polar_label(polar_preds[input_recover], polaritys[input_recover],
                                                             all_input_mask[input_recover])
        pred_asp_label, gold_asp_label = recover_aspect_label(asp_tag_seq[input_recover], asp_labels[input_recover],
                                                              all_input_mask[input_recover])
        pred_opi_label, gold_opi_label = recover_opinion_label(opi_tag_seq[input_recover], opi_labels[input_recover],
                                                               all_input_mask[input_recover])

        pred_pol_results += pred_pol_label
        gold_pol_results += gold_pol_label
        pred_asp_results += pred_asp_label
        gold_asp_results += gold_asp_label
        pred_opi_results += pred_opi_label
        gold_opi_results += gold_opi_label

    # build final triple reulst
    pred_triplets, asp_result, opi_result, asp_pol_result, asp_opi_result = build_triple_pair(pred_asp_results,
                                                                                              pred_opi_results,
                                                                                              pred_pol_results)

    aspPRF, opiPRF, asp_opiPRF, asp_polPRF, triPRF = fmeasure(pred_triplets, test_set)

    aspPRF, opiPRF, asp_polPRF, asp_opiPRF = measure_seperate(asp_result, opi_result, asp_pol_result, asp_opi_result,
                                                              test_set)

    output_file = open(output_file_path, "w", encoding="utf-8")
    for k in range(len(pred_triplets)):
        words = test_set[k].tokens

        gold_pairs = test_set[k].gold_relations
        relations = pred_triplets[k]

        for j in range(len(words)):
            output_file.write(words[j] + "\n")

        output_file.write("#GOLD Relations\n")
        for r in gold_pairs:
            output_file.write(
                str(r[0]) + "\t" + str(r[1]) + "\t" + str(r[2]) + "\t" + str(r[3]) + "\t" + str(r[4]) + "\n")
        output_file.write("\n")

        output_file.write("#PRED Relations\n")
        for r in relations:
            output_file.write(
                str(r[0]) + "\t" + str(r[1]) + "\t" + str(r[2]) + "\t" + str(r[3]) + "\t" + str(r[4]) + "\n")
        output_file.write("\n")
    output_file.close()

    dev_cost = time.time() - dev_start

    print("%s: time: %.2fs, speed: %.2fst/s" % (name, dev_cost, 0))
    print("TRIPLET result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(triPRF))
    print("ASPECT  result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(aspPRF))
    print("OPINION result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(opiPRF))
    print("ASPECT-OPINION  result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(asp_opiPRF))
    print("ASPECT-POLARITY result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(asp_polPRF))

    logging.info("%s: time: %.2fs, speed: %.2fst/s" % (name, dev_cost, 0))
    logging.info("TRIPLET result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(triPRF))
    logging.info("ASPECT  result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(aspPRF))
    logging.info("OPINION result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(opiPRF))
    logging.info("ASPECT-OPINION  result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(asp_opiPRF))
    logging.info("ASPECT-POLARITY result: Precision: %.4f; Recall: %.4f; F1: %.4f" % tuple(asp_polPRF))

    return aspPRF[2], asp_opiPRF[2], triPRF[2]


def main(args):
    data = Data(args)
    # make dir
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    if not os.path.exists('result/log'):
        os.makedirs('result/log')

    logging.basicConfig(filename=args.log_file + ".log", level=logging.INFO)

    #### print config ####
    #print_model_args(args)
    print(args)
    logging.info(args)

    #### read data ####
    logging.info("#### Loading dataset ####")
    datasets = torch.load(args.train_dir)
    train_set = datasets["train"]
    test_set = datasets["test"]
    valid_set = datasets["dev"]

    data.word_alphabet = datasets["word_alpha"]
    data.polar_alphabet = datasets["polar_alpha"]
    data.asp_label_alphabet = datasets["asp_alpha"]
    data.opi_label_alphabet = datasets["opi_alpha"]
    data.char_alphabet = datasets["char_alpha"]

    #### load pretrain embedding ####
    logging.info("#### Loading pretrain embeddings ####")
    pretrain = torch.load(args.pretrain_dir)
    data.pretrain_word_embedding = pretrain["preTrainEmbedding"]
    
    args.embedding_dim = pretrain["emb_dim"]

    #### defined model ####
    logging.info("#### Building model ####")
    model = opinionMining(args,data)
    if args.ifgpu:
        model = model.cuda()

    #### split param ####
    logging.info("#### Building optimizer ####")
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer ]},]

    # make optimizer
    optimizer = torch.optim.RMSprop(optimizer_grouped_parameters,
                                    lr=args.lr_rate)

    if args.mode == "Eval":
        #saved_model = torch.load(data.model_dir+"/modelFinal.model",map_location='cpu')
        #model.load_state_dict(saved_model.state_dict())
        #model = torch.load(data.model_dir+"/modelFinal.model")
        model = torch.load(args.model_dir)#"results/model/eval.model"
        evaluate(test_set, args, model, "TEST", args.eval_dir + "/test_output_eval",
                                                            args.attention_dir + "/eval")
        return

    #### train ####
    best_Score = -10000
    lr = args.lr_rate
    logging.info(args.lr_rate)
    ## start training
    for idx in range(args.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        logging.info("Epoch: %s/%s" % (idx, args.iteration))

        # adjust learning rate
        if idx > 10:
            lr = args.lr_decay * lr
            optimizer.param_groups[0]["lr"] = lr
            logging.info(lr)

        sample_loss = 0
        total_loss = 0
        right_aspect_token = 0
        whole_aspect_token = 0
        right_opinion_token = 0
        whole_opinion_token = 0
        right_polar_token = 0
        whole_polar_token = 0

        # set model in train model
        model.train()
        model.zero_grad()

        random.shuffle(train_set)
        batch_size = args.batchSize
        train_num = len(train_set)
        total_batch = train_num // batch_size + 1
        for step in range(total_batch):
            start = step * batch_size
            end = (step + 1) * batch_size
            if end > train_num:
                end = train_num

            # make batch input data
            batch = train_set[start:end]
            if len(batch) == 0:
                continue
            all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length, \
            char_recover, char_mask, asp_labels,opi_labels,polaritys = make_data(batch, args.ifgpu)

            asp_tag_seq, opi_tag_seq, polar_preds, aspect_loss, opinion_loss, polar_loss = model.nll_loss(all_input_ids,
                                                                                                          input_length,
                                                                                                          input_recover,
                                                                                                          all_input_mask,
                                                                                                          all_char_ids,
                                                                                                          char_length,
                                                                                                          char_recover,
                                                                                                          char_mask,
                                                                                                          asp_labels,
                                                                                                          opi_labels,
                                                                                                          polaritys)

            # check right number
            polarRight, polarWhole = tokenPredictCheck(polar_preds, polaritys, all_input_mask)
            aspectRight, aspectWhole = tokenPredictCheck(asp_tag_seq, asp_labels, all_input_mask)

            seq_len = opi_tag_seq.size(1)
            opinion_mask = all_input_mask.unsqueeze(1).repeat(1, seq_len, 1) * all_input_mask.unsqueeze(2).repeat(1, 1,
                                                                                                                  seq_len)
            opinionRight, opinionWhole = tokenPredictCheck(opi_tag_seq, opi_labels, opinion_mask.bool())

            # cal right and whole label number
            right_polar_token += polarRight
            whole_polar_token += polarWhole
            right_aspect_token += aspectRight
            whole_aspect_token += aspectWhole
            right_opinion_token += opinionRight
            whole_opinion_token += opinionWhole
            # cal loss
            sample_loss += aspect_loss.item() + polar_loss.item()
            total_loss += aspect_loss.item() + polar_loss.item()
            # print train info
            if step % 20 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time

                print(
                    "     Instance: %s; Time: %.2fs; loss: %.4f; ASPECT_acc: %s/%s=%.4f; OPINION_acc: %s/%s=%.4f; POLARITY_acc: %s/%s=%.4f" % (
                        step * args.batchSize, temp_cost, sample_loss, right_aspect_token, whole_aspect_token,
                        (right_aspect_token + 0.) / whole_aspect_token, right_opinion_token, whole_opinion_token,
                        (right_opinion_token + 0.) / whole_opinion_token, right_polar_token,
                        whole_polar_token, (right_polar_token + 0.) / whole_polar_token))

                logging.info(
                    "     Instance: %s; Time: %.2fs; loss: %.4f; ASPECT_acc: %s/%s=%.4f; OPINION_acc: %s/%s=%.4f; POLARITY_acc: %s/%s=%.4f" % (
                        step * args.batchSize, temp_cost, sample_loss, right_aspect_token, whole_aspect_token,
                        (right_aspect_token + 0.) / whole_aspect_token, right_opinion_token, whole_opinion_token,
                        (right_opinion_token + 0.) / whole_opinion_token, right_polar_token,
                        whole_polar_token, (right_polar_token + 0.) / whole_polar_token))

                if sample_loss > 1e9 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()  # update output show
                sample_loss = 0
            # if step % 2 ==0:
            #     loss = aspect_loss
            #     loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()
            # else:
            # loss = args.asp_lambda * aspect_loss + args.pol_lambda * polar_loss + args.opi_lambda * opinion_loss
            loss = args.asp_lambda * aspect_loss + args.opi_lambda * opinion_loss + args.pol_lambda * polar_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start

        print(
            "     Instance: %s; Time: %.2fs; loss: %.4f; ASPECT_acc: %s/%s=%.4f; OPINION_acc: %s/%s=%.4f; POLARITY_acc: %s/%s=%.10f" % (
                step * args.batchSize, temp_cost, sample_loss, right_aspect_token, whole_aspect_token,
                (right_aspect_token + 0.) / whole_aspect_token, right_opinion_token, whole_opinion_token,
                (right_opinion_token + 0.) / whole_opinion_token, right_polar_token,
                whole_polar_token, (right_polar_token + 0.) / whole_polar_token))

        logging.info(
            "     Instance: %s; Time: %.2fs; loss: %.4f; ASPECT_acc: %s/%s=%.4f; OPINION_acc: %s/%s=%.4f; POLARITY_acc: %s/%s=%.10f" % (
                step * args.batchSize, temp_cost, sample_loss, right_aspect_token, whole_aspect_token,
                (right_aspect_token + 0.) / whole_aspect_token, right_opinion_token, whole_opinion_token,
                (right_opinion_token + 0.) / whole_opinion_token, right_polar_token,
                whole_polar_token, (right_polar_token + 0.) / whole_polar_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, len(train_set) / epoch_cost, total_loss))
        logging.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, len(train_set) / epoch_cost, total_loss))
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        # dev evaluate
        asp_F, asp_opi_F, tri_F = evaluate(valid_set, args, model, "VALID", args.eval_dir + "/dev_output_" + str(idx),
                                           args.attention_dir + "/" + str(idx))

        if tri_F > best_Score:
            print("Exceed previous best F score with ASPECT F: %.4f , ASP-OPI F: %.4f , TRI F: %.4f" % (
                asp_F, asp_opi_F,tri_F))
            logging.info("Exceed previous best F score with ASPECT F: %.4f , ASP-OPI F: %.4f , TRI F: %.4f" % (
                asp_F, asp_opi_F,tri_F))
            best_Score = tri_F
            torch.save(model, args.model_dir)

        # if idx > 10 and best_Score < 30:
        #     break

        # test evaluate
        evaluate(test_set, args, model, "TEST", args.eval_dir + "/test_output_" + str(idx),
                 args.attention_dir + "/" + str(idx))

        gc.collect()

if __name__ == '__main__':
    data_name = "14lap"
    model_name = "14lap_ablation_test"
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--mode', type=str, default="Train", choices=["Train", "Eval"])
    parser.add_argument('--train_dir', type=str, default='bin_data/'+data_name)
    parser.add_argument('--pretrain_dir', type=str, default='bin_data/'+data_name+'_embs')
    parser.add_argument('--model_dir', type=str, default='result/model/'+data_name+'/'+model_name+'.model')
    parser.add_argument('--eval_dir', type=str, default='result/evalResult/'+data_name+'/'+model_name)
    parser.add_argument('--attention_dir', type=str, default='result/attention_vis/'+data_name+'/'+model_name)
    parser.add_argument('--log_file', type=str, default='result/log/'+data_name+'/'+model_name)

    #### global config ####
    parser.add_argument('--ifLowcase', type=bool, default=True)
    parser.add_argument('--ifNumZero', type=bool, default=False)
    parser.add_argument('--tagScheme', type=str, default="BIO")

    #### Embedding ####
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--char_embedding_dim', type=int, default=50)

    #### Word Rep ####
    parser.add_argument('--encoderExtractor', type=str, default="LSTM")
    parser.add_argument('--encoder_dim', type=int, default=500)
    parser.add_argument('--encoder_layer', type=int, default=1)
    parser.add_argument('--encoder_Bidirectional', type=bool, default=True)
    parser.add_argument('--useChar', type=bool, default=True)
    parser.add_argument('--charExtractor', type=str, default="LSTM")
    parser.add_argument('--char_hidden_dim', type=int, default=150)

    # aspect extractor
    parser.add_argument('--asp_label_dim', type=int, default=100)

    # opinion extractor
    parser.add_argument('--opinion_input_dim', type=int, default=400)

    # polarity prediction
    parser.add_argument('--head_dim', type=int, default=400)
    parser.add_argument('--num_head', type=int, default=8)

    # train config
    parser.add_argument('--asp_lambda', type=float, default=2.0)
    parser.add_argument('--opi_lambda', type=float, default=1.0)
    parser.add_argument('--pol_lambda', type=float, default=1.2)

    parser.add_argument('--ifgpu', type=bool, default=False)
    parser.add_argument('--iteration', type=int, default=70)
    parser.add_argument('--batchSize', type=int, default=10)
    parser.add_argument('--lr_rate', type=float, default=0.001)

    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    main(args)