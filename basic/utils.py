import numpy as np
import torch

#### target token level precision ####
def tokenPredictCheck(targetPredict, batch_target_label, mask):
    pred = targetPredict.cpu().data.numpy()
    gold = batch_target_label.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token

def recover_aspect_label(targetPredict, all_labels, all_input_mask):
    pred_variable = targetPredict
    gold_variable = all_labels
    mask_variable = all_input_mask
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [pred_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [gold_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def recover_opinion_label(targetPredict, all_labels, all_input_mask):
    pred_variable = targetPredict
    gold_variable = all_labels

    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)

    mask = all_input_mask.unsqueeze(1).repeat(1,seq_len,1) * all_input_mask.unsqueeze(2).repeat(1,1,seq_len)
    mask = mask.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred_cur_bz = []
        gold_cur_bz = []
        for idy in range(seq_len):
            pred = [pred_tag[idx][idy][idz] - 1 for idz in range(seq_len) if mask[idx][idy][idz] != 0]
            gold = [gold_tag[idx][idy][idz] - 1 for idz in range(seq_len) if mask[idx][idy][idz] != 0]
            assert (len(pred) == len(gold))
            pred_cur_bz.append(pred)
            gold_cur_bz.append(gold)
        pred_label.append(pred_cur_bz)
        gold_label.append(gold_cur_bz)
    return pred_label, gold_label

def recover_polar_label(targetPredict, all_labels, all_input_mask):
    pred_variable = targetPredict
    gold_variable = all_labels
    mask_variable = all_input_mask
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [pred_tag[idx][idy]  for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [gold_tag[idx][idy]  for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def get_entity_boundary(preds):
    # get cur instance aspect entity list
    entityList = []
    for idy in range(len(preds)):
        if preds[idy] == 1:
            if idy == len(preds) - 1:
                entityList.append([idy, idy + 1])
            else:
                for k in range(idy + 1, len(preds)):
                    if preds[k] != preds[idy] + 1:
                        entityList.append([idy, k])
                        break
                    elif preds[k] == preds[idy] + 1 and k == len(preds) - 1:
                        entityList.append([idy, k + 1])
                        break
    return entityList

def build_pair(aspect_results, opinion_results):

    asp_result = []
    opi_result = []
    asp_opi_result = []

    for idx in range(len(aspect_results)):
        cur_instance_opi_result = []
        cur_instance_asp_opi_result = []

        entity_list = get_entity_boundary(aspect_results[idx])
        asp_result.append(entity_list)

        # use aspect entity list to find mapped opinion entity results
        for asp_start,asp_end in entity_list:
            # 使用aspect第一个单词对应的opinion tag 判断结果
            opi_list = get_entity_boundary(opinion_results[idx][asp_start])
            for opi_start,opi_end in opi_list:
                cur_instance_asp_opi_result.append([opi_start, opi_end, asp_start, asp_end])

        asp_opi_result.append(cur_instance_asp_opi_result)

        # 单独的opinion结果提取
        for idy in range(len(aspect_results[idx])):
            opi_res = get_entity_boundary(opinion_results[idx][idy])
            cur_instance_opi_result += opi_res
        opi_result.append(cur_instance_opi_result)

    return asp_result,opi_result,asp_opi_result



def build_triple_pair(aspect_results,opinion_results,polar_results):
    total_result = []
    asp_result = []
    opi_result = []
    asp_opi_result = []
    asp_pol_result = []

    for idx in range(len(aspect_results)):
        cur_instance_result = []
        cur_instance_asp_pol_result = []
        cur_instance_asp_opi_result = []

        entity_list = get_entity_boundary(aspect_results[idx])
        asp_result.append(entity_list)

        # use aspect entity list to find mapped opinion entity results
        for asp_start,asp_end in entity_list:
            # aspect polarity result construct
            polar = polar_results[idx][asp_start]
            if polar != 0:
                cur_instance_asp_pol_result.append([asp_start, asp_end, polar])

            opi_list = get_entity_boundary(opinion_results[idx][asp_start])
            for opi_start,opi_end in opi_list:
                cur_instance_asp_opi_result.append([opi_start, opi_end, asp_start, asp_end])
                if polar !=0:
                    cur_instance_result.append([opi_start, opi_end, asp_start,asp_end, polar])

        total_result.append(cur_instance_result)
        asp_pol_result.append(cur_instance_asp_pol_result)
        asp_opi_result.append(cur_instance_asp_opi_result)
        # 单独的opinion结果提取
        cur_instance_opi_result = []
        for idy in range(len(aspect_results[idx])):
            opi_res = get_entity_boundary(opinion_results[idx][idy])
            cur_instance_opi_result += opi_res
        opi_result.append(cur_instance_opi_result)

    return total_result,asp_result,opi_result,asp_pol_result,asp_opi_result



def fmeasure(preds, gold):
    gold_asp = 0
    pred_asp = 0
    asp_correct = 0

    gold_opi = 0
    pred_opi = 0
    opi_correct = 0

    gold_asp_pol = 0
    pred_asp_pol = 0
    asp_pol_correct = 0

    gold_asp_opi = 0
    pred_asp_opi = 0
    asp_opi_correct = 0

    gold_tri = 0
    pred_tri = 0
    triple_correct = 0
    polar_map = {'NEG':1,'NEU':2,"POS":3}
    for idx in range(len(preds)):
        standard = gold[idx].gold_relations
        pred = preds[idx]

        gold_tri += len(standard)
        pred_tri += len(pred)

        # build different gold results for calc correct num
        gold_aspects = set((asp_s,asp_e) for _,_,asp_s,asp_e,_ in standard)
        gold_opinions = set((opi_s,opi_e) for opi_s,opi_e,_,_,_ in standard)
        gold_asp_opis = set((opi_s,opi_e,asp_s,asp_e) for opi_s,opi_e,asp_s,asp_e,_ in standard)
        gold_asp_pols = set((asp_s,asp_e,polar_map[pol]) for _,_,asp_s,asp_e,pol in standard)
        gold_tris = set((opi_s,opi_e,asp_s,asp_e,polar_map[pol]) for opi_s,opi_e,asp_s,asp_e,pol in standard)

        gold_asp += len(gold_aspects)
        gold_opi += len(gold_opinions)
        gold_asp_opi += len(gold_asp_opis)
        gold_asp_pol += len(gold_asp_pols)

        pred_aspects = set((asp_s,asp_e) for _,_,asp_s,asp_e,_ in pred)
        pred_opinions = set((opi_s,opi_e) for opi_s,opi_e,_,_,_ in pred)
        pred_asp_opis = set((opi_s,opi_e,asp_s,asp_e) for opi_s,opi_e,asp_s,asp_e,_ in pred)
        pred_asp_pols = set((asp_s,asp_e,pol) for _,_,asp_s,asp_e,pol in pred)
        pred_tris = set((opi_s,opi_e,asp_s,asp_e,pol) for opi_s,opi_e,asp_s,asp_e,pol in pred)

        pred_asp += len(pred_aspects)
        pred_opi += len(pred_opinions)
        pred_asp_opi += len(pred_asp_opis)
        pred_asp_pol += len(pred_asp_pols)

        for r in gold_aspects:
            if r in pred_aspects:
                asp_correct += 1

        for r in gold_opinions:
            if r in pred_opinions:
                opi_correct += 1

        for r in gold_asp_opis:
            if r in pred_asp_opis:
                asp_opi_correct += 1

        for r in gold_asp_pols:
            if r in pred_asp_pols:
                asp_pol_correct += 1

        for r in gold_tris:
            if r in pred_tris:
                triple_correct += 1

    # aspect P R F
    aspPRF = calc_PRF(asp_correct,pred_asp,gold_asp)
    opiPRF = calc_PRF(opi_correct,pred_opi,gold_opi)
    asp_opiPRF =  calc_PRF(asp_opi_correct,pred_asp_opi,gold_asp_opi)
    asp_polPRF = calc_PRF(asp_pol_correct, pred_asp_pol, gold_asp_pol)

    triPRF = calc_PRF(triple_correct,pred_tri,gold_tri)

    return aspPRF,opiPRF,asp_opiPRF,asp_polPRF,triPRF

def calc_PRF(correct,pred,gold):
    precision = float(correct) / (pred + 1e-6)
    recall = float(correct) / (gold + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return [precision, recall, f1]


def measure_seperate(asp_result,opi_result,asp_pol_result,asp_opi_result,gold):
    gold_asp = 0
    pred_asp = 0
    asp_correct = 0

    gold_opi = 0
    pred_opi = 0
    opi_correct = 0

    gold_asp_pol = 0
    pred_asp_pol = 0
    asp_pol_correct = 0

    gold_asp_opi = 0
    pred_asp_opi = 0
    asp_opi_correct = 0

    polar_map = {'NEG':1,'NEU':2,"POS":3}
    for idx in range(len(gold)):
        standard = gold[idx].gold_relations

        # build different gold results for calc correct num
        gold_aspects = set((asp_s,asp_e) for _,_,asp_s,asp_e,_ in standard)
        gold_opinions = set((opi_s,opi_e) for opi_s,opi_e,_,_,_ in standard)
        gold_asp_pols = set((asp_s,asp_e,polar_map[pol]) for _,_,asp_s,asp_e,pol in standard)
        gold_asp_opis = set((opi_s, opi_e, asp_s, asp_e) for opi_s, opi_e, asp_s, asp_e, _ in standard)

        gold_asp += len(gold_aspects)
        gold_opi += len(gold_opinions)
        gold_asp_pol += len(gold_asp_pols)
        gold_asp_opi += len(gold_asp_opis)

        pred_aspects = set((asp_s,asp_e) for asp_s,asp_e in asp_result[idx])
        pred_opinions = set((opi_s,opi_e) for opi_s,opi_e in opi_result[idx])
        pred_asp_pols = set((asp_s,asp_e,pol) for asp_s,asp_e,pol in asp_pol_result[idx])
        pred_asp_opis = set((opi_s, opi_e, asp_s, asp_e) for opi_s, opi_e, asp_s, asp_e in asp_opi_result[idx])

        pred_asp += len(pred_aspects)
        pred_opi += len(pred_opinions)
        pred_asp_pol += len(pred_asp_pols)
        pred_asp_opi += len(pred_asp_opis)

        for r in gold_aspects:
            if r in pred_aspects:
                asp_correct += 1

        for r in gold_opinions:
            if r in pred_opinions:
                opi_correct += 1

        for r in gold_asp_pols:
            if r in pred_asp_pols:
                asp_pol_correct += 1

        for r in gold_asp_opis:
            if r in pred_asp_opis:
                asp_opi_correct += 1

    # aspect P R F
    aspPRF = calc_PRF(asp_correct,pred_asp,gold_asp)
    opiPRF = calc_PRF(opi_correct,pred_opi,gold_opi)
    asp_polPRF = calc_PRF(asp_pol_correct, pred_asp_pol, gold_asp_pol)
    asp_opiPRF = calc_PRF(asp_opi_correct,pred_asp_opi,gold_asp_opi)

    return aspPRF,opiPRF,asp_polPRF,asp_opiPRF


# old for asp - opi measure
# def measure_seperate(asp_result,opi_result,asp_opi_result,gold):
#     gold_asp = 0
#     pred_asp = 0
#     asp_correct = 0
#
#     gold_opi = 0
#     pred_opi = 0
#     opi_correct = 0
#
#     gold_asp_opi = 0
#     pred_asp_opi = 0
#     asp_opi_correct = 0
#
#     polar_map = {'NEG':1,'NEU':2,"POS":3}
#     for idx in range(len(gold)):
#         standard = gold[idx].gold_relations
#
#         # build different gold results for calc correct num
#         gold_aspects = set((asp_s,asp_e) for _,_,asp_s,asp_e,_ in standard)
#         gold_opinions = set((opi_s,opi_e) for opi_s,opi_e,_,_,_ in standard)
#         gold_asp_opis = set((opi_s, opi_e, asp_s, asp_e) for opi_s, opi_e, asp_s, asp_e, _ in standard)
#
#         gold_asp += len(gold_aspects)
#         gold_opi += len(gold_opinions)
#         gold_asp_opi += len(gold_asp_opis)
#
#         pred_aspects = set((asp_s,asp_e) for asp_s,asp_e in asp_result[idx])
#         pred_opinions = set((opi_s,opi_e) for opi_s,opi_e in opi_result[idx])
#         pred_asp_opis = set((opi_s, opi_e, asp_s, asp_e) for opi_s, opi_e, asp_s, asp_e in asp_opi_result[idx])
#
#         pred_asp += len(pred_aspects)
#         pred_opi += len(pred_opinions)
#         pred_asp_opi += len(pred_asp_opis)
#
#         for r in gold_aspects:
#             if r in pred_aspects:
#                 asp_correct += 1
#
#         for r in gold_opinions:
#             if r in pred_opinions:
#                 opi_correct += 1
#
#         for r in gold_asp_opis:
#             if r in pred_asp_opis:
#                 asp_opi_correct += 1
#
#     # aspect P R F
#     aspPRF = calc_PRF(asp_correct,pred_asp,gold_asp)
#     opiPRF = calc_PRF(opi_correct,pred_opi,gold_opi)
#     asp_opiPRF = calc_PRF(asp_opi_correct,pred_asp_opi,gold_asp_opi)
#
#     return aspPRF,opiPRF,asp_opiPRF
