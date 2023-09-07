from pdb import set_trace as stop

def metric_two(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(
            # ele.pred_rel,
            ele.sub_start_index, ele.sub_end_index, 
            ele.obj_start_index, ele.obj_end_index, 
            # ele.aspect_start_index, ele.aspect_end_index, 
            # ele.opinion_start_index, ele.opinion_end_index, 
        ) for ele in pred[sent_idx]]))
        pred_num += len(prediction)

        for ele in prediction:
            if len(gold[sent_idx]) > 0:
                gold_two = []
                for t in gold[sent_idx]:
                    ele1 = t[1:5]
                    gold_two.append(ele1)
            elif len(gold[sent_idx]) == 0:
                gold_two = gold[sent_idx]

            if ele in gold_two:  # eg: (3, 3, 5, 32, 34, 24, 27, 28, 29)
                right_num += 1
                pred_correct_num += 1


    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    precision = precision * 100
    recall = recall * 100
    f_measure = f_measure * 100
    print("+++++++++++++++++++++++Two Results+++++++++++++++++++++++++++++")
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num,  " entity_right_num = ", ent_num)
    print("two precision = ", precision, " two recall = ", recall, " two f1_value = ", f_measure)
    return {"two precision": precision, " two recall": recall, " two f1": f_measure}

def metric_three(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(ele.pred_rel,
            ele.aspect_start_index, ele.aspect_end_index, 
            ele.opinion_start_index, ele.opinion_end_index, 
        ) for ele in pred[sent_idx]]))
        pred_num += len(prediction)

        for ele in prediction:
            if len(gold[sent_idx]) > 0:
                three_gold = []
                for t in gold[sent_idx]:
                    ele1 = t[:1] + t[5:]
                    three_gold.append(ele1)
            elif len(gold[sent_idx]) == 0:
                three_gold = gold[sent_idx]

            if ele in three_gold:  # eg: (3, 3, 5, 32, 34, 24, 27, 28, 29) type,sub,obj,asp,op
                right_num += 1
                pred_correct_num += 1

    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    precision = precision * 100
    recall = recall * 100
    f_measure = f_measure * 100
    print("+++++++++++++++++++++++Three Results+++++++++++++++++++++++++++++")
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num)
    print("three precision = ", precision, " three recall = ", recall, " three f1_value = ", f_measure)
    return {"three precision": precision, " three recall": recall, " three f1": f_measure}

def metric_sub_absa(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(
            ele.sub_start_index, ele.sub_end_index, 
            ele.obj_start_index, ele.obj_end_index, 
            ele.aspect_start_index, ele.aspect_end_index, 
        ) for ele in pred[sent_idx]]))
        # stop()
        pred_num += len(prediction)
    
        for ele in prediction:
            if ele in gold[sent_idx]:  # eg: (3, 3, 5, 32, 34, 24, 27, 28, 29)
                right_num += 1
                pred_correct_num += 1

    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)


    precision = precision * 100
    recall = recall * 100
    f_measure = f_measure * 100

    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    return {"precision": precision, "recall": recall, "f1": f_measure}

def tuple_to_five_ele(ele_tuple):
    ele_list = list(ele_tuple)
    sub_pred, obj_pred, aspect_pred = (ele_list[0],ele_list[1]),(ele_list[2],ele_list[3]), (ele_list[4],ele_list[5])
    return sub_pred, obj_pred, aspect_pred

def convert_tuple_to_set(tuple1):
    rel_set = set()
    for i in range(tuple1[0], tuple1[1]+1): # 左闭右闭,闭需要+1
        rel_set.add(i)
    return rel_set

def binary_metric_sub_absa(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(
            ele.sub_start_index, ele.sub_end_index, 
            ele.obj_start_index, ele.obj_end_index, 
            ele.aspect_start_index, ele.aspect_end_index, 
            ele.opinion_start_index, ele.opinion_end_index, 
        ) for ele in pred[sent_idx]]))
        pred_num += len(prediction)

        for ele_pred in prediction:
            for ele_gold in gold[sent_idx]:
                # ele_pred = (3, 3, 5, 32, 34, 24, 27, 28, 29)
                # ele_gold = (3, 3, 4, 32, 34, 24, 26, 28, 29)
                sub_pred, obj_pred, aspect_pred = tuple_to_five_ele(ele_pred)
                sub_gold, obj_gold, aspect_gold = tuple_to_five_ele(ele_gold)

                sub_pred_set, obj_pred_set, asp_pred_set = convert_tuple_to_set(sub_pred), convert_tuple_to_set(obj_pred), convert_tuple_to_set(aspect_pred)
                sub_gold_set, obj_gold_set, asp_gold_set = convert_tuple_to_set(sub_gold), convert_tuple_to_set(obj_gold), convert_tuple_to_set(aspect_gold)

                if (sub_pred_set&sub_gold_set) and (obj_pred_set & obj_gold_set) and (asp_pred_set & asp_gold_set):
                    right_num += 1
                

    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num


    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num


    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    precision = precision * 100
    recall = recall * 100
    f_measure = f_measure * 100

    print("+++++++++++++Binary Results ++++++++++++++++++++++++++==")
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    return {"Binary precision": precision, " Binary  recall": recall, "Binary  f1": f_measure}

def proportional_metric_sub_absa(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(
            ele.sub_start_index, ele.sub_end_index, 
            ele.obj_start_index, ele.obj_end_index, 
            ele.aspect_start_index, ele.aspect_end_index, 
            ele.opinion_start_index, ele.opinion_end_index, 
        ) for ele in pred[sent_idx]]))
        pred_num += len(prediction)

        for ele_pred in prediction:
            for ele_gold in gold[sent_idx]:
                # ele_pred = (3, 3, 5, 32, 34, 24, 27, 28, 29)
                # ele_gold = (3, 3, 4, 32, 34, 24, 26, 28, 29)
                sub_pred, obj_pred, aspect_pred = tuple_to_five_ele(ele_pred)
                sub_gold, obj_gold, aspect_gold = tuple_to_five_ele(ele_gold)

                sub_pred_set, obj_pred_set, asp_pred_set = convert_tuple_to_set(sub_pred), convert_tuple_to_set(obj_pred), convert_tuple_to_set(aspect_pred)
                sub_gold_set, obj_gold_set, asp_gold_set = convert_tuple_to_set(sub_gold), convert_tuple_to_set(obj_gold), convert_tuple_to_set(aspect_gold)

                if (sub_pred_set&sub_gold_set) and (obj_pred_set & obj_gold_set) and (asp_pred_set & asp_gold_set):
                    sub_union = sub_pred_set & sub_gold_set
                    obj_union = obj_pred_set & obj_gold_set
                    asp_union = asp_pred_set & asp_gold_set
                    all_union_len = len(sub_union) + len(obj_union) + len(asp_union)
                    all_gold_len = len(sub_gold_set) + len(obj_gold_set) + len(asp_gold_set)
                    cur_num = all_union_len / all_gold_len
                    right_num = right_num + cur_num


    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num


    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num


    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
        
    precision = precision * 100
    recall = recall * 100
    f_measure = f_measure * 100

    print("+++++++++++++ Proportional Results ++++++++++++++++++++++++++==")
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    return {"Proportional precision": precision, " Proportional  recall": recall, "Proportional  f1": f_measure}


def num_metric(pred, gold):
    test_1, test_2, test_3, test_4, test_other = [], [], [], [], []
    for sent_idx in gold:
        if len(gold[sent_idx]) == 1:
            test_1.append(sent_idx)
        elif len(gold[sent_idx]) == 2:
            test_2.append(sent_idx)
        elif len(gold[sent_idx]) == 3:
            test_3.append(sent_idx)
        elif len(gold[sent_idx]) == 4:
            test_4.append(sent_idx)
        else:
            test_other.append(sent_idx)

    pred_1 = get_key_val(pred, test_1)
    gold_1 = get_key_val(gold, test_1)
    pred_2 = get_key_val(pred, test_2)
    gold_2 = get_key_val(gold, test_2)
    pred_3 = get_key_val(pred, test_3)
    gold_3 = get_key_val(gold, test_3)
    pred_4 = get_key_val(pred, test_4)
    gold_4 = get_key_val(gold, test_4)
    pred_other = get_key_val(pred, test_other)
    gold_other = get_key_val(gold, test_other)
    # pred_other = dict((key, vals) for key, vals in pred.items() if key in test_other)
    # gold_other = dict((key, vals) for key, vals in gold.items() if key in test_other)
    print("--*--*--Num of Gold Triplet is 1--*--*--")
    _ = metric(pred_1, gold_1)
    print("--*--*--Num of Gold Triplet is 2--*--*--")
    _ = metric(pred_2, gold_2)
    print("--*--*--Num of Gold Triplet is 3--*--*--")
    _ = metric(pred_3, gold_3)
    print("--*--*--Num of Gold Triplet is 4--*--*--")
    _ = metric(pred_4, gold_4)
    print("--*--*--Num of Gold Triplet is greater than or equal to 5--*--*--")
    _ = metric(pred_other, gold_other)

def overlap_metric(pred, gold):
    normal_idx, multi_label_idx, overlap_idx = [], [], []
    for sent_idx in gold:
        triplets = gold[sent_idx]
        if is_normal_triplet(triplets):
            normal_idx.append(sent_idx)
        if is_multi_label(triplets):
            multi_label_idx.append(sent_idx)
        if is_overlapping(triplets):
            overlap_idx.append(sent_idx)
    pred_normal = get_key_val(pred, normal_idx)
    gold_normal = get_key_val(gold, normal_idx)
    pred_multilabel = get_key_val(pred, multi_label_idx)
    gold_multilabel = get_key_val(gold, multi_label_idx)
    pred_overlap = get_key_val(pred, overlap_idx)
    gold_overlap = get_key_val(gold, overlap_idx)
    print("--*--*--Normal Triplets--*--*--")
    _ = metric(pred_normal, gold_normal)
    print("--*--*--Multiply label Triplets--*--*--")
    _ = metric(pred_multilabel, gold_multilabel)
    print("--*--*--Overlapping Triplets--*--*--")
    _ = metric(pred_overlap, gold_overlap)



def is_normal_triplet(triplets):
    entities = set()
    for triplet in triplets:
        head_entity = (triplet[1], triplet[2])
        tail_entity = (triplet[3], triplet[4])
        entities.add(head_entity)
        entities.add(tail_entity)
    return len(entities) == 2 * len(triplets)


def is_multi_label(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    return len(entity_pair) != len(set(entity_pair))


def is_overlapping(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.append((pair[0], pair[1]))
        entities.append((pair[2], pair[3]))
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


def get_key_val(dict_1, list_1):
    dict_2 = dict()
    for ele in list_1:
        dict_2.update({ele: dict_1[ele]})
    return dict_2
