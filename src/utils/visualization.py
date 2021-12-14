from sty import bg, rs #, fg, ef
from nltk import Tree


def get_sentence_tree(sent_tree_batch, soft_idx_batch, depth_batch, sent_idx):
    sentence_tree = sent_tree_batch[sent_idx]
    soft_indices = soft_idx_batch[sent_idx]
    depth_list = depth_batch[sent_idx]
    tree = []
    for word, soft_idx, depth in zip(sentence_tree, soft_indices, depth_list):
        soft_idx = int(soft_idx)
        depth = int(depth)
        if word == '[PAD]':
            break
        if depth == 0:
            tree.append([(depth, word, soft_idx)])
        if depth == 1:
            tree[-1].append([(depth, word, soft_idx)])
        if depth == 2:
            tree[-1][-1].append([(depth, word, soft_idx)])
    return tree


def compute_attributions_sum(all_tokens, attributions_sum, pred, depth):
    # depth: depth[sentence_idx]
    pos_og_tok = []; pos_og_att = []
    pos_kg_tok = []; pos_kg_att = []
    neg_og_tok = []; neg_og_att = []
    neg_kg_tok = []; neg_kg_att = []
    for d, token, att in zip(depth, all_tokens, attributions_sum[pred]):
        if att > 0:
            if d == 0:
                pos_og_tok.append(token)
                pos_og_att.append(att)
            else:
                pos_kg_tok.append(token)
                pos_kg_att.append(att)
        elif att < 0:
            if d == 0:
                neg_og_tok.append(token)
                neg_og_att.append(att)
            else:
                neg_kg_tok.append(token)
                neg_kg_att.append(att)
        else:
            continue
    print(f"Positive attributions of original sentence is %.4f" % sum(pos_og_att))
    print(f"Positive attributions of knowledge graph is   %.4f" % sum(pos_kg_att))
    print(f"Negative attributions of original sentence is %.4f" % sum(neg_og_att))
    print(f"Negative attributions of knowledge graph is   %.4f" % sum(neg_kg_att))


def att_color(token, att):
    thr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 100]
    brightness = [250, 200, 150, 100, 50, 0]
    br = 255
    for i in range(5):
        if thr[i] < abs(att) <= thr[i+1]:
            br = brightness[i]
    if att > 0:
        token = bg(br, 255, br) + token + rs.bg
    elif att < 0:
        token = bg(255, br, br) + token + rs.bg
    return token


def print_color_sentence_tree(tokens, depth_list, attributions_sum_pred, soft_idx):
    for token, dep, idx, att in zip(tokens, depth_list, soft_idx, attributions_sum_pred):
        if token == '[PAD]':
            break
        token = att_color(token, att)
        # print(token)
        if dep == 0:
            print(token)
        elif dep == 1:
            print('\t'+token)
        elif dep == 2:
            print('\t\t'+token)


def tree_structure(all_tokens, attributions_sum, soft_idx_batch, depth_batch, sent_idx, threshold=(0.1, -0.05)):
    sentence_tree = all_tokens#sent_tree_batch[sent_idx]
    soft_indices = soft_idx_batch[sent_idx]
    depth_list = depth_batch[sent_idx]
    attributions = attributions_sum # attributions_sum[pred]
    tree = ''
    branch_0 = ''
    prev_depth = -1
    prev_soft_idx = -1
    for word, soft_idx, depth, att in zip(sentence_tree, soft_indices, depth_list, attributions):
        soft_idx = int(soft_idx)
        depth = int(depth)
        if att > threshold[0]:
            word += '[+]'
        elif att < threshold[1]:
            word += '[-]'
        else:
            pass
        if depth == 0: # (0->0, 1->0, 2->0)
            if branch_0 == '[PAD]':
                return f"(Tree ({tree[1:]}))"
            else:
                if prev_depth == 0:
                    tree += f" ({branch_0}  (. .))"
                elif prev_depth == 1:
                    tree += f" ({branch_1} .))"
                elif prev_depth == 2:#se: # 2
                    tree += f" {branch_2}))"
                else:
                    pass
                branch_0 = word
        elif depth == 1: # (0->1, 1->1, 2->1)
            if prev_depth == 0: # 0->1
                tree += f" ({branch_0}"
                branch_1 = word
            elif prev_depth == 1: # 1->1
                if soft_idx == prev_soft_idx + 1: # (rel + obj) set
                    branch_1 += f"_{word}"
                else: # other neighbour
                    tree += f" ({branch_1} .)"
                    branch_1  = word#+= f" {word}"
            else: # 2->1
                tree += f" {branch_2})"
                branch_1 = word
        else: # (1->2, 2->2)
            if prev_depth == 0:
                raise Exception()
            elif prev_depth == 1:
                tree += f" ({branch_1}"
                branch_2 = word
            else:
                if soft_idx == prev_soft_idx + 1: # (rel + obj) set
                    branch_2 += f"_{word}"
                else: # other neighbour
                    tree += f" {branch_2}"
                    branch_2  = word#+= f" {word}"
        prev_depth = depth
        prev_soft_idx = soft_idx

    return f"(Tree ({tree[1:]})))"