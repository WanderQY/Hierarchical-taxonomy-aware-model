from class_labels import *

def path_correction2(input_hier, predict_score):
    """ e.g.
        input_hier = [150, 122, 42, 14]
        predict_score,(hier_num, num_spe)
        class_list = [SELECT_CLASS, SELECT_GENUS, SELECT_FAMILY, SELECT_ORDER]
    """
    match_list = []  # Match nodes between the predicted path and each correct path, [[False, False, True, True],...]
    match_score = []  # The number of matching nodes between the predicted path and each correct path, [2,...]
    for t in tree:
        match_list.append([input_hier[0] == t[0], input_hier[1] == t[1], input_hier[2] == t[2], input_hier[3] == t[3]])
    for i in match_list:
        match_score.append(sum(i))
    # Candidate path location
    loc = [i for i in range(len(match_score)) if match_score[i] == max(match_score)]

    if len(loc) == 1:  # If there is only one candidate path, return the path
        return tree[loc[0]]
    else:  # If there are multiple candidate paths, resorted according to the finer level priority.
        loc_score = []
        for i in loc:
            list = match_list[i][::-1]
            num = 0  # Cumulative score of node matching quantity
            for j, k in enumerate(list):
                if k:
                    num += j
            loc_score.append([i, num])
        loc_score.sort(key=lambda x: x[1], reverse=True)
        choose_loc = loc_score[0][0]
        corr_hier = tree[choose_loc]

        path_match = match_list[choose_loc]  # candidate path matching, [False, False, True, True]
        # If the matching node is a leaf node, it will be returned directly.
        k = 0
        while not path_match[k] and not path_match[k+1]:
            if k+1 == len(path_match):
                return input_hier
            k += 1
        while not path_match[k] and k > -1:
            candidate_score = []
            for i in range(len(tree)):
                if tree[i][k+1] == corr_hier[k+1]:
                    candidate_score.append([tree[i], predict_score[k][tree[i][k]]])
            candidate_score.sort(key=lambda x: x[1], reverse=True)
            choose_loc = candidate_score[0][0][0]
            path_match = match_list[choose_loc]
            #path_match[k] = True
            corr_hier = tree[choose_loc]
            k -= 1

        return corr_hier





