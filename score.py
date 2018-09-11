#======================================================
#===================score.py==============
#======================================================

#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1
        
    if sum(cm[0])==0:
        agree_recall=0
    else:
        agree_recall=cm[0][0]/sum(cm[0])
    if sum(cm[1])==0:
        disagree_recall=0
    else:
        disagree_recall=cm[1][1]/sum(cm[1])
    if sum(cm[2])==0:
        discuss_recall=0
    else:
        discuss_recall=cm[2][2]/sum(cm[2])
    if sum(cm[3])==0:
        unrelated_recall=0
    else:    
        unrelated_recall=cm[3][3]/sum(cm[3])
    if sum(cm[0]+cm[1]+cm[2]+cm[3])==0:
        all_recall=0
    else:
        all_recall=(cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3])/sum(cm[0]+cm[1]+cm[2]+cm[3])
    
    return score, cm,agree_recall,disagree_recall,discuss_recall,unrelated_recall,all_recall


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(actual,predicted):
    score,cm,agree_recall,disagree_recall,discuss_recall,unrelated_recall,all_recall = score_submission(actual,predicted)
    best_score, _,_,_,_,_ ,_= score_submission(actual,actual)

    #print_confusion_matrix(cm)
    #print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    competition_grade=score*100/best_score
    
    return competition_grade,agree_recall,disagree_recall,discuss_recall,unrelated_recall,all_recall


if __name__ == "__main__":
    actual = [0,0,0,0,1,1,0,3,3]
    predicted = [0,0,0,0,1,1,2,3,3]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])