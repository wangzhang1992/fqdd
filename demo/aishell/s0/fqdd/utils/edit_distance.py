

# Cost of alignment types
SUB_COST = 3
DEL_COST = 3
INS_COST = 3

CRT_ALIGN = 0
SUB_ALIGN = 1
DEL_ALIGN = 2
INS_ALIGN = 3
END_ALIGN = 4


class entry:
    'Alignment chart entry, contains cost and align-type'

    def __init__(self, cost = 0, align = CRT_ALIGN):
        self.cost = cost
        self.align = align

# align_name = ['corr', 'sub', 'del', 'ins', 'end']
'''
a = "词错率但一般称为字错率"
b = "字符错误率中文一般用CER来表示字错率"
print(a+'\n'+b)
print(distance(a, b))
'''
def distance(ref, hyp):
    ref_len = len(ref)
    hyp_len = len(hyp)

    chart = []
    for i in range(0, ref_len + 1):
        chart.append([])
        for j in range(0, hyp_len + 1):
            chart[-1].append(entry(i * j, CRT_ALIGN))

    # Initialize the top-most row in alignment chart, (all words inserted).
    for i in range(1, hyp_len + 1):
        chart[0][i].cost = chart[0][i - 1].cost + INS_COST;
        chart[0][i].align = INS_ALIGN
    # Initialize the left-most column in alignment chart, (all words deleted).
    for i in range(1, ref_len + 1):
        chart[i][0].cost = chart[i - 1][0].cost + DEL_COST
        chart[i][0].align = DEL_ALIGN

    # Fill in the rest of the chart
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            min_cost = 0
            min_align = CRT_ALIGN
            if hyp[j - 1] == ref[i - 1]:
                min_cost = chart[i - 1][j - 1].cost
                min_align = CRT_ALIGN
            else:
                min_cost = chart[i - 1][j - 1].cost + SUB_COST
                min_align = SUB_ALIGN

            del_cost = chart[i - 1][j].cost + DEL_COST
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = DEL_ALIGN

            ins_cost = chart[i][j - 1].cost + INS_COST
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = INS_ALIGN

            chart[i][j].cost = min_cost
            chart[i][j].align = min_align

    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    alignment = []
    while i > 0 or j > 0:
        #if i < 0 or j < 0:
            #break;
        if chart[i][j].align == CRT_ALIGN:
            i -= 1
            j -= 1
            crt += 1
            alignment.append(CRT_ALIGN)
        elif chart[i][j].align == SUB_ALIGN:
            i -= 1
            j -= 1
            sub += 1
            alignment.append(SUB_ALIGN)
        elif chart[i][j].align == DEL_ALIGN:
            i -= 1
            det += 1
            alignment.append(DEL_ALIGN)
        elif chart[i][j].align == INS_ALIGN:
            j -= 1
            ins += 1
            alignment.append(INS_ALIGN)

    total_error = sub + det + ins

    alignment.reverse()
    return total_error, crt, sub, det, ins, alignment

class Static_Info:
    def __init__(self):
       self.N = 0
       self.Del = 0
       self.Ins = 0
       self.Sub = 0
       self.Corr = 0
       self.Error = 0
       self.loss = 0.0
       self.inter_loss = 0.0

    def one_iter(self, batch_ref, batch_hyp, loss):
       assert len(batch_ref) == len(batch_hyp)
       batch_n, batch_error, batch_corr, batch_sub, batch_del, batch_ins = 0, 0, 0, 0, 0, 0
       for i in range(len(batch_ref)):
           tol_err, corr, sub, det, ins, _ = distance(batch_ref[i], batch_hyp[i])
           batch_error += tol_err
           batch_corr += corr
           batch_sub += sub
           batch_del += det
           batch_ins += ins
           batch_n += len(batch_ref[i])
       self.Del += batch_del
       self.N += batch_n
       self.Ins += batch_ins
       self.Sub += batch_sub
       self.Corr += batch_corr
       self.Error += batch_error
       self.loss += loss.item()
       self.inter_loss += loss.item()
       return (batch_del+ batch_sub) / batch_n * 100

    def avg_one_epoch_cer(self):
       return (self.Del+self.Sub)/self.N *100, self.Corr, self.Del, self.Ins, self.Sub
       #return self.Error/self.N *100, self.Corr, self.Del, self.Ins, self.Sub
    
    def get_inter_loss(self, iter_times):
       inter_loss = self.inter_loss / iter_times 
       self.inter_loss = 0.0
       return inter_loss
   
    def avg_one_epoch_loss(self, iters):
        return self.loss / iters
