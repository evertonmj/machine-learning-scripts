#Program to execute Naive-Bayes Method on dataset
from __future__ import division
import pandas as pd
import numpy as np
# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

def naivebayes(dataset, class_field, testset):
    niveis = ['true', 'false']
    
    p_class_field = []
    tamanho = len(class_field)

    for i in range(len(niveis)):
        tamanho_nivel = 0
        
        for y in range(len(class_field)):
            if class_field[y] == niveis[i]:
                tamanho_nivel = tamanho_nivel + 1
                
        p_class_field.append(tamanho_nivel / tamanho)

        for j in range(len(dataset.columns)):
            index = []
            selected_lines = []       
            for z in range(len(class_field)):
                if class_field[z] == niveis[i]:
                    index.append(niveis[i])
                    selected_lines.append(z)
                                 
            result = dataset.ix[selected_lines, dataset.columns[j]].value_counts() / len(index)
            selected_att = 0

            for x in range(len(result)):
                if result.keys()[x] == testset[j]:
                    selected_att = result[x]
            
            p_class_field[i] = p_class_field[i] * selected_att
                                
    final_values = (p_class_field / sum(p_class_field)) * 100
    result = []
    
    for x in range(len(niveis)):
        result.append(niveis[x])
        result.append(final_values[x])
    
    formatted_result = dict([(k, v) for k,v in zip (result[::2], result[1::2])])
    
    return formatted_result
    
dataset_file = pd.read_csv('40-60_trainData.txt', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('40-60_testData.txt', sep = ';', dtype=str)
testset_df = pd.DataFrame(testset_file)

#testset = ['high','medium','low','low','true','true','true','true','true']
media_final_true = []
media_final_false = []
row_true = 0
row_false = 0
vp = 0
vn = 0
fp = 0
fn = 0

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:9], trainingset.ix[:,10], row.values[1:9])
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[10] == 'true' and resultado['true'] > resultado['false']:
        vp = vp + 1
    else:
        vn = vn + 1
        
    if row.values[10] == 'false' and resultado['false'] > resultado['true']:
        fp = fp + 1
    else:
        fn = fn + 1
    
    
    if row.values[10] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1

print "true-naive: " + str(np.nanmean(media_final_true))
print "false-naive: " + str(np.nanmean(media_final_false))

print "true-dataset: " + str((row_true / len(testset_df) * 100))
print "false-dataset: " + str((row_false / len(testset_df) * 100))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)