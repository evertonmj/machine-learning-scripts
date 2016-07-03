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
                                
    #final_values = (p_class_field / sum(p_class_field)) * 100
    final_values = (p_class_field / sum(p_class_field))
    result = []
    
    for x in range(len(niveis)):
        result.append(niveis[x])
        result.append(final_values[x])
    
    formatted_result = dict([(k, v) for k,v in zip (result[::2], result[1::2])])
    
    return formatted_result
    
    
    
#fold1
dataset_file = pd.read_csv('folds/01/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/01/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
            
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold1.csv')

print "********fold1*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)

print "********fold1*********"

#fold2
dataset_file = pd.read_csv('folds/02/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/02/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold2.csv')

print "********fold2*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold2*********"

#fold3
dataset_file = pd.read_csv('folds/03/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/03/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold3.csv')

print "********fold3*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold3*********"

#fold4
dataset_file = pd.read_csv('folds/04/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/04/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold4.csv')

print "********fold4*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold4*********"

#fold5
dataset_file = pd.read_csv('folds/05/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/05/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold5.csv')

print "********fold5*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold5*********"

#fold6
dataset_file = pd.read_csv('folds/06/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/06/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold6.csv')

print "********fold6*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold6*********"

#fold7
dataset_file = pd.read_csv('folds/07/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/07/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold7.csv')

print "********fold7*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold7*********"

#fold8
dataset_file = pd.read_csv('folds/08/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/08/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold8.csv')

print "********fold8*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold8*********"

#fold9
dataset_file = pd.read_csv('folds/09/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/09/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold9.csv')

print "********fold9*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold9*********"

#fold10
dataset_file = pd.read_csv('folds/10/dadosbrutos/01_train.csv', sep = ';', dtype=str)
trainingset = pd.DataFrame(dataset_file)

testset_file = pd.read_csv('folds/10/dadosbrutos/02_test.csv', sep = ',', dtype=str)
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
resultado = []
resultado_arquivo = []

for index, row in testset_df.iterrows():
    resultado = naivebayes(trainingset.ix[:,1:12], trainingset.ix[:,13], row.values[1:12])
    #resultado.append(naivebayes(trainingset.ix[:,0:8], trainingset.ix[:,9], row.values[1:9]))
    media_final_true.append(resultado['true'])
    media_final_false.append(resultado['false'])
    
    if row.values[13] == 'true' and (float(resultado['true']) > float(resultado['false'])):
        vp = vp + 1
    elif row.values[13] == 'true' and (float(resultado['true']) < float(resultado['false'])):
        fn = fn + 1
        
    if row.values[13] == 'false' and (float(resultado['false']) > float(resultado['true'])):
        vn = vn + 1
    elif row.values[13] == 'false' and (float(resultado['false']) < float(resultado['true'])):
        fp = fp + 1
    
    if row.values[13] == 'true':
        row_true = row_true + 1
    else:
        row_false = row_false + 1
        
    if float(resultado['true']) > float(resultado['false']):
        predict_value = resultado['true']
        predict_label = 'true'
    else:
        predict_value = resultado['false']
        predict_label = 'false'
    
    resultado_arquivo.append([predict_value, predict_label, resultado['true'], resultado['false'], row.values[13]])

print_df = pd.DataFrame(resultado_arquivo, columns = ['PREDICT-RESULT', 'PREDICT-LABEL', 'PREDICT-TRUE', 'PREDICT-FALSE', 'REAL-LABEL'])
print_df.to_csv('fold10.csv')

print "********fold10*********"

print "true-naive: " + str(np.mean(media_final_true))
print "false-naive: " + str(np.mean(media_final_false))

#print "true-dataset: " + str((row_true / len(testset_df) * 100))
#print "false-dataset: " + str((row_false / len(testset_df) * 100))
print "true-dataset: " + str((row_true / len(testset_df)))
print "false-dataset: " + str((row_false / len(testset_df)))

print "vp: " + str(vp) + " / vn: " + str(vn) + " / fp: " + str(fp) + " / fn: " + str(fn)
print "********fold10*********"