import numpy as np
from _dbus_bindings import String
import os, pickle
from utilfile import FileUtil
from crf import CRF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class NLPTask(object):

    def __init__(self):
        '''
        Constructor
        '''

    def process_ans(self, lst):
        b_str = ''
        ans_str = ''
        for line_data in lst:
            try :
                data = line_data.split('\t')
                b_data = data[3][:-1]
                if b_data == 'B':
                    b_str = b_str + 'B'
                else:
                    b_str = b_str + 'I'
                ans_str = ans_str+data[0]
            except Exception, e:
                b_str = b_str+'B'
                ans_str = ans_str+' '
        return b_str, ans_str
    
    def crfpp(self, msg):
        crf = CRF()
        fileUtil = FileUtil()
        crf.create_file_input(msg)
        os.system('crf_test -m ../model1 crf.test.data > crf.result')

        lst = fileUtil.read_file('crf.result')
#         lst = [a for a in lst if a != u'\n']
#         str_ans = reduce(lambda x,y:x+y, [a.split('\t')[0] for a in lst])
         
        # ans = reduce(lambda x,y:x+y, [a.split('\t')[3][:-1] for a in lst])
#         lst_col3 = [a.split('\t')[3][:-1] for a in lst]
        lst_col3, str_ans = self.process_ans(lst)
        lst_ans = [n for (n, e) in enumerate(lst_col3) if e == 'B']
        result_lst = []
        for i in range(len(lst_ans)-1):
            a = lst_ans[i]
            b = lst_ans[i+1]
            result_lst.append(str_ans[a:b])
        result_lst.append(str_ans[b:len(str_ans)])
        return result_lst    
            
    def load_data(self):
#         filter_file = np.loadtxt('../data/data.txt', delimiter='||||', dtype=String)
#         y_list = []
# #         corpus = []
#         for data in filter_file:
#             if data[0] == 'yes':
# #                 filter
#                 y_list.append(1)
#             else:
# #                 no filer
#                 y_list.append(0)
#             message = unicode(data[1],'utf-8')
#             message_lst = self.crfpp(message)
#             msg_orgin = ' '.join(message_lst)
#             corpus.append(msg_orgin)
#         pickle.dump(corpus, open('../result/corpus.dat','wb'))
#         pickle.dump(y_list, open('../result/y_list.dat','wb'))

        y_list = pickle.load(open('../result/y_list.dat','rb'))
        corpus = pickle.load(open('../result/corpus.dat','rb'))
        vectorizer = CountVectorizer()
        x_vector = vectorizer.fit_transform(corpus)
        return x_vector.toarray(), y_list
    
    def classification(self, x_train, y_train):
        ml = BaggingClassifier(DecisionTreeClassifier())
        ml.fit(x_train, y_train)
#         print y_train[0]
#         print x_train[0]
        y_pred = ml.predict(x_train)
        print 'y_train ',y_train
        print 'y_pred ',y_pred.tolist()
    
if __name__ == '__main__':
    obj = NLPTask()
    x, y = obj.load_data()
    obj.classification(x, y)
    
    