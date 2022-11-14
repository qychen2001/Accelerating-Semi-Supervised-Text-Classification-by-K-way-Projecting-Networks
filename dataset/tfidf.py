from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
def get_sentence_list():
    # unsup_csv_path="./data/{}/unsup.csv".format(args.task)
    # sup_csv_path="./data/{}/train_{}.csv".format(args.task, str(args.num_sup))
    unsup_csv_path="./data/imdb/unsup.csv"
    sup_csv_path="./data/imdb/train_300.csv"
    
    sentence_list=pd.read_csv(sup_csv_path)['sentence'].to_list()
    sentence_list.extend(pd.read_csv(unsup_csv_path)['sentence'].to_list())
    print(len(sentence_list))
    return sentence_list
    
    
sentence_list=get_sentence_list()
vectorizer = CountVectorizer()
tf_idf_transformer = TfidfTransformer()
#计算每一个词语出现的次数#将文本转换为词频并计算tf-idf;fit_transform()方法用于计算每一个词语出现的次数
X = vectorizer.fit_transform(sentence_list)
tf_idf = tf_idf_transformer.fit_transform(X)
print(tf_idf.toarray())