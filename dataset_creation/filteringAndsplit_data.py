import argparse
import datetime
import string

from tqdm import tqdm
from tqdm import trange
from typing import List

import pandas as pd
import numpy as np



# Show Data
def show_data_analysis(df):
    time = datetime.datetime.now()
    random_number = int(time.strftime("%S"))

    # Positive / Negative : 갯수 확인
    num_positive = len(df[ df['label'] == 1 ])
    num_negative = len(df[ df['label'] == 0 ])
    num_extra = len(df[ df['label'] == -1 ])

    # Positive / Negative : 갯수 출력
    print("[ Positive(1) / Negative(0) / Extra(-1) ]")
    print(" Num of Positive : {}".format( num_positive ))
    print(" Num of Negative : {}".format( num_negative ))
    print(" Num of Extra : {}\n".format( num_extra ))

    # Positive / Negative : text 종류 확인
    print("{ Positive Random Example }")
    print(df[ df['label'] == 1 ].sample(n=20, random_state=random_number))
    print("{ Negative Random Example }")
    print(df[ df['label'] == 0 ].sample(n=20, random_state=random_number))
    if num_extra != 0:
        print("{ Extra Random Example }")
        print(df[ df['label'] == -1 ].sample(n=20, random_state=random_number))

    # text 문자열 길이 확인
    print("\n[ text length ]")
    print(df['text_length'].describe())



# train : validation : test = 8 : 1 : 1
def split_train_validation_test(df):
    print("\n[ Start Split the Dataset ]")

    movie_list = df['title'].unique()
    # 최신 영화 순으로 나열하기 위해 (split 하기 위해) 필요한 dataframe
    need_to_split_df = pd.DataFrame()
    need_to_split_df.index = movie_list
    need_to_split_df['release_date'] = ''
    need_to_split_df['positive'] = 0
    need_to_split_df['negative'] = 0
    need_to_split_df['sum'] = 0

    # 갯수 확인
    for movie in tqdm(movie_list):
        filt = df['title'] == movie
        filt_positive = (df['title'] == movie) & (df['label'] == 1)
        filt_negative = (df['title'] == movie) & (df['label'] == 0)

        # release_date
        need_to_split_df.loc[movie, 'release_date'] = df[filt].iloc[0, 3]#column=첫번째,index='release_date'
        # num of positive review
        need_to_split_df.loc[movie, 'positive'] = len(df[filt_positive])
        # num of negative review
        need_to_split_df.loc[movie, 'negative'] = len(df[filt_negative])
        # num of movie review
        need_to_split_df.loc[movie, 'sum'] = need_to_split_df.loc[movie, 'positive'] + need_to_split_df.loc[movie, 'negative']


    # 32,000 : 4,000 : 4,000
    print("[ 최신 영화 순서로 나열 ]")
    need_to_split_df.sort_values('release_date', inplace=True)
    print(need_to_split_df)


    # 갯수 확인
    total_num = len(df)
    train_num = int(total_num*0.8)
    validation_num = int(total_num*0.1)
    test_num = int(total_num*0.1)



    # Set Train Dataset
    train_idx=0#여기까지 train set
    for i in range(total_num):
        num = need_to_split_df.iloc[0:i, -1].sum()
        if num > train_num:
            train_idx = i
            break

    # train_filt = [ x for x, title in df['title'].items() if title in list(need_to_split_df.index[0:idx_train]) ]
    train_filt=[]
    for x, title in df['title'].items():
        if title in list(need_to_split_df.index[0:train_idx]):
            train_filt.append(x)
    train_df = df.iloc[train_filt]





    # Set Validation Dataset
    validation_idx=train_idx#여기까지 validation set
    for i in range(train_idx, total_num):
        num = need_to_split_df.iloc[train_idx:i, -1].sum()
        if num > validation_num:
            validation_idx = i
            break
    validation_filt=[]
    for x, title in df['title'].items():
        if title in list(need_to_split_df.index[train_idx:validation_idx]):
            validation_filt.append(x)
        
    validation_df = df.iloc[validation_filt]




    # Set Test Dataset
    test_filt=[]
    for x, title in df['title'].items():
        if title in list(need_to_split_df.index[validation_idx:]):
            test_filt.append(x)
    test_df = df.iloc[test_filt]


    check_df = pd.DataFrame(
        {
            "number of unique titles" : [train_idx+1, validation_idx-train_idx+1, len(need_to_split_df)-validation_idx], 
            "number of reviews" : [len(train_df), len(validation_df), len(test_df)],
            "number of positive reviews" : [len(train_df[train_df['label']==1]), len(validation_df[validation_df['label']==1]), len(test_df[test_df['label']==1])],
            "number of negative reviews" : [len(train_df[train_df['label']==0]), len(validation_df[validation_df['label']==0]), len(test_df[test_df['label']==0])]
        }, index=["train", "validation", "test"])
        
    print(check_df)

    print("\n[ train, validation, test dataset 꼭 확인해보기 (positive, negative, 비율) ]\n\n")
    from IPython import embed; embed()



    

    # 분류한 train, validation, test : 각각 Shuffle
    # Shuffle Data, initialized index + (inplace=True)
    train_df = train_df.sample(frac=1, replace=False, random_state=100)#Shuffle
    train_df.reset_index(drop=True, inplace=True)
    validation_df = validation_df.sample(frac=1, replace=False, random_state=100)#Shuffle
    validation_df.reset_index(drop=True, inplace=True)
    test_df = test_df.sample(frac=1, replace=False, random_state=100)#Shuffle
    test_df.reset_index(drop=True, inplace=True)

    return train_df, validation_df, test_df



# Delete By Extra class & punctuation & duplicated text review & text length
def delete_data_by_text(df):
    print("\n[ Before Delete Data : {} ]".format(len(df)))


    # Delete Extra class
    filt = (df['label']==1) | (df['label']==0)
    df = df[filt].copy()#뒤에서 값이 같이 바뀜으로인해 오류 발생 (여기 상황에서 크게 의미가 있지 않음)


    # Delete if df['text'] == NAN
    df.dropna(axis='index', how='any', inplace=True)


    # Delete punctuation ( ex. “영화 너무 좋음.” == “영화 너무 좋음!” )
    # df['text'].apply(lambda x: ''.join([k for k in str(x) if k not in string.punctuation]))


    # Delete Spoiler text ( 필요없는 부분 삭제, 전체 문장을 삭제하는거 아님 )
    df['text'] = df['text'].apply(delete_spoiler)

    # filt = df['text'].str.startswith('스포일러가 포함된 감상평입니다. 감상평 보기\n')
    # df = df[filt]


    # index initialized
    df.reset_index(drop=True, inplace=True)


    # de-duplication
    print("[ Check de-duplication ]")
    text_set = set([])
    for idx in trange(len(df)):
        try:
            text = df.loc[idx, 'text']
            if text in text_set:
                df.drop(index=idx, inplace=True)
            else:
                text_set.add(text)
        except Exception as e:
            from IPython import embed; embed()
    # index initialized
    df.reset_index(drop=True, inplace=True)


    # Update text length
    df['text_length'] = df['text'].apply(text_length)


    # 적절한 Threshold 값 찾기
    # from IPython import embed; embed()
    df = df[ df['text_length'] >= 4 ]

    # index initialized
    df.reset_index(drop=True, inplace=True)


    # 삭제하는 부분 확인
    # print(df[ df['text_length'] < 10 ])#전체
    # print(df[ df['text_length'] < 3 ])
    # print(df[ df['text_length'] == 3 ])
    # print(df[ df['text_length'] == 4 ])
    # print(df[ df['text_length'] == 5 ])
    # print(df[ df['text_length'] == 6 ])
    # print(df[ df['text_length'] == 7 ])
    # print(df[ df['text_length'] == 8 ])
    # print(df[ df['text_length'] == 9 ])
    # print(df[ df['text_length'] == 10 ])
    # print(df[ df['text_length'] > 200 ])

    print("[ After Delete Data : {} ]\n".format(len(df)))
    return df





def main(args):
    df = pd.read_csv('./data/' + args.file_name, sep='\t', index_col='Unnamed: 0')

    # Find "label" : Negative(0), Positive(1)
    df['label'] = -1
    df.loc[ df['score'] >= args.scorePositive, 'label'] = 1
    df.loc[ df['score'] <= args.scoreNegative, 'label'] = 0

    # Find text length
    df['text_length'] = df['text'].apply(text_length)

    # Show Data
    show_data_analysis(df)
    # By duplicated text review & text length
    df = delete_data_by_text(df)
    # Show Data
    show_data_analysis(df)

    train_df, validation_df, test_df = split_train_validation_test(df)

    # show_data_analysis(train_df)
    # show_data_analysis(validation_df)
    # show_data_analysis(test_df)

    # Save dataset
    train_df.to_csv('./data/train_{}'.format(args.file_name), index=True, sep='\t')
    validation_df.to_csv('./data/validation_{}'.format(args.file_name), index=True, sep='\t')
    test_df.to_csv('./data/test_{}'.format(args.file_name), index=True, sep='\t')











def delete_spoiler(text):
    if text.startswith('스포일러가 포함된 감상평입니다. 감상평 보기\n'):
        # print(text[25:].strip())
        return text[25:].strip()
    return text

def text_length(text):
    return len(str(text))


if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser(description='This is load movie review dataset python program.')

    # 2. add arguments to parser
    parser.add_argument('file_name')
    parser.add_argument('--scoreNegative', '-n', type=int, default=4, help='Negative review when it is less or equal than score_negative ( 0 ~ 10 )')
    parser.add_argument('--scorePositive', '-p', type=int, default=8, help='Positive review when it is more or equal than score_positive ( 0 ~ 10 )')


    # 3. parse arguments
    args = parser.parse_args()

    # 4. use arguments
    print("Score Negative : {}".format(args.scoreNegative))
    print("Score Positive : {}\n".format(args.scorePositive))
    main(args)