import os
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import train, test

from huggingface_hub import login


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


def tokenized_data(tokenizer, inputs):
    return tokenizer.batch_encode_plus(
            # 다양한 방법이 있지만, 여기서는 List[str]
            inputs,
            # If set, will return tensors instead of list of python integers.
            # Return PyTorch torch.Tensor objects.
            return_tensors="pt",
            # True or 'longest': 
            #       Pad to the longest sequence in the batch 
            #       (or no padding if only a single sequence if provided).
            # 'max_length': 
            #       Pad to a maximum length specified with the argument max_length
            #       or to the maximum acceptable input length for the model 
            #       if that argument is not provided.
            padding="max_length",
            # Controls the maximum length 
            # to use by one of the truncation/padding parameters.
            max_length=64,
            # True or 'longest_first' : 
            # 'only_first' :
            # 'only_second' :
            # False or 'do_not_truncate' (default) : 
            truncation=True)


# padding을 할 때, 적절한 "max_lenght" 찾기
def determine_length(tokenizer, inputs):
    # padding과 truncation을 하지 않고 tokenized 하기
    tokenized = tokenizer.batch_encode_plus(inputs)
    
    print("\n[ padding을 할 때, 적절한 'max_lenght' 찾기 ] ")
    print("평균: ", np.mean( [ len(input_ids) for input_ids in tokenized['input_ids'] ] ))
    print("중간값: ", np.median( [ len(input_ids) for input_ids in tokenized['input_ids'] ] ))
    print("상위90%: ", np.percentile( [ len(input_ids) for input_ids in tokenized['input_ids'] ] , 90))
    print("상위95%: ", np.percentile( [ len(input_ids) for input_ids in tokenized['input_ids'] ] , 95))
    print("상위99%: ", np.percentile( [ len(input_ids) for input_ids in tokenized['input_ids'] ], 99))

    # XXX : Error 
    from IPython import embed; embed()
    print("Max: ", np.max( [ len(input_ids) for input_ids in tokenizer['input_ids'] ] ))
    print("Max: ", np.percentile( [ len(input_ids) for input_ids in tokenizer['input_ids'] ], 100))











# Print Top 10 Accuracy
def show_top_ten(df):
    idx = df['accuracy'].argsort()[-10:][::-1]
    idx_unnamed = df.index[idx]
    best_10_accuracy = df.iloc[idx]

    print("[ Best Top 10 Accuracy : {} ]".format(args.model))
    for i in range(len(best_10_accuracy)):
        print("{}: {:.2f}% ( lr = {} / bs = {} / step = {} )"
            .format(i+1, best_10_accuracy['accuracy'][idx_unnamed[i]], 
                best_10_accuracy['learning_rate'][idx_unnamed[i]], 
                int(best_10_accuracy['batch_size_train'][idx_unnamed[i]]), 
                int(best_10_accuracy['step'][idx_unnamed[i]])))
    print()


# learning_rate and batch_size_train 기준 Best Top Accuracy
def show_specific_lr_bs(df, lr, bs):
    filt = (df['learning_rate'] == lr) & (df['batch_size_train'] == bs)
    df = df[filt]

    # Accuracy 정렬 이후, 가장 높은 정확도 10개 추출, 마지막으로 내림차순 정렬
    idx = df['accuracy'].argsort()[-10:][::-1]
    best_10_accuracy = df.iloc[idx]

    print("[ Best Top 10 Accuracy : ( lr = {} bs = {} ) ]".format(lr, bs))
    print(best_10_accuracy)






def main(args):

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else :
        device = 'cpu'

    print(device)

    # Draw Graph
    if args.graph:
        # File Name about Save Accuracy
        save_dir = "save_accuracy.xlsx"

        # Determine if it exists
        if not os.path.exists(save_dir):
            raise NotImplementedError("Not exists save_accuracy.xlsx")
        else :
            df = pd.read_excel(save_dir, index_col='Unnamed: 0')

        # Filter about input model
        filt = (df['model_name'] == args.model)
        df = df.loc[filt, ['learning_rate', 'batch_size_train', 'step', 'accuracy']]


        # Kind of learning_rate
        lr_list=list(set(df['learning_rate'].values))
        # Each of Learning Rate
        for lr in lr_list:
            # learning_rate Grouping
            df1 = df.groupby('learning_rate').get_group(lr)
            # Kind of learning_rate
            bs_list=list(set(df1['batch_size_train'].values))
            # Each of Batch Size
            for bs in bs_list:
                # batch_size_train Grouping
                df2 = df1.groupby('batch_size_train').get_group(bs)

                plt.plot(df2['step'], df2['accuracy'], 
                        label='{} / {}'.format(lr, bs),
                        ls='-', marker='o', markersize=2)


        # Best Accuracy
        best_accuracy = df.iloc[np.argmax(df['accuracy'])]
        # Max Step Size
        step_size = df.iloc[np.argmax(df['step']), 2]


        # Print Top 10 Accuracy
        show_top_ten(df)
        # learning_rate and batch_size_train 기준 Best Top Accuracy
        show_specific_lr_bs(df, 7e-5, 64)



        plt.legend(loc='lower right', fontsize=10)
        plt.axis([0, step_size+100, 0, 100])
        plt.title('Model Name : {0}\n Best Accuracy : {1:.2f}% ( lr = {2} / bs = {3} / step = {4})'
            .format(args.model, best_accuracy['accuracy'], best_accuracy['learning_rate'], 
            int(best_accuracy['batch_size_train']), int(best_accuracy['step'])))
        plt.xlabel('Step')# Write X-axis
        plt.ylabel('Accuracy')# Write Y-axis
        plt.show()

        exit()




    if args.movie:
        # Load Movie Dataset
        print("\n[ Load Movie Dataset ]")

        # File Name (train, validation, test)
        file_name_train = "./data/train_" + args.movie_dataset_file
        file_name_validation = "./data/validation_" + args.movie_dataset_file
        file_name_test = "./data/test_" + args.movie_dataset_file

        # Load Dataset (train, validation, test)
        train_df = pd.read_csv(file_name_train, sep='\t', index_col='Unnamed: 0')
        validation_df = pd.read_csv(file_name_validation, sep='\t', index_col='Unnamed: 0')
        test_df = pd.read_csv(file_name_test, sep='\t', index_col='Unnamed: 0')
        
        for key, value in {'train' : train_df, 'validation' : validation_df, 'test' : test_df}.items():
            print ("%d examples in split=%s" % (len(value), key))

        # Movie Dataset 에서는 각각 
        # 'score', 'text', 'title', 'label', 'text_length' 로 구성되어 있다.

        # Load Tokenizer
        print("\n[ Load Tokenizer ]")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # 시작하기 전에 항상 데이터셋을 shuffle
        np.random.seed(100)
        train_dataset = np.random.permutation(train_df)

        train_inputs = list(train_df['text']) # list of strings
        train_labels = list(train_df['label'])
        validation_inputs = list(validation_df['text']) # list of strings
        validation_labels = list(validation_df['label'])
        test_inputs = list(test_df['text']) # list of strings
        test_labels = list(test_df['label'])

        # train : validation : test ( = 8 : 1 : 1 )
        print("{} examples in train".format(len(train_inputs)))
        print("{} examples in validation".format(len(validation_inputs)))
        print("{} examples in test".format(len(test_inputs)))


        # padding을 할 때, 적절한 "max_lenght" 찾기
        # determine_length(tokenizer, train_inputs)

        output_dir = "save/{}".format(args.model) + "/lr({})bs_train({})".format(args.learning_rate, args.batch_size_train)


    else:
        # Load Dataset
        print("\n[ Load sst2 Dataset ]")
        sst2_dataset = load_dataset('sst2')
        for split in ["train", "validation", "test"]:
            print ("%d examples in split=%s" % (len(sst2_dataset[split]), split))

        # sst_dataset 의 각각의 ['train'], ['validation'], ['train'] 은
        # 'idx', 'sentence', 'label' 로 구성되어 있다.


        # Load Tokenizer
        print("\n[ Load Tokenizer ]")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # train, validation = 9 : 1
        dataset_size = len(sst2_dataset["train"])
        train_size = int(dataset_size * 0.9)
        validation_size = dataset_size - train_size


        # 시작하기 전에 항상 데이터셋을 shuffle
        np.random.seed(100)
        train_dataset = np.random.permutation(sst2_dataset['train'])

        train_inputs = [dp["sentence"] for dp in train_dataset[:train_size]] # list of strings
        train_labels = [dp["label"] for dp in train_dataset[:train_size]]
        validation_inputs = [dp["sentence"] for dp in train_dataset[train_size:]] # list of strings
        validation_labels = [dp["label"] for dp in train_dataset[train_size:]]
        test_inputs = [dp["sentence"] for dp in sst2_dataset["validation"]] # list of strings
        test_labels = [dp["label"] for dp in sst2_dataset["validation"]]

        # train : validation : test ( = 9 : 1 : sst2_dataset['validation'] )
        print("{} examples in train".format(len(train_inputs)))
        print("{} examples in validation".format(len(validation_inputs)))
        print("{} examples in test".format(len(test_inputs)))


        # padding을 할 때, 적절한 "max_lenght" 찾기
        # determine_length(tokenizer, train_inputs)

        output_dir = "save/{}".format(args.model) + "/lr({})bs_train({})".format(args.learning_rate, args.batch_size_train)




    if args.train:

        # Before :  type(inputs) - list
        #           len(inputs) - len(dataset)
        train_inputs = tokenized_data(tokenizer, train_inputs) # tokenized about train data
        validation_inputs = tokenized_data(tokenizer, validation_inputs) # tokenized about train data
        # After : type(inputs) - transformers.tokenization_utils_base.BatchEncoding
        # len(inputs) = 2
        # inputs = {'input_ids' : torch.Tensor, 'attention_mask' : torch.Tensor}

        # type(inputs['input_ids']) = torch.Tensor
        # inputs['input_ids'].shape = torch.Size([ len(dataset), padding_max_length ])
        #       'input_ids' : token들의 id 리스트 (sequence of token id)

        # type(inputs['attention_mask']) = torch.Tensor
        # inputs['attention_mask'].shape = torch.Size([ len(dataset), padding_max_length ])
        #       'attention_mask' : 
        #           attention 연산이 수행되어야 할 token과 무시해야 할 token을 구별하는 정보가 담긴 리스트.
        #           bert-base-uncased tokenizer는 attention 연산이 수행되어야 할, 일반적인 token에는 1을 부여하고,
        #           padding과 같이 attention 연산이 수행할 필요가 없는 token들에는 0을 부여한다.


        
        # Load pretrained model
        print("\n[ Load pretrained model ]")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, id2label=id2label, label2id=label2id
        )


        # Determine if it exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            print ("%s already exists!" % output_dir)


        print("\n[ Start Train ]")
        train(args, model, train_inputs, train_labels, validation_inputs, validation_labels, 
            batch_size_train=args.batch_size_train, 
            batch_size_validation=args.batch_size_test,
            output_dir=output_dir,
            save_boundary_accuracy=args.save_boundary_accuracy,
            learning_rate=args.learning_rate,
            warmup_steps=50,
            num_training_steps=args.num_training_steps,
            num_epochs=args.epochs,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            eval_period=100,
            device=device)
    

    if args.validation or args.test:

        # Which dataset to use ( validation or test )
        if args.validation:
            print("\n[ Set the validation dataset ]\n")
            inputs = validation_inputs
            labels = validation_labels
        elif args.test:
            print("\n[ Set the test dataset ]\n")
            inputs = test_inputs
            labels = test_labels
        else:
            raise NotImplementedError("Problem about validation or test")

        # Determine if it exists
        if not os.path.exists(output_dir):
            raise NotImplementedError("Not exists {} directory", output_dir)



        # Before :  type(inputs) = list
        #           len(inputs) = len(dataset)
        inputs = tokenized_data(tokenizer, inputs) # tokenized about train data
        # After : type(inputs) = transformers.tokenization_utils_base.BatchEncoding
        # len(inputs) = 2
        # inputs = {'input_ids' : torch.Tensor, 'attention_mask' : torch.Tensor}

        # type(inputs['input_ids']) = torch.Tensor
        # inputs['input_ids'].shape = torch.Size([ len(dataset), padding_max_length ])
        #       'input_ids' : token들의 id 리스트 (sequence of token id)

        # type(inputs['attention_mask']) = torch.Tensor
        # inputs['attention_mask'].shape = torch.Size([ len(dataset), padding_max_length ])
        #       'attention_mask' : 
        #           attention 연산이 수행되어야 할 token과 무시해야 할 token을 구별하는 정보가 담긴 리스트.
        #           bert-base-uncased tokenizer는 attention 연산이 수행되어야 할, 일반적인 token에는 1을 부여하고,
        #           padding과 같이 attention 연산이 수행할 필요가 없는 token들에는 0을 부여한다.


        # 입력한게 없으면 해당 디렉토리 모든 .pt 파일 실행
        if args.steps is None:
            steps = [ckpt.split("-")[1].split(".")[0] \
                for ckpt in os.listdir(output_dir) \
                if ckpt.startswith("model-") and ckpt.endswith(".pt")]
        # 입력할 때, 콤마(,)로 구분
        else:
            steps = args.steps.split(",")
        
        steps = sorted([int(step) for step in steps])

        # model / learning_rate / batch_size_train / batch_size_test
        print("[ Model : {} / learning_rate : {} / batch_size_train : {} / steps : {} / batch_size_test : {} ]"
            .format(args.model, args.learning_rate, args.batch_size_train, args.steps, args.batch_size_test))
        # 실행한 step들 출력
        for step in steps:
            file_name = "model-{}.pt".format(step)

            # Load 200-th step weights
            # state_dict = torch.load(os.path.join(output_dir, "model-200.pt"))
            state_dict = torch.load(os.path.join(output_dir, file_name))

            # Load pretrained model with weights ( state_dict )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=2, id2label=id2label, label2id=label2id,
                state_dict=state_dict
            )

            # Return : Accuracy
            acc = test(args, model, inputs, labels, batch_size=args.batch_size_test, step=step, device=device) # [ len(dataset), 2 ]

            print("File Name: %s\tAccuracy: %.1f%%" % (file_name, acc))
            print("# of parameters : %d\n" % (np.sum([p.numel() for p in model.parameters()]))) # 11,123,023 == 11M

            from IPython import embed; embed()
            exit()

            login(token='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')
            tokenizer.push_to_hub("movie_review_score_discriminator_eng")
            model.save_pretrained('movie_review_score_discriminator_eng', push_to_hub=True)
    
        







if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser(description='This is text classification python program.')

    # 2. add arguments to parser
    parser.add_argument('--model', '-m', type=str, default="distilbert-base-uncased", 
        help='model name (Example) : distilbert-base-uncased, distilbert-base-cased, \
            bert-base-uncased, bert-base-cased, roberta-base, etc')

    # 넷 중에 하나 선택
    parser.add_argument('--train', action="store_true", help='Train distilbert-base-uncased model with sst2') # type : boolean
    parser.add_argument('--validation', action="store_true", help='Validation distilbert-base-uncased model with sst2') # type : boolean
    parser.add_argument('--test', action="store_true", help='Test distilbert-base-uncased model with sst2') # type : boolean
    parser.add_argument('--graph', action="store_true", help='Draw Graph') # type : boolean

    parser.add_argument('--batch_size_train', '-bstr', type=int, default=16, help='Size of batch (train)')
    parser.add_argument('--batch_size_test', '-bste', type=int, default=4, help='Size of batch (test)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='Learning Rate to train')

    # 두 개 차이 중요
    parser.add_argument('--steps', '-s', type=str, default=None, help='Writing''steps accuracy when validation or testing')
    parser.add_argument('--num_training_steps', '-n', type=int, default=2000, help='how many steps when we training')
    parser.add_argument('--save_boundary_accuracy', '-sba', type=float, default=93.0, help='save boundary accuracy in excel file when we training')

    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epoch to train.')#ignore


    # Movie Dataset
    parser.add_argument('--movie', action="store_true", help='movie dataset if it is true. or not sst2') # type : boolean
    parser.add_argument('--movie_dataset_file', '-mdf', type=str, default=None, help='movie dataset')


    # 3. parse arguments
    args = parser.parse_args()

    # 4. use arguments
    print("num_training_steps : ", args.num_training_steps)
    print("lr : ", args.learning_rate)
    print("save_boundary_accuracy : ", args.save_boundary_accuracy)
    # print(args)
    # print('model :',args.model)
    # print('epochs :',args.epochs)
    # print('batch-size :',args.batch_size)

    assert args.train or args.validation or args.test or args.graph, 'Choose train or validation or test or graph'


    # Check movie dataset
    if args.movie:
        assert args.movie_dataset_file != None, 'Check movie dataset file name'
    else:
        assert args.movie_dataset_file == None, 'Check movie dataset file name'
    

    main(args)