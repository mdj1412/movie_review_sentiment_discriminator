import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def train(args, model, train_inputs, train_labels, validation_inputs, validation_labels, 
            batch_size_train, batch_size_validation, 
            output_dir, 
            learning_rate=1e-5,
            # XXX
            warmup_steps=50,
            # how many steps will training
            num_training_steps=200,
            num_epochs=20,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            # when will you save it during the steps
            eval_period=50,
            device="mps"):


    # batch size = 16, step = 200 : 3200 training examples
    # 1 epoch = 4210 steps (4210 * 16 training examples)


    # optimizer와 scheduler를 가져온다. 
    optimizer, scheduler = get_optimizer_and_scheduler(
            # 해당 모델 + weight 정보
            model,
            # optimizer 에서 필요한 정보
            learning_rate=learning_rate,
            # XXX
            warmup_steps=warmup_steps,
            # scheduler 에서 필요한 정보 : 몇 steps 학습 시킬 것인지
            num_training_steps=num_training_steps)


    # inputs["input_ids"], inputs["attention_mask"], laebls를 묶어주고 ( TensorDataset )
    # sampler와 batch_size를 적용시킨다, ( DataLoader )
    train_dataloader = get_dataloader(
        # type(train_inputs) = transformers.tokenization_utils_base.BatchEncoding
        train_inputs, 
        train_labels, batch_size_train, 
        # RandomSampler를 할 것인지, SequentialSampler를 할 것인지
        is_training=True)


    # 모델의 매개변수 Tensor를 mps Tensor로 변환
    model.to(device)
    # evaluation mode or training mode
    model.train()


    global_step = 0
    train_losses = []

    # Early Stopping Variable
    early_stopping = 0
    best_accuracy = -1

    # start_time = time.time()
    # end_time

    stop_training=False

    print("Start training")
    for epoch in range(num_epochs):
        # tqdm(dataloader) : len(dataset) / batch_size = 67349 / 16 = 4209.3125 = 4210
        for batch in tqdm(train_dataloader):
            global_step += 1

            # Tensor를 mps Tensor로 변환
            input_ids=batch[0].to(device) # torch.Size([batch_size, length])
            attention_mask=batch[1].to(device) # torch.Size([batch_size, length])
            labels=batch[2].to(device) # torch.Size([batch_size])

            # XXX
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            # ex) loss = tensor(0.6938, device='mps:0', grad_fn=<NllLossBackward0>)
            # type(loss) = torch.Tensor
            # loss.shape = torch.Size([])

            # XXX
            if torch.isnan(loss).data:
                print ("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            
            train_losses.append(loss.detach().cpu())
            # [ Example ]
            # loss = tensor(0.6904, device='mps:0', grad_fn=<NllLossBackward0>)
            # loss.detach() = tensor(0.6904, device='mps:0')
            # loss.detach().cpu() = tensor(0.6904)

            # XXX : 실행 했을 때 오류
            # Error backpropagation
            # 예측 손실(prediction loss)을 역전파
            # 각 매개변수에 대한 손실의 변화도를 저장
            loss.backward()

            # ===============================================================
            # 개인적 질문
            # https://tutorials.pytorch.kr/beginner/blitz/neural_networks_tutorial.html
            # print(loss.grad_fn)  # MSELoss
            # print(loss.grad_fn.next_functions[0][0])  # Linear
            # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
            # ===============================================================


            # XXX : 확신이 없음
            # loss.backward()로 인해 발생한 오차를 이용해서
            # optimizer과 scheduler를 조정한다.
            if global_step % gradient_accumulation_steps == 0:
                # XXX
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # "loss.backward()"에서 저장된 변화도를 이용해서 매개변수를 조정
                optimizer.step()

                # 모든 매개변수의 변화도 버펴를 0으로 만듦 
                # ( 기존에 계산된 변화도의 값을 누적 시키고 싶지 않을 때, 기존에 계산된 변화도를 0으로 만드는 작업 )
                model.zero_grad()

                # XXX
                if scheduler is not None:
                    scheduler.step()

            # 50 (eval_period) 번 마다 저장시킨다.
            if global_step % eval_period == 0:
                # Check the accuracy with validation
                acc = test(args=args, model=model, 
                    inputs=validation_inputs, labels=validation_labels, 
                    batch_size=batch_size_validation, 
                    step=global_step,
                    device=device,
                    version="train")

                print("\n[ Validation ] step: %d\tAccuracy: %.1f%%" 
                                % (global_step, acc))

                # Do Not Save
                if acc > 88.3:
                    # XXX
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}

                    # Check Time
                    # end_time = time.time()
                    # time_taken = end_time-start_time

                    # Save
                    torch.save(model_state_dict,
                        os.path.join(output_dir, "model-{}.pt".format(global_step)))
                    print("Saving model at global_step: %d\t(train loss: %.2f)\t(learning_rate: %f)\t(time: {?})" 
                                % (global_step, np.mean(train_losses), learning_rate))

                    # Initialization train_losses(List)
                    train_losses = []

                    # Check Early Stopping
                    if best_accuracy == -1:
                        best_accuracy = acc
                    elif best_accuracy <= acc:
                        best_accuracy = acc
                        early_stopping = 0
                    else:
                        early_stopping += 1
                    
                    # eval_period * 10 steps 동안 정확도가 개선되지 않으면 Stop
                    if early_stopping == 10:
                        break

            # Arrive num_training_steps
            if global_step==num_training_steps:
                break
        
        # Arrive num_training_steps
        if global_step==num_training_steps:
            break

    print("Finish training")


def test(args, model, inputs, labels, batch_size, step, device="mps", version="test"):

    if version == "train":
        # Load Save Accuracy
        save_dir = "save_accuracy.xlsx"
        if not os.path.exists(save_dir):
            data = {'model_name' : [],
                    'learning_rate' : [],
                    'batch_size_train' : [],
                    'step' : [],
                    'accuracy' : []}
            df = pd.DataFrame(data)
        else:
            df = pd.read_excel(save_dir, index_col='Unnamed: 0')


    # 모델의 매개변수 Tensor를 mps Tensor로 변환
    model.to(device)
    # evaluation mode or training mode
    model.eval()

    # inputs["input_ids"], inputs["attention_mask"], laebls를 묶어주고 ( TensorDataset )
    # sampler와 batch_size를 적용시킨다, ( DataLoader )
    dataloader = get_dataloader(inputs, None, batch_size, is_training=False)
    all_logits = []

    # tqdm(dataloader) : len(dataloader) / batch_size = 872 / 16 = 54.5 = 55
    for batch in tqdm(dataloader):
        # Tensor를 mps Tensor로 변환
        input_ids=batch[0].to(device) # torch.Size([ batch_size, length ])
        attention_mask=batch[1].to(device) # torch.Size([ batch_size, length ])

        # Since evaluation mode
        with torch.no_grad():
            # logits.shape = torch.Size([ batch_size, 2 ])
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # device : 'mps' -> 'cpu'
        logits = logits.detach().cpu()
        all_logits.append(logits)

    # all_logits.shape = torch.Size([ len(dataloader), 2 ])
    all_logits = torch.cat(all_logits, axis=0) # [872, 2]

    # torch.argmax(input) :
    #       Returns the "indices" of the maximum value of all elements in the input tensor.
    predictions = torch.argmax(all_logits, axis=1)

    # list -> torch.Tensor(int)
    labels = torch.LongTensor(labels)

    acc = torch.sum(predictions==labels) / labels.shape[0]
    acc = acc * 100

    if version == "train":
        # Save Accuracy
        if df.empty : #First Time
            df.loc[0] = [args.model, args.learning_rate, args.batch_size_train, step, acc.item()]
        else : 
            # Filter 
            filt = (df['model_name'] == args.model) & \
                    (df['learning_rate'] == args.learning_rate) & \
                    (df['batch_size_train'] == args.batch_size_train) & \
                    (df['step'] == step)
            # When I first recorded it,
            if df[filt].empty :
                df.loc[len(df)] = [args.model, args.learning_rate, 
                    args.batch_size_train, step, acc.item()]
            # When it's already recorded, overwriting accuracy
            else :
                df.loc[filt, 'accuracy'] = acc.item()

        df.to_excel(save_dir)

    return acc
    # np.mean(np.array(predictions) == np.array(labels))


# type(inputs) = transformers.tokenization_utils_base.BatchEncoding ( 이미 완료 )
# type(labels) = list -> torch.Tensor ( 변경 )
# inputs["input_ids"], inputs["attention_mask"], labels를 TensorDataset type으로 묶어준다.
# 그 다음, sampler와 batch_size를 추가하여 "DataLoader" type으로 변경해준다.
def get_dataloader(inputs, labels, batch_size, is_training):

    if labels is not None:

        # Before :  type(labels) = list
        #           len(labels) = len(dataset)
        labels = torch.LongTensor(labels)   # list -> int형 torch.Tensor ( type 변환 )
        # After :   type(labels) = torch.Tensor
        #           len(labels) = len(dataset)
        #           labels.shape = torch.Size([ len(dataset) ])


        # TensorDataset() : Dataset wrapping tensors.
        dataset = TensorDataset(
            # 세 개 arguments 모두 torch.Tensor
            inputs["input_ids"],
            inputs["attention_mask"],
            labels)
        # type(dataset) = torch.utils.data.dataset.TensorDataset
        # len(dataset) = len(dataset) ( = 67349 )
    else:
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"])


    if is_training:
        # train 할 때, 한 번 섞어준다.
        sampler=RandomSampler(dataset)
        # type(sampler) : torch.utils.data.sampler.RandomSampler
    else:
        # test 할 때, 순차적으로 한다.
        sampler=SequentialSampler(dataset)

    # XXX
    # 전체 데이터셋을 batch_size 만큼 grouping 하는 듯...
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    # type(dataloader) = torch.utils.data.dataloader.DataLoader
    
    return dataloader


def get_optimizer_and_scheduler(
        # 해당 모델 + weight 정보
        model,
        # optimizer 에서 필요한 정보
        learning_rate=1e-5,
        # XXX
        warmup_proportion=0.01,
        # scheduler 에서 필요한 정보 : 
        warmup_steps=50,
        # 
        weight_decay=0.0,
        # optimizer 에서 필요한 정보
        adam_epsilon=1e-8,
        # scheduler 에서 필요한 정보 : 몇 steps 학습 시킬 것인지
        num_training_steps=1000):

    no_decay = ['bias', 'LayerNorm.weight']

    # optimizer_grouped_parameters : List = [dict, dict]
    # optimizer_grouped_parameters[0] = { 'params' : [], 'weight_decay' : weight_decay }
    # optimizer_grouped_parameters[1] = { 'params' : [], 'weight_decay' : 0.0 }
    optimizer_grouped_parameters = [
        # for n, p in model.named_parameters():

        # type(model.named_parameters()) : <class 'generator'> : ???
        # type(n) : <class 'str'> : 모델에서 각 Layer의 weight 위치 이름 저장
        # type(p) : <class 'torch.nn.parameter.Parameter'> : 그 위치에서 저장된 weight 값들

        # len( list(model.named_parameters()) ) == 104
        # len( optimizer_grouped_parameters[0]['params'] ) == 52
        # len( optimizer_grouped_parameters[1]['params'] ) == 52
        # 두 개로 grouping 했다.
        # 기준 : XXX
        {'params': [p for n, p in model.named_parameters() if not any (nd in n for nd in no_decay)], 
        'weight_decay': weight_decay},

        {'params': [p for n, p in model.named_parameters() if any (nd in n for nd in no_decay)], 
        'weight_decay': 0.0}
    ]


    # "Adam" module in transformers package
    # type(optimizer) - transformers.optimization.AdamW
    optimizer = AdamW(
        # params (Iterable[nn.parameter.Parameter]) 
        #       - Iterable of parameters to optimize or dictionaries defining parameter groups.
        optimizer_grouped_parameters, 
        # lr (float, optional, defaults to 1e-3) — The learning rate to use.
        lr=learning_rate, 
        # eps (float, optional, defaults to 1e-6) — Adam’s epsilon for numerical stability.
        eps=adam_epsilon)


    # "get_linear_schedule_with_warmup" module in transformers package
    # type(scheduler) - torch.optim.lr_scheduler.LambdaLR
    # XXX
    scheduler = get_linear_schedule_with_warmup(
        # optimizer (~torch.optim.Optimizer) 
        #           — The optimizer for which to schedule the learning rate.
        #           - Learning Rate가 설정된 optimizer
        optimizer,
        # XXX
        # num_warmup_steps (int) — The number of steps for the warmup phase.
        num_warmup_steps=warmup_steps,
        # num_training_steps (int) — The total number of training steps.
        num_training_steps=num_training_steps)

    return optimizer, scheduler


