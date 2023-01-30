import torch
from tqdm import tqdm

from model_util import get_optimizer_and_scheduler, get_dataloader

def train(logger, model, inputs, batch_size, output_dir,
          learning_rate=1e-5,
          warmup_steps=50,
          num_training_steps=200,
          gradient_accumulation_steps=1,
          max_grad_norm=1.0,
          eval_period=20,
          prompt_tune=False,
          head_tune=False,
          transform_tune=False):
    
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps)

    dataloader = get_dataloader(inputs, batch_size, is_training=True)

    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Start training")
    for epoch in tqdm(range(num_training_steps)):
        for batch in dataloader:
            global_step += 1

            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()

            loss = None # TODO with model, input_ids, attention_mask, token_type_ids, labels
            loss = loss.mean()

            if torch.isnan(loss).data:
                print ("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            
            train_losses.append(loss.detach().cpu())
            loss.backward()
            
            if global_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                model.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if global_step % eval_period == 0:
                model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                torch.save(model_state_dict,
                           os.path.join(output_dir, "model-{}.pt".format(global_step)))
                logger.info("Saving model at global_step=%d (train loss %.2f)" % (global_step, np.mean(train_losses)))
                train_losses = []

            if global_step==num_training_steps:
                break

        if global_step==num_training_steps:
            break

    logger.info("Finish training")

def inference(model, inputs, batch_size, return_logits=False):
    dataloader = get_dataloader(inputs, batch_size, is_training=False)
    predictions = []
    for batch in tqdm(dataloader):
        input_ids=batch[0].cuda()
        attention_mask=batch[1].cuda()
        token_type_ids=batch[2].cuda()

        if len(batch)==3:
            labels=None
        else:
            labels=batch[3].cuda()

        with torch.no_grad():
            logits = None # TODO

        # TODO

    return predictions = []

def run_model(model, input_ids, attention_mask, token_type_ids,
              labels=None, return_logits=False):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()

    if return_logits:
        softmax = torch.nn.Softmax(dim=-1)
        return -torch.log(softmax(logits))

    if labels is None:
        labels = input_ids
    labels = labels[..., 1:].contiguous()
    label_mask = token_type_ids[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    losses = loss_fct(logits.view(-1, logits.size(-1)),
                      labels.view(-1)) # [batch_size, length]
    losses = losses.view(logits.size(0), logits.size(1)) * label_mask
    return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

def get_dataloader(inputs, batch_size, is_training):

    if "labels" in inputs:
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"],
                                inputs["labels"])
    else:
        dataset = TensorDataset(inputs["input_ids"],
                                inputs["attention_mask"],
                                inputs["token_type_ids"])

    if is_training:
        sampler=RandomSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader

def get_optimizer_and_scheduler(model,
                                learning_rate=1e-5,
                                warmup_proportion=0.01,
                                warmup_steps=50,
                                weight_decay=0.0,
                                adam_epsilon=1e-8,
                                num_training_steps=1000):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps)

    return optimizer, scheduler


