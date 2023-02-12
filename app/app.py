import gradio as gr
import fasttext

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import torch


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


title = "Movie Review Score Discriminator"
description = "It is a program that classifies whether it is positive or negative by entering movie reviews.  \
                You can choose between the Korean version and the English version.  \
                It also provides a version called ""Default"", which determines whether it is Korean or English and predicts it."


class LanguageIdentification:
    def __init__(self):
        pretrained_lang_model = "./lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=200) # returns top 200 matching languages
        return predictions

LANGUAGE = LanguageIdentification()



def tokenized_data(tokenizer, inputs):
    return tokenizer.batch_encode_plus(
        [inputs],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True)



examples = []
df = pd.read_csv('examples.csv', sep='\t', index_col='Unnamed: 0')
np.random.seed(100)

idx = np.random.choice(50, size=5, replace=False)
eng_examples = [ ['Eng', df.iloc[i, 0]] for i in idx ]
kor_examples = [ ['Kor', df.iloc[i, 1]] for i in idx ]
examples = eng_examples + kor_examples



eng_model_name = "roberta-base"
eng_step = 1900
eng_tokenizer = AutoTokenizer.from_pretrained(eng_model_name)
eng_file_name = "{}-{}.pt".format(eng_model_name, eng_step)
eng_state_dict = torch.load(eng_file_name)
eng_model = AutoModelForSequenceClassification.from_pretrained(
    eng_model_name, num_labels=2, id2label=id2label, label2id=label2id,
    state_dict=eng_state_dict
)


kor_model_name = "klue/roberta-small"
kor_step = 2400
kor_tokenizer = AutoTokenizer.from_pretrained(kor_model_name)
kor_file_name = "{}-{}.pt".format(kor_model_name.replace('/', '_'), kor_step)
kor_state_dict = torch.load(kor_file_name)
kor_model = AutoModelForSequenceClassification.from_pretrained(
    kor_model_name, num_labels=2, id2label=id2label, label2id=label2id,
    state_dict=kor_state_dict
)


def builder(Lang, Text):
    percent_kor, percent_eng = 0, 0
    text_list = Text.split(' ')


    # [ output_1 ]
    if Lang == '언어감지 기능 사용':
        pred = LANGUAGE.predict_lang(Text)
        if '__label__en' in pred[0]:
            Lang = 'Eng'
            idx = pred[0].index('__label__en')
            p_eng = pred[1][idx]
        if '__label__ko' in pred[0]:
            Lang = 'Kor'
            idx = pred[0].index('__label__ko')
            p_kor = pred[1][idx]
        # Normalize Percentage
        percent_kor = p_kor / (p_kor+p_eng)
        percent_eng = p_eng / (p_kor+p_eng)

    if Lang == 'Eng':
        model = eng_model
        tokenizer = eng_tokenizer
        if percent_eng==0: percent_eng=1

    if Lang == 'Kor':
        model = kor_model
        tokenizer = kor_tokenizer
        if percent_kor==0: percent_kor=1
        

    # [ output_2 ]
    inputs = tokenized_data(tokenizer, Text)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']).logits
    
    m = torch.nn.Softmax(dim=1)
    output = m(logits)
    # print(logits, output)


    # [ output_3 ]
    output_analysis = []
    for word in text_list:
        tokenized_word = tokenized_data(tokenizer, word)
        with torch.no_grad():
            logit = model(input_ids=tokenized_word['input_ids'], 
                attention_mask=tokenized_word['attention_mask']).logits
        word_output = m(logit)
        if word_output[0][1] > 0.99:
            output_analysis.append( (word, '+++') )
        elif word_output[0][1] > 0.9:
            output_analysis.append( (word, '++') )
        elif word_output[0][1] > 0.8:
            output_analysis.append( (word, '+') )
        elif word_output[0][1] < 0.01:
            output_analysis.append( (word, '---') )
        elif word_output[0][1] < 0.1:
            output_analysis.append( (word, '--') )
        elif word_output[0][1] < 0.2:
            output_analysis.append( (word, '-') )
        else:
            output_analysis.append( (word, None) )
    

    return [ {'Kor': percent_kor, 'Eng': percent_eng}, 
            {id2label[1]: output[0][1].item(), id2label[0]: output[0][0].item()}, 
            output_analysis ]
            
    # prediction = torch.argmax(logits, axis=1)
    return id2label[prediction.item()]


# demo3 = gr.Interface.load("models/mdj1412/movie_review_score_discriminator_eng", inputs="text", outputs="text", 
#                          title=title, theme="peach",
#                          allow_flagging="auto",
#                          description=description, examples=examples)



# demo = gr.Interface(builder, inputs=[gr.inputs.Dropdown(['Default', 'Eng', 'Kor']), gr.Textbox(placeholder="리뷰를 입력하시오.")], 
#                     outputs=[ gr.Label(num_top_classes=3, label='Lang'), 
#                             gr.Label(num_top_classes=2, label='Result'),
#                             gr.HighlightedText(label="Analysis", combine_adjacent=False)
#                             .style(color_map={"+++": "#CF0000", "++": "#FF3232", "+": "#FFD4D4", "---": "#0004FE", "--": "#4C47FF", "-": "#BEBDFF"}) ],
#                     # outputs='label',
#                     title=title, description=description, examples=examples)



with gr.Blocks() as demo1:
    gr.Markdown(
    """
    <h1 align="center">
    Movie Review Score Discriminator
    </h1>
    """)

    gr.Markdown(
    """
    영화 리뷰를 입력하면, 리뷰가 긍정인지 부정인지 판별해주는 모델이다. \
    영어와 한글을 지원하며, 언어를 직접 선택할수도, 혹은 모델이 언어감지를 직접 하도록 할 수 있다.  
    리뷰를 입력하면, (1) 감지된 언어, (2) 긍정 리뷰일 확률과 부정 리뷰일 확률, (3) 입력된 리뷰의 어느 단어가 긍정/부정 결정에 영향을 주었는지 \
    (긍정일 경우 빨강색, 부정일 경우 파란색)를 확인할 수 있다.
    """)

    with gr.Accordion(label="모델에 대한 설명 ( 여기를 클릭 하시오. )", open=False):
        gr.Markdown(
        """
        영어 모델은 bert-base-uncased 기반으로, 영어 영화 리뷰 분석 데이터셋인 SST-2로 학습 및 평가되었다.  
        한글 모델은 klue/roberta-base 기반이다. 기존 한글 영화 리뷰 분석 데이터셋이 존재하지 않아, 네이버 영화의 리뷰를 크롤링해서 영화 리뷰 분석 데이터셋을 제작하고, 이를 이용하여 모델을 학습 및 평가하였다.  
        영어 모델은 SST-2에서 92.8%, 한글 모델은 네이버 영화 리뷰 데이터셋에서 94%의 정확도를 가진다 (test set 기준).  
        언어감지는 fasttext의 language detector를 사용하였다. 리뷰의 단어별 영향력은, 단어 각각을 모델에 넣었을 때 결과가 긍정으로 나오는지 부정으로 나오는지를 바탕으로 측정하였다.
        """)

    with gr.Row():
        with gr.Column():
            inputs_1 = gr.Dropdown(choices=['언어감지 기능 사용', 'Eng', 'Kor'], value='언어감지 기능 사용', label='Lang')
            inputs_2 = gr.Textbox(placeholder="리뷰를 입력하시오.", label='Text')
            with gr.Row():
                # btn2 = gr.Button("클리어")
                btn = gr.Button("제출하기")
        with gr.Column():
            output_1 = gr.Label(num_top_classes=3, label='Lang')
            output_2 = gr.Label(num_top_classes=2, label='Result')
            output_3 = gr.HighlightedText(label="Analysis", combine_adjacent=False) \
                .style(color_map={"+++": "#CF0000", "++": "#FF3232", "+": "#FFD4D4", "---": "#0004FE", "--": "#4C47FF", "-": "#BEBDFF"})
    
    # btn2.click(fn=fn2, inputs=[None, None], output=[output_1, output_2, output_3])
    btn.click(fn=builder, inputs=[inputs_1, inputs_2], outputs=[output_1, output_2, output_3])
    gr.Examples(examples, inputs=[inputs_1, inputs_2])
    


if __name__ == "__main__":
    # print(examples)
    # demo.launch()
    demo1.launch()