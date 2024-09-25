import os
import json
import vllm
import tqdm
import torch
import openai
import random
import backoff
import argparse
import numpy as np
import torch.nn.functional as F

from rouge_score import rouge_scorer
from vllm import LLM, SamplingParams
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    
from nltk.translate.gleu_score import sentence_gleu # 24.05.31 update - fluency of prediction compared to long-form answer
import pprint
import time
# from openai.error import APIError, Timeout, APIConnectionError

# openai.api_key_path = "./key.txt"
# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
# def completions_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)


def BERTSCORE(pred, answer):
    import bert_score
    from bert_score import score
    prec, rec, f1 = score([pred], [answer], lang='en', verbose=True)
    return prec.mean().item(), rec.mean().item(), f1.mean().item()

def ROUGESCORE(pred, answer):
    # for ASQA, K-QA datset
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    # scorer.score(pred, answer)['rougeL']
    # precision, recall, fmeasure
    rouge1_score = scorer.score(pred, answer)['rouge1']
    rouge2_score = scorer.score(pred, answer)['rouge2']
    rougel_score = scorer.score(pred, answer)['rougeL']
    # return scorer.score(pred, answer)['rougeL'].fmeasure
    return rouge1_score, rouge2_score, rougel_score

def BLEURT(pred, answer, model=None, tokenizer=None, device=None):
    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12') # lucadiliello/BLEURT-20
    model.eval()
    with torch.no_grad():
        try:
            inputs = tokenizer([answer], [pred], padding='longest', return_tensors='pt').to(device)
            output = model(**inputs)
            res = output.logits.flatten().tolist()
        except:
            # truncate to max length
            inputs['input_ids'] = inputs['input_ids'][:, :512].to(device)
            inputs['attention_mask'] = inputs['attention_mask'][:, :512].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'][:, :512].to(device)
            output = model(**inputs)
            res = output.logits.flatten().tolist()

    return res

def HALLUCINATION(query, pred, must_have, nice_to_have, use_gpt=False, model=None, tokenizer=None, device=None):
    all_statements = must_have + nice_to_have
    hall_cnt = 0
    for statement in tqdm.tqdm(all_statements, desc="hallucination"):
        if use_gpt:
            pass
        else:
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
            encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            # encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt') # no gpu
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            cos = torch.nn.CosineSimilarity(dim=0)
            cos_score = cos(sentence_embeddings[0], sentence_embeddings[1]).item()
            
            if cos_score < 0.5:
                hall_cnt += 1
            
    try:
        return hall_cnt / len(all_statements) * 100
    except ZeroDivisionError:
        return 0


def COMPREHENSIVENESS(query, pred, must_have, use_gpt=False, model=None, tokenizer=None, device=None):
    if len(must_have) == 0:
        return 0
    
    comp_cnt = 0
    for statement in tqdm.tqdm(must_have, desc="Comprehensiveness"):
        if use_gpt:
            pass
        else:
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            # encoded_input = tokenizer([pred, statement], padding=True, truncation=True, max_length=512, return_tensors='pt') # no gpu
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            cos = torch.nn.CosineSimilarity(dim=0)
            cos_score = cos(sentence_embeddings[0], sentence_embeddings[1]).item()
            if cos_score >= 0.5:
                comp_cnt += 1

    return comp_cnt / len(must_have) * 100
    

def response():
    pass


def vllm_infer(client, tokenizer, prompt, stop_seq, max_new_tokens=1024, cot=False, temperature=0.0):
    """
    Generates a single output for a given input prompt using the VLLM backend (offline mode).
    Returns the output text.

    Reference:

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param prompt: str, the prompt to generate from
    :param stop_seq: list, the stop sequence to use for generation
    :param max_new_tokens: int, the maximum number of tokens to generate
    :param cot: bool, whether to use chain-or-thought or not
    :param temperature: float, the temperature to use for sampling
    """

    response = client.generate(prompt, sampling_params=vllm.SamplingParams(
        # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=1.0,
        top_k=-1,
        top_p=1.0,
        temperature=temperature,
        stop=stop_seq,
        use_beam_search=False,
        max_tokens=max_new_tokens,
        logprobs=5
    ))

    def top_answer(logprob):
        top_token = max(logprob, key=logprob.get)
        output_text = tokenizer.decode(top_token, skip_special_tokens=True)
        return output_text

    if len(response) > 0:
        return [r.outputs[0].text for r in response]

    if not cot:
        return top_answer(response[0].outputs[0].logprobs[0])
    else:
        return response[0].outputs[0].text

def tokenizer_param(tokenizer, target, shots=0, cot=False, task_type="mcq"):
    """
    Determines the maximum number of tokens to generate for a given prompt and target.
    Also determines the stop sequence to use for generation.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param target: str, the target to generate
    :param shots: int, the number of shots to use for few-shot learning
    :param cot: bool, whether to use chain-or-thought or not
    :param task_type: str, the type of answer to generate (mcq or open)
    """
    max_new_tokens = len(tokenizer(target, add_special_tokens=True)['input_ids'])
    stop_seq = [tokenizer.eos_token, tokenizer.pad_token, "###"]

    if not cot and task_type == "mcq":
        max_new_tokens = len(tokenizer(target[0], add_special_tokens=False)['input_ids'])
        if shots > 0:
            max_new_tokens += 8
    if cot:
        max_new_tokens = 1024

    return max_new_tokens, stop_seq

def extract_long_form_answer(pred, use_prompt=True):
    """
    提取预测中的长篇回答。

    参数:
    - pred (str): 预测的文本。
    - use_prompt (bool): 是否使用提示模式。如果为 True，则查找并提取 "Long-Form Answer" 部分；否则直接返回预测内容。

    返回:
    - str: 提取的长篇回答或预测文本。
    """
    all_output = None
    long_form_answer = None

    if use_prompt:
        print("Use prompt all pred:\n----------------------------------------", pred)
        print("\n----------------------------------------")

        # 查找 "Long-Form Answer:" 的位置
        long_form_start = pred.find("Long-Form Answer")

        # 如果找到了，提取相应内容
        if long_form_start != -1:
            long_form_start += len("Long-Form Answer:")  # 更新起始位置
            long_form_end = pred.find("END", long_form_start)  # 查找 "END" 的位置

            if long_form_end != -1:
                long_form_answer = pred[long_form_start:long_form_end].strip()  # 提取到 "END"
                all_output = pred[:long_form_end].strip()  # 提取到 "END"
            else:
                long_form_answer = pred[long_form_start:].strip()  # 如果没有找到 "END"，提取到结尾
        else:
            raise ValueError("长篇回答未找到。")
        
        print("long_form_answer:\n----------------------------------------", long_form_answer)
        print("\n----------------------------------------")
        return long_form_answer, all_output
    else:
        print("no prompt pred:", pred)
        return pred, all_output


def main():
    # check Evaluation Metrics
    """
    query = "Alright so I dont know much about Lexapro would you tell me more about it?"
    answer = "Escitalopram, sold under the brand names Lexapro and Cipralex, is an antidepressant of the SSRI (selective serotonin reuptake inhibitors) class. It is a medication for major depressive disorder and several types of anxiety disorders. It is considered an effective and well-tolerated antidepressant. The benefits of Lexapro for treating depression occur within a few weeks, but it can take about 1 to 2 months before you feel its full effects.\nLike other SSRIs, side effects include headache, nausea, sleepiness, ejaculation disorder, and insomnia. The FDA had published a black box warning for Escitalopram and other antidepressants, alerting for an increased risk of suicidal thinking and behavior in children, adolescents, and young adults. Therefore, Lexapro is not approved for use in pediatric patients less than 12 years of age."
    pred = "Lexapro is a medication that belongs to a class of drugs called selective serotonin reuptake inhibitors (SSRIs).Lexapro is primarily used to treat depression and anxiety disorders.It may take a few weeks for Lexapro to take effect, so it is important to be patient and continue taking the medication as prescribed by your healthcare provider.It is also important to discuss any potential side effects with your doctor.Lexapro can cause some side effects, but not everyone experiences them.Remember, everyone's response to medication can vary, so it's important to work closely with your healthcare provider to determine if Lexapro is right for you."
    must_have = ["Escitalopram is an antidepressant of the SSRI (Selective serotonin reuptake inhibitors) class","Escitalopram is sold under the brand names Lexapro and Cipralex","Side effects of Escitalopram include GI symptoms such as nausea, diarrhoea, constipation","Side effects of Escitalopram include headache","Side effects of Escitalopram include ejaculation disorder","The benefits of Lexapro for treating depression occurs within a few weeks","Side effects of Escitalopram include sleepiness","Side effects of Escitalopram include insomnia","The FDA had published a black box warning regarding Escitalopram, alerting for an increased risk of suicidal thinking and behavior in children","The FDA had published a black box warning for Escitalopram, alerting for an increased risk of suicidal thinking and behavior in adolescents and young adults"," Lexapro is not approved for use in pediatric patients less than 12 years of age."]
    nice_to_have = ["Escitalopram is a medication for major depressive disorder","Escitalopram is a medication for several types of anxiety disorders","Escitalopram is considered an effective and well-tolerated antidepressant"]
    """
    
    # load NLI model - gpt4
    # rougel_score = rougel(pred, answer)
    # hall_score = hallucination(query, pred, must_have, nice_to_have)
    # comp_score = comprehensiveness(query, pred, must_have)
    # 0.0 / 18.18
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="dmis-lab/selfbiorag_7b") # mistralai/Mistral-7B-v0.1, BioMistral/BioMistral-7B, meta-llama/Llama-2-7b-hf, dmis-lab/selfbiorag_7b, epfl-llm/meditron-7b
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default="/ssd0/minbyul/cache/") # need change
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    parser.add_argument("--sampling_trials",  type=int, default=5,
                        help="sampling_trials to derive sampled predictions")
    parser.add_argument("--use_gpt", action="store_true", help="use gpt-4 with openai key")
    parser.add_argument("--eval_data", type=str, default="")
    parser.add_argument('--wodata_name', type=str, default="")
    parser.add_argument('--data_size', type=str, default="")
    parser.add_argument('--after_dpo', action="store_true")
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    
    # 添加prompt参数，默认值为False
    parser.add_argument('--use_prompt', action='store_true', help='Enable or disable prompt, default is False')
    parser.add_argument('--few_shot', type=int, default=0, help="Number of few-shot examples, default is 0")

    args = parser.parse_args()


    # 添加逻辑来处理 use_prompt 和 few_shot
    if not args.use_prompt:
        # 如果没有传递 use_prompt，few_shot 设置为 0 并提示
        args.few_shot = 0
        print("No 'use_prompt' provided, setting 'few_shot' to 0.")
    else:
        # 如果有 use_prompt，检查 few_shot 的值
        if args.few_shot not in [1, 3]:
            # 如果 few_shot 不是 1 或 3，将其设置为 1 并提示
            args.few_shot = 1
            print("'few_shot' must be 1 or 3. Setting 'few_shot' to 1 by default.")
        
    pprint.pprint(vars(args))
    
    if not os.path.exists("./alignment-handbook/predictions"):
        os.mkdir("./alignment-handbook/predictions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_name_or_path.split("/")[1]

    if "meditron" in args.model_name_or_path.lower() or "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    else:
        model = LLM(model=args.model_name_or_path, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        
    # load prediction and dataset
    
    prompt1 = ""
    prompt2 = f""" 
# Task: You are a helpful assistant. Step-by-Step Thinking for Structured Medical Question Answering.

## General Instructions:
- Generate detailed and structured medical responses based on the given medical question. Answers should be grounded in current medical knowledge, covering all key aspects of the question.
- Ensure the answer includes background, etiology, symptoms, diagnosis, treatment, and prevention.
- The answer should be logically organized and provide accurate, comprehensive medical information.

## Task Instructions:
- Generate a comprehensive response based on the input question. The response should cover everything from background information to diagnosis and treatment recommendations, ensuring a structured and coherent output.
- The answer should address as many aspects of the medical question as possible, considering risk factors, complications, and related medical conditions.
- Consider the relationship between diseases and medications.
- Do not output duplicate content.
- Each thought process should not exceed 200 words.

## Output Structure:
- The output should follow the structured template below to ensure the completeness and professionalism of the medical response.
- Please ensure that the output contains Long-Form Answer

## Chain of Thought:
### 1. Understand the Question:
- {{Explain the background or definition of the medical issue. Provide a brief description of basic concepts and possibly affected systems or organs.}}
- {{Identify and define key medical terms and concepts.}}
- {{Clarify the specific information or details requested.}}

### 2. Recall Relevant Medical Knowledge:
- {{Retrieve information related to the disease, medication, or procedure.}}
- {{Consider anatomy, physiology, pathology, pharmacology, and current medical guidelines.}}

### 3. Analyze Medical Information:
- {{Combine 1. understanding the question and 2. relevant medical knowledge to connect the issue with pertinent medical knowledge using clinical reasoning.}}
- {{Consider possible explanations, mechanisms, or interactions.}}

### 4. Assess Impacts and Considerations:
- {{Evaluate any risks, side effects, or contraindications.}}
- {{Consider specific patient factors (age, comorbidities, allergies).}}

### 5. Provide Additional Relevant Information:
- {{Include important details that help in understanding.}}
- {{Mention any exceptions, alternative options, or preventive measures.}}

### 6. Suggest Follow-Up Steps or Actions:
- {{If necessary, recommend consulting a healthcare professional.}}
- {{Advise on monitoring, follow-up, or further evaluation.}}

### 7. Reference Reliable Sources:
- {{Base responses on evidence from authoritative medical texts or guidelines.}}
- {{Cite clinical studies, professional organizations, or regulatory agency information.}}

### 8.Long-Form Answer:
- {{Combine the above reasoning to accurately and comprehensively answer the question. Provide a "long-form answer" that contains 400-500 words. The word count must not be less than 400 words.}} 
### END

-{{Please end the output here.}}

Please refer to the following questions, along with examples of chain of thought and long-form answers.

Question: What is the relationship between Noonan syndrome and polycystic renal disease?

Chain of Thought:
1. Understand the Question:
Noonan syndrome is a genetic disorder characterized by distinct facial features, short stature, heart defects, and developmental delays. It affects multiple systems in the body, including the cardiovascular, musculoskeletal, and endocrine systems. Polycystic renal disease, particularly autosomal dominant polycystic kidney disease (ADPKD), is a genetic condition leading to the formation of numerous cysts in the kidneys, resulting in kidney enlargement and impaired function. The question seeks to explore the potential relationship between these two conditions, particularly any shared genetic or pathological mechanisms.

2. Recall Relevant Medical Knowledge:
Noonan syndrome is primarily caused by mutations in genes involved in the RAS-MAPK signaling pathway, particularly the PTPN11 gene. It affects approximately 1 in 1,000 to 1 in 2,500 births. On the other hand, polycystic renal disease is commonly caused by mutations in the PKD1 or PKD2 genes. ADPKD has a prevalence of about 1 in 400 to 1 in 1,000. Understanding the genetic basis and clinical manifestations of both conditions is crucial for identifying potential links between them.

3. Analyze Medical Information:
The relationship between Noonan syndrome and polycystic renal disease may stem from shared genetic pathways or phenotypic associations. Some studies suggest that patients with Noonan syndrome exhibit renal anomalies, including renal agenesis or structural abnormalities, although true polycystic kidney disease is less commonly reported. This indicates a potential overlap in genetic vulnerabilities that could lead to renal pathologies in Noonan syndrome patients. The mechanisms may involve disruptions in cellular signaling pathways that are pivotal for kidney development and function.

4. Assess Impacts and Considerations:
Patients with Noonan syndrome may have additional comorbidities that can influence renal health, such as hypertension or congenital heart defects, which could complicate the presentation of renal disease. Conversely, individuals with polycystic kidney disease are at risk of hypertension and kidney failure, potentially impacting their overall health and necessitating careful monitoring. Genetic counseling may be beneficial for families with a history of either condition to better understand the risks and implications of genetic inheritance.

5. Provide Additional Relevant Information:
While there is limited direct evidence linking Noonan syndrome and ADPKD, awareness of renal complications in Noonan syndrome patients is important for clinicians. Furthermore, certain genetic syndromes may predispose individuals to multiple anomalies, making regular screenings for renal function essential in affected individuals.

6. Suggest Follow-Up Steps or Actions:
For individuals diagnosed with Noonan syndrome, it is advisable to perform regular renal assessments, including ultrasound examinations to check for any renal structural anomalies. Genetic counseling can provide insights into the risks of polycystic kidney disease and the implications for family planning. Patients should be educated on signs of kidney dysfunction, such as changes in urination patterns, hypertension, or abdominal pain.

7. Reference Reliable Sources:
Sources for this information include clinical guidelines from the National Kidney Foundation, the American Academy of Pediatrics, and recent genetic studies published in peer-reviewed journals regarding the genetics of Noonan syndrome and polycystic kidney disease.

8. Long-Form Answer:
Noonan's syndrome is an eponymic designation that has been used during the last 8 years to describe a variable constellation of somatic and visceral congenital anomalies, which includes groups of patients previously referred to as male Turner's, female pseudo-Turner's and Bonnevie-Ullrich syndromes. It is now recognized that both sexes may show the stigmas of this condition and, unlike Turner's syndrome, there is no karyotype abnormality although there is often a familial pattern. The most commonly observed anomalies include webbing of the neck, hypertelorism, a shield-shaped chest and short stature. Congenital heart disease, principally pulmonary stenosis, and sexual infantilism often with cryptorchidism in the male subject are additional associated anomalies in this syndrome. Renal anomalies have been described rarely and usually consist of rotational errors, duplications and hydronephrosis. We report the first case of an infant who displayed many of the stigmas of Noonan's syndrome and also showed early evidence of frank renal failure secondary to renal dysplasia with cystic disease.
END
- Please ensure that the output contains Long-Form Answer
"""

    prompt3 = f""" # Task: You are a helpful assistant. Step-by-Step Thinking for Structured Medical Question Answering.

## General Instructions:
- Generate detailed and structured medical responses based on the given medical question. Answers should be grounded in current medical knowledge, covering all key aspects of the question.
- Ensure the answer includes background, etiology, symptoms, diagnosis, treatment, and prevention.
- The answer should be logically organized and provide accurate, comprehensive medical information.

## Task Instructions:
- Generate a comprehensive response based on the input question. The response should cover everything from background information to diagnosis and treatment recommendations, ensuring a structured and coherent output.
- The answer should address as many aspects of the medical question as possible, considering risk factors, complications, and related medical conditions.
- Consider the relationship between diseases and medications.
- Do not output duplicate content.
- Each thought process should not exceed 100 words.

## Output Structure:
- The output should follow the structured template below to ensure the completeness and professionalism of the medical response.
- Please ensure that the output contains Long-Form Answer

## Chain of Thought:
### 1. Understand the Question:
- {{Explain the background or definition of the medical issue. Provide a brief description of basic concepts and possibly affected systems or organs.}}
- {{Identify and define key medical terms and concepts.}}
- {{Clarify the specific information or details requested.}}

### 2. Recall Relevant Medical Knowledge:
- {{Retrieve information related to the disease, medication, or procedure.}}
- {{Consider anatomy, physiology, pathology, pharmacology, and current medical guidelines.}}

### 3. Analyze Medical Information:
- {{Combine 1. understanding the question and 2. relevant medical knowledge to connect the issue with pertinent medical knowledge using clinical reasoning.}}
- {{Consider possible explanations, mechanisms, or interactions.}}

### 4. Assess Impacts and Considerations:
- {{Evaluate any risks, side effects, or contraindications.}}
- {{Consider specific patient factors (age, comorbidities, allergies).}}

### 5. Provide Additional Relevant Information:
- {{Include important details that help in understanding.}}
- {{Mention any exceptions, alternative options, or preventive measures.}}

### 6. Suggest Follow-Up Steps or Actions:
- {{If necessary, recommend consulting a healthcare professional.}}
- {{Advise on monitoring, follow-up, or further evaluation.}}

### 7. Reference Reliable Sources:
- {{Base responses on evidence from authoritative medical texts or guidelines.}}
- {{Cite clinical studies, professional organizations, or regulatory agency information.}}

### 8.Long-Form Answer:
- {{Combine the above reasoning to accurately and comprehensively answer the question. Provide a "long-form answer" that contains 400-500 words. The word count must not be less than 400 words.}} 
### END

-{{Please end the output here.}}

Please refer to the following questions, along with examples of chain of thought and long-form answers.

Question: What is the relationship between Noonan syndrome and polycystic renal disease?

Chain of Thought:
1. 
Noonan syndrome is a genetic disorder characterized by distinct facial features, short stature, heart defects, and developmental delays. It affects multiple systems in the body, including the cardiovascular, musculoskeletal, and endocrine systems. Polycystic renal disease, particularly autosomal dominant polycystic kidney disease (ADPKD), is a genetic condition leading to the formation of numerous cysts in the kidneys, resulting in kidney enlargement and impaired function. The question seeks to explore the potential relationship between these two conditions, particularly any shared genetic or pathological mechanisms.

2. 
Noonan syndrome is primarily caused by mutations in genes involved in the RAS-MAPK signaling pathway, particularly the PTPN11 gene. It affects approximately 1 in 1,000 to 1 in 2,500 births. On the other hand, polycystic renal disease is commonly caused by mutations in the PKD1 or PKD2 genes. ADPKD has a prevalence of about 1 in 400 to 1 in 1,000. Understanding the genetic basis and clinical manifestations of both conditions is crucial for identifying potential links between them.

3. 
The relationship between Noonan syndrome and polycystic renal disease may stem from shared genetic pathways or phenotypic associations. Some studies suggest that patients with Noonan syndrome exhibit renal anomalies, including renal agenesis or structural abnormalities, although true polycystic kidney disease is less commonly reported. This indicates a potential overlap in genetic vulnerabilities that could lead to renal pathologies in Noonan syndrome patients. The mechanisms may involve disruptions in cellular signaling pathways that are pivotal for kidney development and function.

4. 
Patients with Noonan syndrome may have additional comorbidities that can influence renal health, such as hypertension or congenital heart defects, which could complicate the presentation of renal disease. Conversely, individuals with polycystic kidney disease are at risk of hypertension and kidney failure, potentially impacting their overall health and necessitating careful monitoring. Genetic counseling may be beneficial for families with a history of either condition to better understand the risks and implications of genetic inheritance.

5. 
While there is limited direct evidence linking Noonan syndrome and ADPKD, awareness of renal complications in Noonan syndrome patients is important for clinicians. Furthermore, certain genetic syndromes may predispose individuals to multiple anomalies, making regular screenings for renal function essential in affected individuals.

6. 
For individuals diagnosed with Noonan syndrome, it is advisable to perform regular renal assessments, including ultrasound examinations to check for any renal structural anomalies. Genetic counseling can provide insights into the risks of polycystic kidney disease and the implications for family planning. Patients should be educated on signs of kidney dysfunction, such as changes in urination patterns, hypertension, or abdominal pain.

7. 
Sources for this information include clinical guidelines from the National Kidney Foundation, the American Academy of Pediatrics, and recent genetic studies published in peer-reviewed journals regarding the genetics of Noonan syndrome and polycystic kidney disease.

8. Long-Form Answer:
Noonan's syndrome is an eponymic designation that has been used during the last 8 years to describe a variable constellation of somatic and visceral congenital anomalies, which includes groups of patients previously referred to as male Turner's, female pseudo-Turner's and Bonnevie-Ullrich syndromes. It is now recognized that both sexes may show the stigmas of this condition and, unlike Turner's syndrome, there is no karyotype abnormality although there is often a familial pattern. The most commonly observed anomalies include webbing of the neck, hypertelorism, a shield-shaped chest and short stature. Congenital heart disease, principally pulmonary stenosis, and sexual infantilism often with cryptorchidism in the male subject are additional associated anomalies in this syndrome. Renal anomalies have been described rarely and usually consist of rotational errors, duplications and hydronephrosis. We report the first case of an infant who displayed many of the stigmas of Noonan's syndrome and also showed early evidence of frank renal failure secondary to renal dysplasia with cystic disease.
END
- Please continue to answer that all 8 steps have been completed
- Please ensure that the output contains Long-Form Answer and generate content of no less than 300 words
"""
    eval_name = args.eval_data
    train_examples = []
    use_prompt = args.use_prompt
    few_shot = args.few_shot

    filename = f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling_{use_prompt}_{few_shot}.jsonl_tmp"
    write_name = f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling_{use_prompt}_{few_shot}.jsonl_tmp"

    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                train_examples.append(json.loads(line))
    else:
        filename = f"./MedLFQA/{eval_name}_test_MedLFQA.jsonl"

        with open(filename, 'r') as fp:
            for idx, line in enumerate(fp.readlines()):
                # 如果 eval_name 是 live_qa 并且当前是前 few_shot 行，跳过
                if eval_name == "live_qa" and idx < args.few_shot:
                    continue
                
                # 如果 eval_name 不是 live_qa 并且超过 100 行，停止读取
                if eval_name != "live_qa" and idx >= 100:
                    break

                train_examples.append(json.loads(line))

    # for inst_idx, inst in enumerate(train_examples):
        # 确保 'Question' 字段存在
        # if 'Question' in inst:
        #     # 获取问题并去掉前后的空白字符
        #     question = inst['Question'].strip()
        #     # 输出问题及其索引
        #     print(f"问题 {inst_idx}: {question}")
        # else:
        #     print(f"实例 {inst_idx} 缺少 'Question' 字段")

    for inst_idx ,inst in enumerate(train_examples):
        # query
        question = inst['Question'].strip()
        
        if args.use_prompt==True:
            query = prompt2 + question
            output_max_length = 4096
            system = "You are a professional doctor; please follow the instructions, think step by step, and provide a comprehensive and accurate long-form medical answer. The long-form answer should be no less than 400 words."

        else:
            query = prompt1 + question
            output_max_length = 512
            system = "You are a helpful assistant."

        if "selfbiorag" in args.model_name_or_path:
            query = prompt3 + question
        
        print(inst_idx, "question:", question)
        # print(inst_idx)
        
        answer = inst['Free_form_answer']

        # add question mark
        if query[-1] != "?":
            query += "?"

        if args.use_prompt:
            query += "Chain of Thought:"

        if "tmp" in filename and "sample_predictions" in inst and "prediction_scores" in inst:
            continue

        # ten generation to make preference collections - check hallucination
        sample_predictions = []
        all_output_p = []
        inf_time = []
 
        if "meditron" in args.model_name_or_path.lower() or "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower() and "instruct" not in args.model_name_or_path.lower():
            print("1",model_name)
            # 记录开始时间
            start_time = time.time()
            input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
            output = model.generate(input_ids, max_length=output_max_length, no_repeat_ngram_size=2, do_sample=False, top_p=1.0, repetition_penalty=args.repetition_penalty).to(device)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            pred = response[len(query):].strip()
            # 记录结束时间
            end_time = time.time()

            # 计算推理总耗时
            inference_time = end_time - start_time
            inf_time.append(inference_time)

            # 输出总耗时
            print(f"推理总耗时: {inference_time:.4f} 秒")

            try:
                long_form_answer, all_output = extract_long_form_answer(pred, use_prompt)
                sample_predictions.append(long_form_answer)
                all_output_p.append(all_output)
            except ValueError as e:
                print(e)

            
        elif "llama3-8b-instruct" == model_name:
            print("2",model_name)
            # 记录开始时间
            start_time = time.time()
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = model.generate(input_ids, max_new_tokens=512, eos_token_id=terminators, do_sample=False, temperature=0.0, top_p=0.9)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = response[len(query):].strip()
            # 记录结束时间
            end_time = time.time()

            # 计算推理总耗时
            inference_time = end_time - start_time
            inf_time.append(inference_time)

            # 输出总耗时
            print(f"推理总耗时: {inference_time:.4f} 秒")

            try:
                long_form_answer, all_output = extract_long_form_answer(pred, use_prompt)
                sample_predictions.append(long_form_answer)
                all_output_p.append(all_output)
            except ValueError as e:
                print(e)

        elif "gpt" in args.model_name_or_path.lower():
            print("3",model_name)

            client = OpenAI()

            if args.few_shot == 0:
                system = "You are a helpful assistant."
            else:
                system = "You are a professional doctor; please follow the instructions, think step by step, and provide a comprehensive and accurate long-form medical answer. The long-form answer should be no less than 400 words."

            # 记录开始时间
            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=1
            )

            
            pred = response.choices[0].message.content.strip()

            # 记录结束时间
            end_time = time.time()

            # 计算推理总耗时
            inference_time = end_time - start_time
            inf_time.append(inference_time)

            # 输出总耗时
            print(f"推理总耗时: {inference_time:.4f} 秒")

            try:
                long_form_answer, all_output = extract_long_form_answer(pred, use_prompt)
                sample_predictions.append(long_form_answer)
                all_output_p.append(all_output)
            except ValueError as e:
                print(e)

        else:
            print("4",model_name,"output_max_length:",output_max_length)

            if "selfbiorag" in args.model_name_or_path and args.use_prompt:
#                 query += "[No Retrieval]"
#                 query_step_1 = query + """Please only generate the results of the first 4 steps Chain of Thought:
# 1. 2.  3.  4. """

#                 print("query_step_1",query_step_1)

#                 sampling_params1 = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=output_max_length)
#                 preds1 = model.generate([query_step_1], sampling_params1)
#                 pred1 = preds1[0].outputs[0].text.strip()
#                 print("step1:------------------------------\n", pred1)

#                 query += pred1
#                 query_step_2 = query + """Please continue generating results from 5 to 8 Chain of Thought:
# 5. 6. 7.  8. """

#                 print("query_step_2",query_step_2)

#                 sampling_params2 = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=output_max_length)
#                 preds2 = model.generate([query_step_2], sampling_params2)
#                 pred2 = preds2[0].outputs[0].text.strip()
#                 print("step2:--------------------------------\n", pred2)

#                 pred = pred1 + pred2

                # query += "[No Retrieval]"
                # 记录开始时间
                start_time = time.time()
                print("query=====================",query)
                
                sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=output_max_length)
                preds = model.generate([query], sampling_params)
                pred = preds[0].outputs[0].text.strip()
                print("pred:\n----------------------------------\n" ,pred)

                # 记录结束时间
                end_time = time.time()

                # 计算推理总耗时
                inference_time = end_time - start_time
                inf_time.append(inference_time)

                # 输出总耗时
                print(f"推理总耗时: {inference_time:.4f} 秒")


            else:
                if "selfbiorag" in args.model_name_or_path:
                    query += "[No Retrieval]"

                # 记录开始时间
                start_time = time.time()
                print("query=====================",query)
                
                sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=output_max_length)
                preds = model.generate([query], sampling_params)
                pred = preds[0].outputs[0].text.strip()
                print("pred:\n----------------------------------\n" ,pred)

                # 记录结束时间
                end_time = time.time()

                # 计算推理总耗时
                inference_time = end_time - start_time
                inf_time.append(inference_time)

                # 输出总耗时
                print(f"推理总耗时: {inference_time:.4f} 秒")

            try:
                long_form_answer, all_output = extract_long_form_answer(pred, use_prompt)
                sample_predictions.append(long_form_answer)
                all_output_p.append(all_output)
            except ValueError as e:
                print(e)
                  
        inst['sample_predictions'] = sample_predictions
        inst['inference_time'] = inf_time

        if all_output is not None:
            inst['all_output'] = all_output

        # load bleurt model
        bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
        bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

        # load nli model for hallucination and comprehensiveness
        if not args.use_gpt:
            nli_model = AutoModel.from_pretrained('gsarti/biobert-nli', max_length=512).to(device)
            # nli_model = AutoModel.from_pretrained('gsarti/biobert-nli', max_length=512) # no gpu
            nli_tokenizer = AutoTokenizer.from_pretrained('gsarti/biobert-nli') #gsarti/biobert-nli

        prediction_scores = []
        for sample_idx,sample in enumerate(sample_predictions):
            sample = sample.strip()
            rouge1, rouge2, rougel = ROUGESCORE(sample, inst['Free_form_answer']) # higher better
            bleurt = BLEURT(sample, inst['Free_form_answer'], model=bleurt_model, tokenizer=bleurt_tokenizer) # higher better
            bs_p, bs_r, bs_f1 = BERTSCORE(sample, inst['Free_form_answer']) # higher better
            
            # hallucination and comprehensiveneess with gpt-4 or biobert-nli model
            hall_score = HALLUCINATION(inst["Question"], sample, inst["Must_have"], inst["Nice_to_have"], use_gpt=args.use_gpt, model=nli_model, tokenizer=nli_tokenizer, device=device) # lower better
            comp_score = COMPREHENSIVENESS(inst["Question"], sample, inst["Must_have"], use_gpt=args.use_gpt, model=nli_model, tokenizer=nli_tokenizer, device=device) # higher better

            # 24.05.31 update - fluency
            fluency_score = sentence_gleu([answer], sample)

            prediction_scores.append({"idx":sample_idx, "rouge1_p":round(rouge1.precision, 4), "rouge1_r": round(rouge1.recall, 4), "rouge1_f1": round(rouge1.fmeasure, 4), "rouge2_p": round(rouge2.precision, 4), "rouge2_r": round(rouge2.recall, 4), "rouge2_f1": round(rouge2.fmeasure, 4), "rougel_p": round(rougel.precision, 4), "rougel_r": round(rougel.recall, 4), "rougel_f1": round(rougel.fmeasure, 4), "bleurt": round(bleurt[0], 4), "bert_score_p": round(bs_p, 4), "bert_score_r": round(bs_r, 4), "bert_score_f1": round(bs_f1, 4), "hallucination": hall_score, "comprehensive": comp_score, "fluency": round(fluency_score, 4)})
        
        inst['prediction_scores'] = prediction_scores

        if (inst_idx+1) % 1 == 0:
            print (inst)
           
            with open(write_name, "w") as outfile:
                for inst in train_examples:
                    outfile.write(json.dumps(inst))
                    outfile.write("\n")

    with open(write_name, "w") as outfile:
        for inst in train_examples:
            outfile.write(json.dumps(inst))
            outfile.write("\n")
            

if __name__ == "__main__":
    main()