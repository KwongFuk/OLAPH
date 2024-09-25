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

from openai import OpenAI

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
    parser.add_argument('--max_new_tokens', type=int, default=512)
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
    args = parser.parse_args()

    if not os.path.exists("./alignment-handbook/predictions"):
        os.mkdir("./alignment-handbook/predictions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model_name = "gpt-3.5"
    # load prediction and dataset
    
    prompt = f""" 
# Task: You are a helpful assistant. Step-by-Step Thinking for Structured Medical Question Answering.

## General Instructions:
- Generate detailed and structured medical responses based on the given medical question. Answers should be grounded in current medical knowledge, covering all key aspects of the question.
- Ensure the answer includes background, etiology, symptoms, diagnosis, treatment, and prevention.
- The answer should be logically organized and provide accurate, comprehensive medical information.

## Task Instructions:
- Generate a comprehensive response based on the input question. The response should cover everything from background information to diagnosis and treatment recommendations, ensuring a structured and coherent output.
- The answer should address as many aspects of the medical question as possible, considering risk factors, complications, and related medical conditions.
- Consider the relationship between diseases and medications.

## Output Structure:
- The output should follow the structured template below to ensure the completeness and professionalism of the medical response.

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
Noonan syndrome is a congenital genetic disorder characterized by various clinical features, including distinctive facial characteristics, short stature, congenital heart defects, and developmental delays. This syndrome primarily arises from mutations in genes that impact the RAS-MAPK signaling pathway, with PTPN11 being the most frequently implicated. Noonan syndrome has an estimated prevalence of about 1 in 1,000 to 1 in 2,500 births, affecting multiple organ systems.

On the other hand, polycystic renal disease, particularly autosomal dominant polycystic kidney disease (ADPKD), is a hereditary condition marked by the development of numerous cysts in the kidneys, leading to renal enlargement and progressive dysfunction. This condition is commonly caused by mutations in either the PKD1 or PKD2 genes, with a prevalence rate of approximately 1 in 400 to 1 in 1,000 individuals.

Exploring the potential relationship between these two syndromes reveals a complex interplay. While Noonan syndrome is not directly associated with polycystic kidney disease, there are instances of renal anomalies observed in patients with Noonan syndrome. Reports indicate that some individuals may experience renal structural issues, including renal agenesis or varying degrees of renal dysplasia, although actual cases of polycystic kidney disease are relatively rare.

The potential connection may arise from overlapping genetic pathways or developmental disruptions affecting renal function. Given the significant roles of cellular signaling in both syndromes, disruptions in the development and function of the kidneys in Noonan syndrome patients may lead to atypical renal findings. Moreover, it is crucial to consider the overall health of patients with Noonan syndrome, as they may have comorbidities such as hypertension or congenital heart defects that can further complicate renal health.

For patients with Noonan syndrome, regular renal evaluations are essential, including ultrasounds to assess kidney structure and function. Genetic counseling is advisable for families affected by either condition to understand the implications of genetic inheritance and the associated risks of renal disease.

In summary, while a direct causal relationship between Noonan syndrome and polycystic renal disease has not been established, awareness of potential renal complications in Noonan syndrome patients is vital for comprehensive patient care. Regular monitoring and genetic counseling are key to managing these complex conditions and ensuring optimal health outcomes.

"""
    
    eval_name = args.eval_data
    train_examples = []
    
    if args.wodata_name:
        if args.after_dpo:
            filename = f"./alignment-handbook/predictions/pdata_{model_name}_dpo-step{args.iteration}_wo-{args.wodata_name}_{eval_name}_sampling.jsonl_tmp"
            write_name = f"./alignment-handbook/predictions/pdata_{model_name}_dpo-step{args.iteration}_wo-{args.wodata_name}_{eval_name}_sampling.jsonl_tmp"
        else:
            filename = f"./alignment-handbook/predictions/pdata_{model_name}_wo-{args.wodata_name}_{eval_name}_sampling.jsonl_tmp"
            write_name = f"./alignment-handbook/predictions/pdata_{model_name}_wo-{args.wodata_name}_{eval_name}_sampling.jsonl_tmp"
    else:
        if args.after_dpo:
            filename = f"./alignment-handbook/predictions/pdata_{model_name}_dpo-step{args.iteration}_{eval_name}_sampling.jsonl_tmp"
            write_name = f"./alignment-handbook/predictions/pdata_{model_name}_dpo-step{args.iteration}_{eval_name}_sampling.jsonl_tmp"
        else:
            filename = f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling.jsonl_tmp"
            write_name = f"./alignment-handbook/predictions/pdata_{model_name}_{eval_name}_sampling.jsonl_tmp"

    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                train_examples.append(json.loads(line))
    else:
        filename = f"./MedLFQA/{eval_name}_test_MedLFQA.jsonl"
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                train_examples.append(json.loads(line))
    
    
    for inst_idx ,inst in enumerate(train_examples):
        # query
        Question = inst['Question']
        query = prompt + "\n" + Question
        answer = inst['Free_form_answer']

        # add question mark
        if query[-1] != "?":
            query += "?"

        if "tmp" in filename and "sample_predictions" in inst and "prediction_scores" in inst:
            continue

        # ten generation to make preference collections - check hallucination
        sample_predictions = []

        if "gpt" in args.model_name_or_path.lower():
            print("gpt:")
            print("query: ", query)
            
            client = OpenAI()
            # zero-sot no prompt
            # response = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {
            #             "role": "user",
            #             "content": query
            #         }
            #     ],
            #     temperature=1
            # )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional doctor; please follow the instructions, think step by step, and provide a comprehensive and accurate long-form medical answer. The long-form answer should be no less than 400 words."},
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=1
            )
            
            pred = response.choices[0].message.content.strip()
            print("pred:",pred)

            # 查找 "Long-Form Answer:" 的位置
            long_form_start = pred.find("Long-Form Answer")
    
            # 如果找到了，提取相应内容
            if long_form_start != -1:
                long_form_answer = pred[long_form_start + len("Long-Form Answer:"):].strip()
            else:
                raise ValueError("长篇回答未找到。")
            print("long_form_answer:",long_form_answer)

            sample_predictions.append(long_form_answer)
        
        inst['sample_predictions'] = sample_predictions

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

        if (inst_idx+1) % 5 == 0:
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