import random
import pandas as pd
import os

import backoff
#from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from openai import OpenAIError,OpenAI
#import openai
import numpy as np

from parser_utils import *
import openai
import torch
import tiktoken
from ollama import Client as OllamaClient


def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    _tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        }
        | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    )
    return _tokenizer



class LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id
                 ):
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 1 else "he"
        self.debug = sampling_parameters.debug
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        if communication:
            self.generator_prompt_template = df['prompt'][1].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        else:
            self.generator_prompt_template = None

        self.communication = communication
        self.cot = cot
        self.source = source
        self.model = None
        self.tokenizer = None
        #self.lm_id = lm_id
        self.chat = True #'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id or 'chat' in lm_id
        self.lm_id = lm_id#'gemma3'#'gpt-oss:20b'#'openai/gpt-4o-mini'#'gpt-oss:20b' #"ollama_chat/gemma3" #'gemma3:4b' #'router-bert-0.9' #'gpt-oss:20b' #'router-bert-0.9' #lm_id
        self.strong_model = "gemini/gemini-2.0-flash" #'openai/gpt-4o-mini' #"gemini/gemini-2.0-flash"
        self.weak_model = "ollama_chat/gemma3"
        self.OPENAI_KEY = None
        self.total_cost = 0
        self.ollama_client = None
        self.openai_client = None
        if 'gpt-oss' in self.lm_id:
            self.cot = False
        if self.source == "openai":
            # client = AzureOpenAI(
            #     ### Enter your OpenAI API key here
            # )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
            _tok = get_tokenizer()
        else:
            raise ValueError("invalid source")

        # ----------------------------- Engine builder -----------------------------
        def lm_engine(source: str, lm_id: str):
            """Returns a callable: (prompt, sampling_params, is_check=False) -> (outputs, usage_dollars, usage_tokens)
            outputs = [generated_text, generated_reasoning(optional)]
            """

            # OpenAI + Ollama clients are created lazily to avoid unnecessary env lookups.
            def _get_openai(base_url=None, api_key=None):
                if self.openai_client is None:
                    if base_url is None:
                        self.openai_client = OpenAI()
                    else:
                        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)
                return self.openai_client

            def _get_ollama():
                if self.ollama_client is None:
                    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                    self.ollama_client = OllamaClient(host=host)
                return self.ollama_client

            def _is_ollama_model(model_id: str) -> bool:
                return any(
                    model_id.lower().startswith(prefix) for prefix in ["gpt-oss", "gemma", "gemma3", "gemma2"]) \
                    or model_id.lower() in {"gpt-oss:20b", "gemma3:4b", "gemma2:9b"}

            @backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
            def _generate(prompt, sampling_params, is_check: bool = False):
                usage_tokens = 0
                generated_reasoning = ""

                if source == 'openai':
                    # Route to Ollama for local models, OpenAI Responses for 4o-mini etc.
                    if _is_ollama_model(lm_id):
                        if "gpt-oss" in lm_id:
                            ollama = _get_ollama()
                            think_effort = "medium"
                            if isinstance(self.cot, str) and self.cot not in ("none", "true"):
                                think_effort = self.cot
                            resp = ollama.chat(model=lm_id, messages=prompt,
                                               think=(think_effort if not is_check else "medium"),
                                               options=sampling_params)
                            # Ollama python client returns: { 'message': {'content': str, 'thinking': str}, ... }
                            generated_text = resp.message.content
                            generated_reasoning = getattr(resp.message, 'thinking', '') or resp.message.get(
                                'thinking', '') if hasattr(resp, 'message') else ''
                        elif "gemma" in lm_id:
                            ollama = _get_openai(base_url="http://163.152.163.21:11434/v1", api_key="ollama")
                            resp = ollama.chat.completions.create(model=lm_id, messages=prompt, **sampling_params)
                            generated_text = [resp.choices[i].message.content for i in range(sampling_params['n'])][
                                0]
                    else:
                        client = _get_openai()
                        # Use Responses API (works for gpt-4o-mini, gpt-5-nano, etc.)
                        resp = client.responses.create(model=lm_id, input=prompt)
                        generated_text = resp.output_text

                    # Estimate tokens (best-effort, local tokenizer)
                    #print("generated reasoning:",generated_reasoning)
                    #print("generated text:",generated_text)
                    _tok = get_tokenizer()

                    last_user = prompt[-1]['content'] if isinstance(prompt, list) and prompt else ""
                    usage_tokens = len(_tok.encode(last_user)) + len(_tok.encode(generated_text)) + len(
                        _tok.encode(generated_reasoning))
                    print("input token:", len(_tok.encode(last_user)))
                    print("output token:", len(_tok.encode(generated_text)))
                    print('reasoning:',len(_tok.encode(generated_reasoning)))

                    print("Answer:\n",generated_text)
                    print("Reason:\n", generated_reasoning)
                    self.total_cost += usage_tokens
                    print("total cost:", self.total_cost)
                    return [generated_text, generated_reasoning], usage_tokens #[generated_text, generated_reasoning], usage_tokens

                elif source == 'huggingface':
                    # Simple HF text generation (no separate reasoning)
                    if isinstance(prompt, list):
                        # Convert chat format to a flat prompt
                        flat = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in prompt])
                    else:
                        flat = str(prompt)
                    input_ids = self.hf_tokenizer(flat, return_tensors="pt").input_ids.to('cuda')
                    prompt_len = input_ids.shape[-1]
                    with torch.inference_mode():
                        out = self.hf_model.generate(input_ids, pad_token_id=self.hf_tokenizer.eos_token_id,
                                                     **self.sampling_params)
                    text = self.hf_tokenizer.batch_decode(out.sequences[:, prompt_len:])
                    text = [s.strip() for s in text]
                    text = [s[:-4] if s.endswith('</s>') else s for s in text]
                    gen = text[0] if text else ""
                    _tok = get_tokenizer()
                    try:
                        usage_tokens = len(_tok.encode(flat)) + len(_tok.encode(gen))
                    except Exception as e:
                        print(e)
                        usage_tokens = 0
                    self.total_cost += usage_tokens
                    print("total cost:", self.total_cost)


                    return [gen, ""], usage_tokens

                else:
                    raise ValueError("invalid source")

            return _generate
        # def lm_engine():
        #     """
        #     Initializes and configures the RouteLLM controller.
        #     """
        #
        #     def _get_ollama():
        #         if self.ollama_client is None:
        #             # http://163.152.163.21:11434/v1
        #             host = os.environ.get("OLLAMA_HOST", "http://163.152.163.21:11434")
        #             self.ollama_client = OllamaClient(host=host)
        #         return self.ollama_client
        #     # Configure the RouteLLM controller
        #     # This example uses GPT-4 as the strong model and a local Llama 3 model
        #     # served via Ollama as the weak model.
        #     # Make sure the Ollama server is running.
        #     # You can replace 'ollama_chat/llama3' with any other supported weak model.
        #     try:
        #         gemma3_remote = LLMConfig(
        #             provider_model="ollama/gpt-oss:20b",
        #             api_base="http://163.152.163.21:11434",  # External server IP; omit /v1 here.
        #             api_key="ollama"  # Ollama uses a dummy key.
        #         )
        #         self.client =  LLMClient({'gpt-oss': gemma3_remote})
        #         # self.client = openai.OpenAI(
        #         #     base_url="http://163.152.163.21:11434/v1",
        #         #     api_key="ollama"
        #         # )
        #
        #
        #     except Exception as e:
        #         print(f"Failed to initialize RouteLLM Controller: {e}")
        #         # Fallback or error handling
        #         return lambda prompt, sampling_params: (["Error: failed"], 0)
        #
        #     @backoff.on_exception(backoff.expo, OpenAIError)
        #     def _generate(prompt, sampling_params):
        #         """
        #         Generates text using the configured RouteLLM client.
        #         """
        #         usage = 0
        #         try:
        #             original_prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in prompt])
        #             # print("prompt:\n",original_prompt_text)
        #
        #             response = self.client.chat.completions.create(
        #                 model=self.lm_id,
        #                 messages=prompt,  # Assuming prompt is in OpenAI message format
        #                 **self.sampling_params,
        #             )
        #             # ollama = _get_ollama()
        #             # response = ollama.chat(model=self.lm_id, messages=prompt, think='medium',
        #             #                    options=sampling_params)
        #             # if self.debug:
        #             #     with open("LLM/chat_raw.json", 'a') as f:
        #             #         f.write(json.dumps(response.dict(), indent=4))
        #             #         f.write('\n')
        #
        #             generated_samples = [choice.message.content for choice in response.choices]
        #
        #             #usage = response.usage.prompt_tokens + response.usage.completion_tokens
        #
        #             last_user = prompt[-1]['content']
        #             generated_text = "\n".join(generated_samples)
        #             generated_reasoning = ""
        #             if 'gpt-oss' in self.lm_id:
        #                 generated_reasoning = getattr(response.message, 'thinking', '') or response.message.get('thinking', '') if hasattr(response, 'message') else ''
        #             #print("\nprompt:\n",last_user)
        #             #print("output:\n",generated_text)
        #             #print("\nreasoning:\n",generated_reasoning)
        #
        #             usage_tokens = len(_tok.encode(last_user)) + len(_tok.encode(generated_text)) + len(_tok.encode(generated_reasoning))
        #
        #             print("usage_token:",usage_tokens)
        #
        #             return generated_samples, usage_tokens
        #
        #         except OpenAIError as e:
        #             print(f"An OpenAI error occurred: {e}")
        #             raise e
        #         except Exception as e:
        #             print(f"An unexpected error occurred: {e}")
        #             # Fallback for errors not caught by backoff
        #             return [f"Error during generation: {e}"], 0
        #
        #     return _generate


        self.generator = lm_engine(self.source, self.lm_id)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None
        self.physics_memory = None


    def run_efficiency_weight(self, object_name, object_from):
        """
        Evaluates the relevance of an object to achieving the goal based on its interaction type and context.

        Parameters:
            object_name (str): The name of the object.
            object_from (str): The location where the object was found.
            object_interaction (str): The type of interaction possible with the object.

        Returns:
            float: A weight representing the object's relevance to the goal (1 for strong, 0.5 for medium,
            0 for low, -1 for none).
        """
        prompt = f'I do household with my friend {self.oppo_name}. ' \
                 f'Our goal is \'{self.goal_desc}\'. \n' \
                 f'I found {object_name} in {object_from}. ' \


        prompt += f"What relevance does this object \'{object_name}\' have to achieve the goal?  \n" \
                  f"A. Strong relevance means that the object is one of the target obejct or included in achieving the goal. \n" \
                  f"B. Medium relevance means that the object contributes to achieving the goal, but it is not critical. \n" \
                  f"C. Low relevance suggests that the object has a minimal impact on achieving the goal. \n" \
                  f"D. None relevance means that the object is not necessary to achieve the goal. \n"

        if self.cot:
            cot_prompt = prompt
            cot_prompt += "\nAnswer: Let's think step by step."

            chat_prompt = [{"role": "user", "content": cot_prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            cot_output = outputs[0]
            if self.debug:
                print("--------run_efficiency_weight--------")
                print("chat prompt:", cot_prompt)
                print("output:", cot_output)
                print("\n")

            prompt = "Choose from the three answers below. \n"
            prompt += f"A. Strong relevance. \n" \
                      f"B. Medium relevance. \n" \
                      f"C. Low relevance. \n" \
                      f"D. None relevance."

            prompt += "Output format: \n" \
                      "Answer: alphabet"

            chat_prompt = [{"role": "user", "content": cot_prompt},
                           {"role": "assistant", "content": cot_output},
                           {"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            output = outputs[0].replace('Answer: ', '')


        else:
            prompt += "Choose from the three answers below. \n"
            prompt += f"A. Strong relevance. \n" \
                      f"B. Medium relevance. \n" \
                      f"C. Low relevance. \n" \
                      f"D. None relevance."
            prompt += "Output format: \n" \
                      "Answer: alphabet"

            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            output = outputs[0].replace('Answer: ', '')

        # print(self.agent_name, 'Relevance', object_name, object_from, object_interaction, output)

        if 'A' == output[0]:
            # print('A')
            return 1
        elif 'B' == output[0]:
            # print('B')
            return 0.5
        elif 'C' == output[0]:
            # print('C')
            return 0
        elif 'D' == output[0]:
            # print('D')
            return -1
        else:
            if 'A' in output:
                # print('A')
                return 1
            elif 'B' in output:
                # print('B')
                return 0.5
            elif 'C' in output:
                # print('C')
                return 0
            else:
                # print('D')
                return -1

    def run_init(self, message):
        """
        Refines the given message to be more polite and concise using a generator model.

        Parameters:
            message (str): The original message to be rephrased.

        Returns:
            str: The rephrased message.
        """

        prompt = f"I'd like to forward the message below to my collaborator {self.oppo_name}. " \
                 f"Please make it a little kinder and brief.\n" \
                 f"Original message: {message}\n" \
                 f"Output format:\n" \
                 f"Message: \"your message\""
        chat_prompt = [{"role": "user", "content": prompt}]

        if self.debug:
            print("--------run_init(self, message):--------")
            print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0].replace('Message: ', '')
        if self.debug:
            print("output:", output)
            print("\n")

        return output

    def reset(self, rooms_name, goal_objects):
        self.rooms = rooms_name
        self.goal_desc = self.goal2description(goal_objects)
        self.total_cost = 0


    def goal2description(self, goals):  # {predicate: count}
        s = "Transport "
        r = None
        for object_name, count in goals.items():
            s += f"{count} {object_name}{'s' if count > 1 else ''}, "

        s = s[:-2] + f" to the bed."
        return s


    def parse_answer(self, available_actions, text):

        if "-" in text:
            text = text.split('-')[0][:-1]
        for i in range(len(available_actions)):
            if '-' in available_actions[i]:
                available_actions[i] = available_actions[i].split('-')[0][:-1]

        flags = 'AC'
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message:"):
                action = "send a message"
            if action.lower() in text.lower():
                return available_actions[i], flags
        sents = text.split('\n')  # Split by space
        words = []
        for sent in sents:
            words.extend(sent.split(' '))
        words = list(filter(None, words))  # Remove empty strings from the result

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in words or f"{option}," in words or f"{option}\n" in text.split(" ") or f"Option {option}" in text or f"({option})" in words or f"action {option}" in text or (len(text) <= 2 and option in text):
                return action, flags
        print("WARNING! Fuzzy match!")
        flags = "Fuzzy match"
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('go to'):
                # act = 'go to'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('explore'):
                act = 'explore'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('go grasp'):
                act = 'grasp'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if name in text and id in text:
                return action, flags
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('go to'):
                # act = 'go to'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('explore'):
                act = 'explore'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('go grasp'):
                act = 'grasp'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action, flags
        if len(text) == 1:
            i = ord(text) - ord('A')
            if i in range(len(available_actions)):
                return available_actions[i]
        print("WARNING! No available action parsed!!! Random choose one")
        flags = "failed to parse"
        return random.choice(available_actions), flags


    def progress2text(self, current_step, satisfied, opponent_grabbed_objects, opponent_last_room,):
        s = f"I've taken {current_step}/3000 steps. "

        sss = {}
        for room, obj_list in self.obj_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            s_bed = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss
            if len(obj_list[2]) > 0:
                s_bed = 'the goal position bed'
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += 'nothing'
            elif s_obj != "" and s_con != "" and s_bed == "":
                sr += s_obj + ', and ' + s_con
            elif s_obj != "" and s_con == "" and s_bed != "":
                sr += s_obj + ', and ' + s_bed
            elif s_obj == "" and s_con != "" and s_bed != "":
                sr += s_con + ', and ' + s_bed
            elif s_obj != "" and s_con != "" and s_bed != "":
                sr += s_obj + ', ' + s_con + ', and ' + s_bed
            else:
                sr += s_obj + s_con + s_bed
            sss[room] = sr

        if len(satisfied) == 0:
            if len(self.object_list[2]) == 0:
                s += "I haven't found the goal position bed. "
            else:
                s += ""
        else:
            s += f"{'I' if self.single else 'We'}'ve already transported "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x['type'] == 0]) == 0:
                s += 'nothing'
            s += ', '.join([f"<{x['name']}> ({x['id']})" for x in unique_satisfied if x['type'] == 0])
            s += ' to the bed. '

        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj['type'] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj['type'] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj['contained']):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = 'nothing'
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

        if self.holding_objects[0]["type"] == 0 and self.holding_objects[1]['type'] == 0:
            s += f"I'm holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}). "
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "I'm holding nothing. "
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"I'm holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"I'm holding {s_hold[0]}{s_hold[1]}"

        # print(self.current_room, self.obj_per_room)
        if self.current_room not in self.rooms_explored: pred_room = 'none'
        else: pred_room = self.rooms_explored[self.current_room]
        if pred_room != 'all' and sss[self.current_room] == 'nothing':
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it. "
        else:
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it and found {sss[self.current_room]}. "
        ### opponent modeling
        if not self.single:
            s_hold = ["", ""]
            for i, obj in enumerate(opponent_grabbed_objects):
                if obj['type'] == 0:
                    s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
                elif obj['type'] == 1:
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj['contained']):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = 'nothing'
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "
            if opponent_grabbed_objects[0]["type"] == 0 and opponent_grabbed_objects[1]['type'] == 0:
                ss = f"two target objects <{opponent_grabbed_objects[0]['name']}> ({opponent_grabbed_objects[0]['id']}) and <{opponent_grabbed_objects[1]['name']}> ({opponent_grabbed_objects[1]['id']}). "
            if s_hold[0] == "" and s_hold[1] == "":
                ss = "nothing. "
            elif s_hold[0] != "" and s_hold[1] != "":
                ss = f"{s_hold[0][:-2]}, and {s_hold[1]}"
            else:
                ss = f"{s_hold[0]}{s_hold[1]}"

            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == self.current_room:
                s += f"I also see {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == self.current_room:
                continue
            #s += f"I've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
            if room not in self.rooms_explored: pred_room = 'none'
            else: pred_room = self.rooms_explored[room]
            if pred_room != 'all' and sss[room] == 'nothing':
                s += f"I've explored {pred_room} of the {room}. "
            else:
                s += f"I've explored {pred_room} of the {room}, and I found {sss[room]} there. "

        return s


    def get_available_plans(self, message):
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        send a message: ""
        """
        available_plans = []
        agent_position = self.physics_memory['agent']
        if self.communication and message is not None:
            available_plans.append(f"send a message: {message} - distance:{0}")
        if self.holding_objects[0]['type'] is None or self.holding_objects[1]['type'] is None:
            for obj in self.object_list[0]: # target object
                distance = np.linalg.norm(self.physics_memory[obj['id']] - agent_position, 2).round(2)
                available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']}) - distance:{distance}")
            if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1):
                for obj in self.object_list[1]: # container
                    distance = np.linalg.norm(self.physics_memory[obj['id']] - agent_position, 2).round(2)
                    available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']}) - distance:{distance}")
        else:
            if self.holding_objects[0]['type'] == 1 and self.holding_objects[0]['contained'][-1] is None and self.holding_objects[1]['type'] == 0:
                available_plans.append(f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})")
            elif self.holding_objects[1]['type'] == 1 and self.holding_objects[1]['contained'][-1] is None and self.holding_objects[0]['type'] == 0:
                available_plans.append(f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})")
        if any(obj['type'] is not None for obj in self.holding_objects) and len(self.object_list[2]) != 0:
            bedroom_name = None
            for room_name in self.rooms:
                if 'Bedroom' in room_name:
                    bedroom_name = room_name
            if bedroom_name is not None:
                distance = np.linalg.norm(self.physics_memory[bedroom_name] - agent_position, 2).round(2)
            available_plans.append(f"transport objects I'm holding to the bed - distance:{distance}")
        for room in self.rooms:
            distance = np.linalg.norm(self.physics_memory[room] - agent_position, 2).round(2)
            if room == self.current_room or room is None or room == 'None':
                continue
            available_plans.append(f"go to {room} - distance:{distance}")
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            available_plans.append(f"explore current room {self.current_room} - distance:{4}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans



    def K_available_plans(self, K=3):
        """
        Refactored to:
          - Remove duplication
          - Add relevance score description and relative proximity info
          - Keep consistent plan formatting
        """

        import numpy as np

        # ---------- helpers ----------

        def _dist(pos_key, ref_pos):
            return float(np.linalg.norm(self.physics_memory[pos_key] - ref_pos, 2).round(2))

        def _eff(R, d):
            efficiency = 0
            if d==0: d = 0.1
            if R == 1:
                efficiency = 1 / d + 1000
            elif R == 0.5:
                efficiency = 1 / d + 100
            elif R == 0:
                efficiency = 1 / d + 10
            elif R == -1:
                efficiency = 1 / d + 1

            return efficiency


        def _make_plan(obj_id, name, prompt, my_e, oppo_e, R, d_me, d_op):
            # proximity description
            if d_me < d_op:
                proximity = f"I'm closer than {self.oppo_name}"
            else:
                proximity = f"I'm farther than {self.oppo_name}"
            # relevance description
            relevance_desc = relevance_to_language(R)
            # final formatted prompt with additional info
            full_prompt = f"{prompt} - relevance score: {relevance_desc}, relative proximity: {proximity}"
            return {
                'id': obj_id,
                'name': name,
                'prompt': full_prompt,
                'my_efficiency': my_e,
                'oppo_efficiency': oppo_e,
            }

        def _add_by_advantage(plan, plans1, plans2):
            (plans1 if plan['my_efficiency'] > plan['oppo_efficiency'] else plans2).append(plan)

        def _grasp_plans_for(category_idx, label):
            for obj in self.object_list[category_idx]:
                obj_id = obj['id']
                name = obj['name']
                d_me = _dist(obj_id, agent_pos)
                d_op = _dist(obj_id, oppo_pos)
                R = self.relevance_scores[obj_id]
                my_e = _eff(R, d_me)
                op_e = _eff(R, d_op)
                prompt = f"go grasp {label} <{name}> ({obj_id})"
                plan = _make_plan(obj_id, name, prompt, my_e, op_e, R, d_me, d_op)
                _add_by_advantage(plan, available_plans_1, available_plans_2)

        def _put_into_plans(container_idx, object_idx):
            container = self.holding_objects[container_idx]
            obj = self.holding_objects[object_idx]
            obj_id, obj_name = obj['id'], obj['name']
            R = self.relevance_scores[obj_id]
            d_me, d_op = 1, 100  # fixed since it's already in hand
            my_e = _eff(R, d_me)
            op_e = _eff(-1, d_op)
            prompt = f"put <{obj_name}> ({obj_id}) into the container <{container['name']}> ({container['id']})"
            plan = _make_plan(obj_id, obj_name, prompt, my_e, op_e, R, d_me, d_op)
            _add_by_advantage(plan, available_plans_1, available_plans_2)

        def _room_plan(room_key, prompt, only_me=False):
            room_name, room_id = parse_name_id(room_key)
            d_me = _dist(room_key, agent_pos)
            d_op = _dist(room_key, oppo_pos) if not only_me else 100
            R = self.relevance_scores[room_id]
            my_e = _eff(R, d_me)
            op_e = _eff(-1 if only_me else R, d_op)
            plan = _make_plan(room_id, room_name, prompt, my_e, op_e, R, d_me, d_op)
            _add_by_advantage(plan, available_plans_1, available_plans_2)

        # ---------- init ----------
        available_plans_1, available_plans_2 = [], []
        agent_pos = self.physics_memory['agent']
        oppo_pos = self.physics_memory['oppo']

        # ---------- case 1: grasping targets or containers ----------
        left_empty = self.holding_objects[0]['type'] is None
        right_empty = self.holding_objects[1]['type'] is None
        if left_empty or right_empty:
            _grasp_plans_for(0, "target object")
            if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1):
                _grasp_plans_for(1, "container")
        else:
            # ---------- case 2: put object into container ----------
            if (self.holding_objects[0]['type'] == 1 and
                    self.holding_objects[0]['contained'][-1] is None and
                    self.holding_objects[1]['type'] == 0):
                _put_into_plans(container_idx=0, object_idx=1)
            elif (self.holding_objects[1]['type'] == 1 and
                  self.holding_objects[1]['contained'][-1] is None and
                  self.holding_objects[0]['type'] == 0):
                _put_into_plans(container_idx=1, object_idx=0)

        # ---------- case 3: transport ----------
        if any(obj['type'] is not None for obj in self.holding_objects) and len(self.object_list[2]) != 0:
            bedroom_key = next((r for r in self.rooms if r and 'Bedroom' in r), None)
            if bedroom_key:
                _room_plan(bedroom_key, "transport objects I'm holding to the bed", only_me=True)

        # ---------- case 4: go to ----------
        for room_key in self.rooms:
            if not room_key or room_key == 'None' or room_key == self.current_room:
                continue
            _room_plan(room_key, f"go to {room_key}", only_me=False)

        # ---------- case 5: explore current room ----------
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            d_me = _dist(self.current_room, agent_pos) + 2
            d_op = _dist(self.current_room, oppo_pos) + 2
            room_name, room_id = parse_name_id(self.current_room)
            R = self.relevance_scores[room_id]
            my_e = _eff(R, d_me)
            op_e = _eff(R, d_op)
            prompt = f"explore current room {self.current_room}"
            plan = _make_plan(room_id, room_name, prompt, my_e, op_e, R, d_me, d_op)
            _add_by_advantage(plan, available_plans_1, available_plans_2)

        # ---------- final selection ----------
        all_len = len(available_plans_1) + len(available_plans_2)
        if all_len <= K:
            final_plan = available_plans_1 + available_plans_2
        elif len(available_plans_1) >= K:
            final_plan = sorted(available_plans_1, key=lambda x: x['my_efficiency'], reverse=True)[:K]
        else:
            sp1 = sorted(available_plans_1, key=lambda x: x['my_efficiency'], reverse=True)
            sp2 = sorted(available_plans_2, key=lambda x: x['my_efficiency'], reverse=True)
            final_plan = (sp1 + sp2)[:K]

        # ---------- formatted output ----------
        plans = ""
        plan_list = []
        for i, plan in enumerate(final_plan):
            plans += f"{chr(ord('A') + i)}. {plan['prompt']}\n"
            plan_list.append(plan['prompt'])
        return plans, len(final_plan), plan_list


    def run(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room,
            action_history, dialogue_history, opponent_grabbed_objects = None, opponent_last_room = None, physics_memory = None,
            relevance_scores = None):
        info = {}
        print("current_step", current_step)
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        self.physics_memory = physics_memory
        self.relevance_scores = relevance_scores
        progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        #dialogue_history_desc = '\n'.join(dialogue_history[-3:][1] if len(dialogue_history) > 3 else dialogue_history)
        dialogue_history_desc = '\n'.join(
            text for _, text in (dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history)
        )
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        message = None
        prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)

        # if self.communication:
        #     prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
        #     if not action_history[-1].startswith('send a message'):
        #         gen_prompt = self.generator_prompt_template.replace('$GOAL$', self.goal_desc)
        #         gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
        #         gen_prompt = gen_prompt.replace('$ACTION_HISTORY$', action_history_desc)
        #         gen_prompt = gen_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
        #         gen_prompt = gen_prompt + f"\n{self.agent_name}:"
        #         chat_prompt = [{"role": "user", "content": gen_prompt}]
        #         outputs, usage = self.generator(chat_prompt if self.chat else gen_prompt, self.sampling_params)
        #         print("outputs:\n",outputs)
        #         print("outpus[0]:\n",outputs[0])
        #         self.total_cost += usage
        #         message = outputs[0]
        #         # print("message:\n",message)
        #         # if len(message) > 0 and message[0] != '"':
        #         #     message = re.search(r'"([^"]+)"', message)
        #         #     if message:
        #         #         message = '"' + message.group(1) + '"'
        #         info['prompt_comm'] = gen_prompt
        #         info['output_comm'] = outputs
        #         info['usage_comm'] = usage
        #         if self.debug:
        #             print(f"prompt_comm:\n{gen_prompt}")
        #             print(f"output_comm:\n{message}")

        available_plans, num, available_plans_list = self.K_available_plans(K=3)
        if num == 0 or (message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                     "plan": None})
            return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        if self.cot:
            if 'gpt-oss' not in self.lm_id:
                prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            ## truncate the unfinished cot
            last_index = output.rfind('.')
            if last_index != -1:
                output = output[:last_index + 1]
            else:
                output += '.'

            # info['outputs_cot'] = outputs
            # info['usage_plan_stage_1'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")

            if 'gpt-oss' not in self.lm_id:
                chat_prompt = [{"role": "user", "content": prompt},
                               {"role": "assistant", "content": output},
                               {"role": "user", "content": "Answer with only one best next action. So the answer is option"}]
                normal_prompt = prompt + ' ' + output + ' Answer with only one best next action. So the answer is option'
                outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]

            # info['usage_plan_stage_2'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            normal_prompt = prompt
            chat_prompt = [{"role": "user", "content": prompt}]
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]
            # info['usage_step_1'] = usage
            if self.debug:
                print(f"output_plan_stage_1:\n{output}")
        plan, flags = self.parse_answer(available_plans_list, output)
        print(self.total_cost)

        if 'distance' in plan:
            plan = plan.split('-')[0][:-1]
        if self.debug:
            print(f"plan: {plan}\n")
        info.update({"num_available_actions": num,
                     "output_plan_stage_2": output,
                     "parse_exception": flags,
                     "plan": plan,
                     "total_cost": self.total_cost})
        return plan, info


    def run_parse_init(self, oppo_message):
        """
        Parses a conversation log from the opponent and extracts specific data in a predefined format.

        Parameters:
            oppo_message (str): The opponent's message to be parsed.

        Returns:
            str: Extracted data from the conversation log.
        """
        example_data = "[{\"id\": id1, \"name\": name1}, {\"id\": id2, \"name\": name2}]"
        prompt = f"Understand conversation log from {self.oppo_name} and transform it. " \
                 f"Note that <object_name> (object_id) is the format.\n" \
                 f"Conversation log: [{self.oppo_name}'s Message: {oppo_message}]\n" \
                 f"Output format: \n" \
                 f"Output: Collaborator name: collaborator_name, extracted_data: {example_data}"
        chat_prompt = [{"role": "user", "content": prompt}]

        # if self.debug:
        #     print("--------def run_parse_init(self, oppo_message):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0].replace('Output: ', '').replace(' ', '')
        _name = output.split('Collaboratorname:')[-1].split(',extracted_data:')[0].strip()
        _data = output.split('extracted_data:')[-1].strip()

        # if self.debug:
        #     print("output:", outputs[0].replace('Output: ', ''))
        #     print("\n")
        return _data


    def run_request(self, category, my_plan, current_room):
        """
        Generates a friendly and concise message to send to a collaborator based on the given inputs.

        Parameters:
            category (str): The category or topic of the message.
            my_plan (str): The plan or action you want to communicate.
            current_room (str): The current room or context of the conversation.

        Returns:
            str: The refined message to send to the collaborator.
        """

        prompt = f"Create a brief, friendly message to send to your collaborator. It should contain the content of the original message below. \n" \
                 f"Original Message: \"I want to do {my_plan} at {category}. Have you already done that? {current_room}\"\n" \
                 f"Output format:\n" \
                 f"Message: \"your meessage\""
        chat_prompt = [{"role": "user", "content": prompt}]
        # if self.debug:
        #     print("--------def run_request(self, category, my_plan):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0]

        # if self.debug:
        #     print("output:", output)
        #     print("\n")

        _message = output.replace("Message: ", "")
        return _message

    def run_parse_request(self, oppo_message, plan_list):
        """
        Parses the conversation log from the collaborator and determines the relevant category and
        whether a previous action has been completed based on the plan list.

        Parameters:
            oppo_message (str): The collaborator's message to be parsed.
            plan_list (list): The list of previous plans to check against.

        Returns:
            str: The selected category based on the collaborator's message.
        """
        choices = "\n".join(
            f"{chr(65 + i)}. {room}" for i, room in enumerate(self.rooms)
        )
        prompt = f"Based on the conversation log from my collaborator {self.oppo_name}, select which of the following categories the information" \
                 f"that my collaborator wants corresponds to.\n" \
                 f"Conversation log: [{self.oppo_name}'s Message: {oppo_message}]\n" \
                 f"{choices}\n\n" \
                 f"Additionally, check my previous plans and answer with yes or no whether I've already completed the action my collaborator is attempting to perform.\n" \
                 f"Plan list: [{plan_list[-5:] if len(plan_list) >= 5 else plan_list}]\n" \
                 f"Output format:\n" \
                 f"Output: Collaborator name: collaborator_name, Select: select, Action completed: yes/no"
        chat_prompt = [{"role": "user", "content": prompt}]
        if self.debug:
            print("--------def run_parse_request(self, oppo_message):--------")
            print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0]

        if self.debug:
            print("output:", output)
            print("\n")

        return output


        #
        # _name = output.split('Collaborator name:')[-1].split(', Select:')[0].strip()
        # _select = output.split('Select:')[-1].split(', Action completed:')[0].strip()
        # _answer = output.split('Action completed:')[-1].strip()
        #
        # if 'livingroom' in _select or 'A' in _select:
        #     _select = 'livingroom'
        # elif 'kitchen' in _select or 'B' in _select:
        #     _select = 'kitchen'
        # elif 'bedroom' in _select or 'C' in _select:
        #     _select = 'bedroom'
        # elif 'bathroom' in _select or 'D' in _select:
        #     _select = 'bathroom'
        # return _select


    def run_response(self, room, useful_memory, oppo_current_room, current_plan, grabbed_objects, yes_or_no):
        """
        Generates a response based on past task memory, the current task, and objects held. The response
        is then refined to be polite and concise.

        Parameters:
            category (str): The task category.
            sub_task_memory (list): List of past task memories.
            current_task (list): The current task being performed.
            oppo_current_room (str): The room where the collaborator is located.
            current_plan (str): The current plan of action.
            grabbed_objects (list): List of objects currently held.

        Returns:
            str: The refined message to send to the collaborator.
        """

        # useful_memory = []
        # for memory in sub_task_memory:
        #     if memory['room'] in category and memory['efficiency'] > 0:
        #         if len(current_task) > 0 and memory['id'] == current_task[0][1].split(' ')[2][1: -1]:
        #             continue
        #         useful_memory.append({'id': memory['id'], 'name': memory['name'], 'step': memory['step']})
        #
        # if len(useful_memory) == 0:
        #     return "There's nothing more to do there." + oppo_current_room
        # elif len(useful_memory) == 1:
        #     if useful_memory[0]['name'] == category:
        #         return "I don't have any information about it yet." + oppo_current_room
        current_room_desc = f"And currently I'm in the {oppo_current_room}."
        if len(useful_memory) == 0:
            return f"I don't have any information about it yet. {current_room_desc}"
        if room in self.rooms_explored and self.rooms_explored[room] == 'all':
            return f"There's nothing more to do there. {current_room_desc}"


        useful_memory = sorted(useful_memory, key=lambda x: x['step'])
        original_message = ""
        step = -1
        for memory in useful_memory:
            if memory['step'] != step:
                if original_message == "":
                    step = memory['step']
                    original_message += f"at step {step}, I saw <{memory['name']}> ({memory['id']})"
                else:
                    step = memory['step']
                    original_message += f". at step {step}, I saw <{memory['name']}> ({memory['id']})"

            else:
                original_message += f", <{memory['name']}> ({memory['id']})"
        original_message += "."
        _message = original_message

        held_object = ""
        if len(grabbed_objects) == 0:
            held_object += "I'm holding nothing. "
        else:
            held_object += f"I'm holding target object {grabbed_objects[0]}. "
            if len(grabbed_objects) == 2:
                held_object = held_object[:-2] + f" and {grabbed_objects[1]}. "

        did_didnt = "did" if yes_or_no == 'yes' else "didn't"

        prompt = f"I'd like to forward the message below to my collaborator {self.oppo_name}. " \
                 f"Please make it a little kinder and brief.\n" \
                 f"The name and id of the object are very important information and should not be omitted.\n" \
                 f"Original message: I {did_didnt} do that. I do {current_plan} now and {held_object}And {original_message} {current_room_desc}\n" \
                 f"Output format:" \
                 f"Message: \"your message\""
        chat_prompt = [{"role": "user", "content": prompt}]

        # if self.debug:
        #     print("--------run_response(self, category, sub_task_memory, current_task):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0]

        # if self.debug:
        #     print("output:", output)
        #     print("\n")
        _message = output.replace("Message: ", "")

        return _message

    def run_parse_response(self, oppo_message):
        """
        Parses a conversation log from the collaborator and extracts details such as the collaborator's plan,
        held objects, and relevant data in a predefined format.

        Parameters:
            oppo_message (str): The collaborator's message to be parsed.

        Returns:
            str: Extracted data from the conversation log.
        """
        held_data = "[{\"id\": id1, \"name\": name1}, {\"id\": id2, \"name\": name2]"
        example_data = "[{\"id\": id1, \"name\": name1, \"step\": step1}, {\"id\": id2, \"name\": name2, \"step\": step2}]"
        prompt = f"Understand conversation log from {self.oppo_name} and transform it. " \
                 f"Note that [action] <object_name> (object_id) is the format.\n" \
                 f"Conversation log: [{self.oppo_name}'s Message: {oppo_message}]\n" \
                 f"Output format: \n" \
                 f"Output: Collaborator name: collaborator_name, Collaborator plan: [action] <object_name> (object_id), Held object: {held_data}, extracted_data: {example_data}"
        chat_prompt = [{"role": "user", "content": prompt}]

        if self.debug:
            print("--------def run_parse_response(self, oppo_message):--------")
            print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0].replace('Output: ', '').replace(' ', '')
        _name = output.split('Collaboratorname:')[-1].split(',Collaboratorplan:')[0].strip()
        _plan = output.split('Collaboratorplan:')[-1].split(',Actioncompleted:')[0].strip()
        _completed = output.split('Actioncompleted:')[-1].split(',extracted_data:')[0].strip()
        _data = output.split('extracted_data:')[-1].strip()

        if self.debug:
            print("output:", outputs[0].replace('Output: ', ''))
            print("\n")
        return _data


    def belief_validation(self, is_other_visit, my_plan, room, my_relativity, filtered_communication_memory):
        """
        Validates the plan about whether the collaborator has already interacted with an object based on their trajectory and room visits.

        Parameters:
            is_other_visit (bool): Whether the collaborator visited the room.
            my_plan (str): The action plan involving the object.
            room (str): The room where the object is located.
            my_relativity (float): Relevance of the action to the goal for the agent.
            filtered_communication_memory (list): Communication log filtered to relevant events.

        Returns:
            bool: True if the collaborator likely interacted with the object, False otherwise.
        """

        my_relativity = relevance_to_language(my_relativity)

        act, name, id = my_plan.split('-')
        prompt = f"I want to do an {act} with {name}. " \
                 f"But I wonder if {self.oppo_name} did {act} with {name} {id} before me. \n" \
                 f"The {name} {id} is in the {room}. " \
                 f"The {act} with this {name} {id} has {my_relativity} for me in achieving the goal, and {self.oppo_name} would think so too. \n"
        if is_other_visit:
            prompt += f"I know that {self.oppo_name} visited {room} after the last time I saw that {name} {id} in there. \n"
        else:
            prompt += f"I don't know if {self.oppo_name} visited {room} after the last time I saw that {name} {id} in there. "
            prompt += f"So, I can't tell if {self.oppo_name} did {act} with the {name} {id} in the {room}. \n"

        if len(filtered_communication_memory) > 0:
            prompt += f"This is the communication log from {self.oppo_name} between the last time I saw that {name} {id} in the {room} and now: {filtered_communication_memory}. \n"

        prompt += f"First, predict {self.oppo_name}’s trajectory between last time I saw {name} {id} and current time, and then choose from the two answers below."
        prompt += "Choose from the two answers below. \n"
        prompt += f"A. {self.oppo_name}'s past trajectory wouldn't overlap with the {room} at that time to {act} {name} {id}. \n" \
                  f"B. {self.oppo_name}'s past trajectory would overlap with the {room} at that time to {act} {name} {id}. "

        if self.cot:
            cot_prompt = prompt
            cot_prompt += "\nAnswer: Let's think step by step."

            chat_prompt = [{"role": "user", "content": cot_prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            cot_output = outputs[0]

            if self.debug:
                print("---------------------------- belief_validation -----------------------------")
                print("chat prompt:", cot_prompt)
                print("output:", cot_output)
                print("\n")

            prompt = "Based on your previous thoughts, choose from the two answers below. \n"
            prompt += f"A. {self.oppo_name}'s past trajectory wouldn't overlap with the {room} at that time to {act} {name} {id}. \n" \
                      f"B. {self.oppo_name}'s past trajectory would overlap with the {room} at that time to {act} {name} {id}. \n\n"

            prompt += "Output format: \n" \
                      "Answer: alphabet"

            chat_prompt = [{"role": "user", "content": cot_prompt},
                           {"role": "assistant", "content": cot_output},
                           {"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            output = outputs[0].replace('Answer: ', '')

            if self.debug:
                print("chat prompt:", prompt)
                print("output:", output)
                print("\n")

        else:
            prompt = "Choose from the two answers below. \n"
            prompt += f"A. {self.oppo_name}'s past trajectory wouldn't overlap with the {room} at that time to {act} {name} {id}. \n" \
                      f"B. {self.oppo_name}'s past trajectory would overlap with the {room} at that time to {act} {name} {id}. \n\n"

            prompt += "Output format: \n" \
                      "Answer: alphabet"

            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            output = outputs[0].replace('Answer: ', '')

        # print(self.agent_name, 'Belief', my_plan, output)

        if 'A' == output[0]:
            # print('A')
            return False
        elif 'B' == output[0]:
            # print('B')
            return True
        else:
            if 'A' in output:
                # print('A')
                return False
            elif 'B' in output:
                # print('B')
                return True
            else:
                return True



def relevance_to_language(relevance_score):
    if relevance_score == 1:
        return "Strong relevance"
    elif relevance_score == 0.5:
        return "Medium relevance"
    elif relevance_score == 0:
        return "Low relevance"
    elif relevance_score == -1:
        return "None relevance"
    return "Unknown relevance"
