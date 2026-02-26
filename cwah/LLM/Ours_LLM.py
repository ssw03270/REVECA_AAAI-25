import random
import math
import re

import openai
import torch
import json
import os
import pandas as pd
from openai.error import OpenAIError
import backoff

class Ours_LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_name = "Alice" if agent_id == 2 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 2 else "he"
        self.debug = sampling_parameters.debug
        self.goal_location = None
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path

        self.communication = communication
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id
        self.OPENAI_KEY = None
        self.total_cost = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

        if self.source == 'openai':
            openai.api_key = os.getenv("OPENAI_KEY")
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
        elif source == 'huggingface':
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'early_stopping': True,
            }
        elif source == "debug":
            self.sampling_params = sampling_parameters
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id, device):
            if source == 'huggingface':
                from transformers import AutoModelForCausalLM, AutoTokenizer, LLaMATokenizer, LLaMAForCausalLM
                print(f"loading huggingface model {lm_id}")
                if 'llama' in lm_id or 'alpaca' in lm_id:
                    tokenizer = LLaMATokenizer.from_pretrained(lm_id,
                                                               cache_dir='/work/pi_chuangg_umass_edu/.cahce')  # '/gpfs/u/scratch/AICD/AICDhnng/.cache')
                    model = LLaMAForCausalLM.from_pretrained(lm_id,  # device_map="balanced_low_0",
                                                             # max_memory = {0: "10GB", 1: "20GB", 2: "20GB", 3: "20GB",4: "20GB",5: "20GB",6: "20GB",7: "20GB"},
                                                             torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                             load_in_8bit=False,
                                                             cache_dir='/work/pi_chuangg_umass_edu/.cahce') \
                        .to(device)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce')
                    model = AutoModelForCausalLM.from_pretrained(lm_id, torch_dtype=torch.float16,
                                                                 pad_token_id=tokenizer.eos_token_id,
                                                                 cache_dir='/work/pi_chuangg_umass_edu/.cahce').to(
                        device)
                print(f"loaded huggingface model {lm_id}")

            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            response = openai.ChatCompletion.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['message']['content'] for i in
                                                 range(sampling_params['n'])]
                            if 'gpt-4' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage'][
                                    'completion_tokens'] * 0.06 / 1000
                            elif 'gpt-3.5' in self.lm_id:
                                usage = response['usage']['total_tokens'] * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 			  range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == 'huggingface':
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    # print(sampling_params)
                    output_dict = model.generate(input_ids,
                                                 # max_length=prompt_len + sampling_params['max_new_tokens'],
                                                 **sampling_params)
                    generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                    # vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)
                    # token_log_probs = torch.gather(vocab_log_probs, 2,
                    # 							   output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()
                    for i, sample in enumerate(generated_samples):
                        stop_idx = sample.index('\n') if '\n' in sample else None
                        generated_samples[i] = sample[:stop_idx]
                # 	token_log_probs[i] = token_log_probs[i][:stop_idx]
                # mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]

                elif source == "debug":
                    return ["navigation"]
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id, self.device)

    def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
        self.rooms = rooms_name
        self.roomname2id = roomname2id
        self.goal_location = goal_location
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)

    def calculate_distances(self, coords1, coords2):
        x1, y1, z1 = coords1
        x2, y2, z2 = coords2
        return round(math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2), 2)

    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        # print(goals)
        map_rel_to_pred = {
            'inside': 'into',
            'on': 'onto',
        }
        s = "Find and put target objects "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_')
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
            # s += f"Alice holds a book, "
            elif relation == 'sit':
                continue
            # s += f"Alice sits in {obj2}, "
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return "None."

        s = s[:-2] + f" {map_rel_to_pred[r]} the goal location {self.goal_location}."
        return s, f"{map_rel_to_pred[r]} the {self.goal_location}"


    def location_holding(self, current_room, grabbed_objects):
        s = ""
        s += f"I'm in the <{current_room['class_name']}> ({self.roomname2id[current_room['class_name']]}). "
        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding target object <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        return s


    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room,
                      satisfied, opponent_grabbed_objects, opponent_last_room, room_explored, id2node):
        sss = {}
        my_position = id2node[self.agent_id]['obj_transform']['position']
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                room_position = id2node[self.roomname2id[room]]['obj_transform']['position']
                distance = self.calculate_distances(my_position, room_position)
                sss[room] = f"The <{room}> ({self.roomname2id[room]}) |{distance} m| is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"goal location {extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    obj_position = x['obj_transform']['position']
                    distance = self.calculate_distances(my_position, obj_position)
                    s_obj += f"target object <{x['class_name']}> ({x['id']}) |{distance} m|"
                else:
                    ss = 'target object' + ', '.join([f"<{x['class_name']}> ({x['id']}) |{self.calculate_distances(my_position, x['obj_transform']['position'])} m|" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    container_position = x['obj_transform']['position']
                    distance = self.calculate_distances(my_position, container_position)
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']}) |{distance} m|"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']}) |{self.calculate_distances(my_position, x['obj_transform']['position'])} m|" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        if len(satisfied) == 0:
            s = ""
        else:
            s = f"{'I' if self.single else 'We'}'ve already found and put target object "
            s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
            s += ' ' + self.goal_location_with_r + '. '

        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding target object <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the <{current_room['class_name']}> ({self.roomname2id[current_room['class_name']]}), where I found {sss[current_room['class_name']]}. "

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "

        return s

    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        [send_message] <"">
        """
        available_plans = []
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    def get_scene_desc(self, current_room, grabbed_objects, satisfied, unchecked_containers,
                 ungrabbed_objects,
                 goal_location_room,
                 action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, id2node, room_explored=None):
        scene_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects,
                                        goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room,
                                        room_explored, id2node)
        return scene_desc

    def run_init(self, message):
        """
        Refines the given message to be more polite and concise using a generator model.

        Parameters:
            message (str): The original message to be rephrased.

        Returns:
            str: The rephrased message.
        """

        prompt = f"I'd like to forward the message below to my collaborator {self.agent_name}. " \
                 f"Please make it a little kinder and brief.\n" \
                 f"Original message: {message}\n" \
                 f"Output format:\n" \
                 f"Message: \"your message\""
        chat_prompt = [{"role": "user", "content": prompt}]

        # if self.debug:
        #     print("--------run_init(self, message):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0].replace('Message: ', '')
        # if self.debug:
        #     print("output:", output)
        #     print("\n")

        return output
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

        prompt = f"Based on the conversation log from my collaborator {self.oppo_name}, select which of the following categories the information" \
                 f"that my collaborator wants corresponds to.\n" \
                 f"Conversation log: [{self.oppo_name}'s Message: {oppo_message}]\n" \
                 f"A. livingroom\n" \
                 f"B. kitchen\n" \
                 f"C. bedroom\n" \
                 f"D. bathroom\n\n" \
                 f"Additionally, check my previous plans and answer with yes or no whether I've already completed the action my collaborator is attempting to perform.\n" \
                 f"Plan list: [{plan_list[-5:] if len(plan_list) >= 5 else plan_list}]\n" \
                 f"Output format:\n" \
                 f"Output: Collaborator name: collaborator_name, Select: select, Action completed: yes/no"
        chat_prompt = [{"role": "user", "content": prompt}]
        # if self.debug:
        #     print("--------def run_parse_request(self, oppo_message):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0]

        # if self.debug:
        #     print("output:", output)
        #     print("\n")

        _name = output.split('Collaborator name:')[-1].split(', Select:')[0].strip()
        _select = output.split('Select:')[-1].split(', Action completed:')[0].strip()
        _answer = output.split('Action completed:')[-1].strip()

        if 'livingroom' in _select or 'A' in _select:
            _select = 'livingroom'
        elif 'kitchen' in _select or 'B' in _select:
            _select = 'kitchen'
        elif 'bedroom' in _select or 'C' in _select:
            _select = 'bedroom'
        elif 'bathroom' in _select or 'D' in _select:
            _select = 'bathroom'
        return _select

    def run_response(self, category, sub_task_memory, current_task, oppo_current_room, current_plan, grabbed_objects):
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

        useful_memory = []
        for memory in sub_task_memory:
            if memory['room'] in category and memory['efficiency'] > 0:
                if len(current_task) > 0 and memory['id'] == current_task[0][1].split(' ')[2][1: -1]:
                    continue
                useful_memory.append({'id': memory['id'], 'name': memory['name'], 'step': memory['step']})

        if len(useful_memory) == 0:
            return "There's nothing more to do there." + oppo_current_room
        elif len(useful_memory) == 1:
            if useful_memory[0]['name'] == category:
                return "I don't have any information about it yet." + oppo_current_room

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

        prompt = f"I'd like to forward the message below to my collaborator {self.oppo_name}. " \
                 f"Please make it a little kinder and brief.\n" \
                 f"The name and id of the object are very important information and should not be omitted.\n" \
                 f"Original message: I didn't do that. I do {current_plan} now and {held_object}And {original_message} {oppo_current_room}\n" \
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

        # if self.debug:
        #     print("--------def run_parse_response(self, oppo_message):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0].replace('Output: ', '').replace(' ', '')
        _name = output.split('Collaboratorname:')[-1].split(',Collaboratorplan:')[0].strip()
        _plan = output.split('Collaboratorplan:')[-1].split(',Actioncompleted:')[0].strip()
        _completed = output.split('Actioncompleted:')[-1].split(',extracted_data:')[0].strip()
        _data = output.split('extracted_data:')[-1].strip()

        # if self.debug:
        #     print("output:", outputs[0].replace('Output: ', ''))
        #     print("\n")
        return _data

    def run_subgoal(self, message):
        """
        Refines the given message to be more polite and concise before forwarding it to the collaborator.

        Parameters:
            message (str): The original message to be rephrased.

        Returns:
            str: The rephrased and refined message.
        """
        prompt = f"I'd like to forward the message below to my collaborator {self.agent_name}. " \
                 f"Please make it a little kinder and brief.\n" \
                 f"Original message: {message}\n" \
                 f"Output format:\n" \
                 f"Message: \"your message\""
        chat_prompt = [{"role": "user", "content": prompt}]

        # if self.debug:
        #     print("--------run_subgoal(self, message):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        output = outputs[0].replace('Message: ', '')
        # if self.debug:
        #     print("output:", output)
        #     print("\n")

        return output
    def run_parse_subgoal(self, oppo_message):
        """
        Parses the collaborator's message to extract the object name and object ID in a specified format.

        Parameters:
            oppo_message (str): The collaborator's message to be parsed.

        Returns:
            str: The extracted object name and object ID.
        """
        prompt = f"Understand conversation log and transform it. " \
                 f"Note that <object_name> (object_id) is the format.\n" \
                 f"Conversation log: [{self.oppo_name}'s Message: {oppo_message}]\n" \
                 f"Output format: \n" \
                 f"Output: Collaborator name: collaborator_name, Object name: object_name, Object ID: object_id"

        chat_prompt = [{"role": "user", "content": prompt}]
        # if self.debug:
        #     print("--------run_parse_subgoal(self, oppo_message):--------")
        #     print("chat prompt:", prompt)
        outputs, usage = self.generator(chat_prompt, self.sampling_params)

        output = outputs[0].replace(' ', '').replace(',', '').split('Objectname:')[-1].split('ObjectID:')
        output = f"{output[0]}\n{output[1]}"
        # if self.debug:
        #     print("output:", outputs[0])
        #     print("\n")
        return output

    def run_efficiency_weight(self, object_name, object_from, object_interaction):
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
        #
        # if 'GRABBABLE' in object_interaction:
        #     prompt += f"{object_name} is a grabbable object. \n"
        # elif 'CONTAINERS' in object_interaction:
        #     prompt += f"{object_name} is a container, so the thing I'm looking for might be inside. \n"
        # elif 'Room' in object_interaction:
        #     prompt += f"{object_name} is a room, and it can contain the things I'm looking for"

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
            # if self.debug:
            #     print("--------run_efficiency_weight(self, object_name, object_from, object_interaction):--------")
            #     print("chat prompt:", cot_prompt)
            #     print("output:", cot_output)
            #     print("\n")

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
            # print("chat prompt:", prompt)
            # print("output:", output)
            # print("\n")

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

    def planning(self, sorted_sub_tasks, action_history, grabbed_count, oppo_room, current_room):
        """
        Determines the most efficient action to take in a cooperative household task based on task relevance and proximity.

        Parameters:
            sorted_sub_tasks (list): List of possible sub-tasks sorted by priority.
            action_history (list): List of recent actions taken.
            grabbed_count (int): Number of objects currently being held.
            oppo_room (str): The room where the collaborator is located.
            current_room (str): The room where the agent is located.

        Returns:
            dict: The chosen task that is most efficient to achieve the goal.
        """

        prompt = f"Me and {self.oppo_name} should do household together. " \
                 f"Our goal is \'{self.goal_desc}\'. \n" \
                 f"I am in the {current_room}, and I think {self.oppo_name} is presumably in the {oppo_room}. \n" \
                 f"I can hold up to 2 objects in my hand, and I am currently holding {grabbed_count} objects. " \
                 f"I have just done the following: {action_history[-5:] if len(action_history) >= 5 else action_history} \n"

        prompt += f"I can perform the following actions: \n"
        for idx, my_task in enumerate(sorted_sub_tasks):
            relativity = ""
            if my_task['my_property'] == 1:
                relativity = "Strong relevance"
            elif my_task['my_property'] == 0.5:
                relativity = "Medium relevance"
            elif my_task['my_property'] == 0:
                relativity = "Low relevance"
            elif my_task['my_property'] == -1:
                relativity = "None relevance"

            option = chr(ord('A') + idx)
            if my_task['distance'] < my_task['oppo_distance']:
                distance = f"I'm closer than {self.oppo_name}"
            else:
                distance = f"I'm farther than {self.oppo_name}"
            action = f"[{my_task['action']}] <{my_task['name']}> ({my_task['id']}) - relevance score: {relativity}, relative proximity: {distance}"
            prompt += f"{option}. {action} \n"

        prompt += "Of the actions I can take, which are the most efficient to achieve the common goal? " \
                  "I want to improve my cooperative performance by performing actions that take into account the other person depending on the situation. " \
                  "And make it clear that each action can only be taken by one person. " \
                  "Remember, \'None relevancy\' means distracting from the goal. \n"

        if self.cot:
            cot_prompt = prompt
            cot_prompt += "\nAnswer: Let's think step by step."

            chat_prompt = [{"role": "user", "content": cot_prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            cot_output = outputs[0]

            # if self.debug:
            #     print("--------planning(self, sorted_sub_tasks, action_history, grabbed_count, oppo_room, current_room):--------")
            #     print("chat prompt:", cot_prompt)
            #     print("output:", cot_output)
            #     print("\n")

            prompt = "Choose the action I should take from the answers below. \n"
            for idx, my_task in enumerate(sorted_sub_tasks):
                relativity = ""
                if my_task['my_property'] == 1:
                    relativity = "Strong relevance"
                elif my_task['my_property'] == 0.5:
                    relativity = "Medium relevance"
                elif my_task['my_property'] == 0:
                    relativity = "Low relevance"
                elif my_task['my_property'] == -1:
                    relativity = "None relevance"

                option = chr(ord('A') + idx)
                if my_task['distance'] < my_task['oppo_distance']:
                    distance = f"I'm closer than {self.oppo_name}"
                else:
                    distance = f"I'm farther than {self.oppo_name}"
                action = f"[{my_task['action']}] <{my_task['name']}> ({my_task['id']}) - relevance score: {relativity}, relative proximity: {distance}"
                prompt += f"{option}. {action} \n"

            prompt += "Output format: \n" \
                      "Answer: alphabet"

            chat_prompt = [{"role": "user", "content": cot_prompt},
                           {"role": "assistant", "content": cot_output},
                           {"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            output = outputs[0].replace('Answer: ', '')

            # if self.debug:
            #     print("chat prompt:", prompt)
            #     print("output:", output)
            #     print("\n")
        else:
            prompt = "Choose the action I should take from the answers below. \n"
            for idx, my_task in enumerate(sorted_sub_tasks):
                relativity = ""
                if my_task['my_property'] == 1:
                    relativity = "Strong relevance"
                elif my_task['my_property'] == 0.5:
                    relativity = "Medium relevance"
                elif my_task['my_property'] == 0:
                    relativity = "Low relevance"
                elif my_task['my_property'] == -1:
                    relativity = "None relevance"

                option = chr(ord('A') + idx)
                if my_task['distance'] < my_task['oppo_distance']:
                    distance = f"I'm closer than {self.oppo_name}"
                else:
                    distance = f"I'm farther than {self.oppo_name}"
                action = f"[{my_task['action']}] <{my_task['name']}> ({my_task['id']}) - relevance score: {relativity}, relative proximity: {distance}"
                prompt += f"{option}. {action} \n"

            prompt += "Output format: \n" \
                      "Answer: alphabet. [action] <object_name> (object_id)"

            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            output = outputs[0].replace('Answer: ', '')

        # print(cot_prompt, cot_output)
        for idx, my_task in enumerate(sorted_sub_tasks):
            option = chr(ord('A') + idx)
            if option in output:
                # print(f"{self.agent_name} {option} [{my_task['action']}] <{my_task['name']}> ({my_task['id']})")
                return my_task

        return None

    def belief_validation(self, is_other_visit, my_plan, room, my_relativity, other_relativity, distance, filtered_communication_memory):
        """
        Validates the plan about whether the collaborator has already interacted with an object based on their trajectory and room visits.

        Parameters:
            is_other_visit (bool): Whether the collaborator visited the room.
            my_plan (str): The action plan involving the object.
            room (str): The room where the object is located.
            my_relativity (float): Relevance of the action to the goal for the agent.
            other_relativity (float): Relevance of the action to the goal for the collaborator.
            distance (float): The relative distance to the object.
            filtered_communication_memory (list): Communication log filtered to relevant events.

        Returns:
            bool: True if the collaborator likely interacted with the object, False otherwise.
        """

        if my_relativity == 1:
            my_relativity = "Strong relevance"
        elif my_relativity == 0.5:
            my_relativity = "Medium relevance"
        elif my_relativity == 0:
            my_relativity = "Low relevance"
        elif my_relativity == -1:
            my_relativity = "None relevance"

        act, name, id = my_plan.split(' ')
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

            # if self.debug:
            #     print("--------belief_validation(self, is_other_visit, my_plan, room, my_relativity, other_relativity, distance):--------")
            #     print("chat prompt:", cot_prompt)
            #     print("output:", cot_output)
            #     print("\n")

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

            # if self.debug:
            #     print("chat prompt:", prompt)
            #     print("output:", output)
            #     print("\n")

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