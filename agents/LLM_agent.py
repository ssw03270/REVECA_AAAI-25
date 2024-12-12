import json
import math
import os

from LLM import *
import random


class LLM_agent:
    """
    LLM agent class
    """

    def __init__(self, agent_id, char_index, args):
        self.debug = args.debug
        self.agent_type = 'LLM'
        self.agent_names = ["Zero", "Alice", "Bob"]
        self.agent_id = agent_id
        self.opponent_agent_id = 3 - agent_id
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = Ours_LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args,
                            self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}
        self.unsatisfied = {}
        self.steps = 0
        # self.location = None
        # self.last_location = None
        self.plan = []
        self.stuck = 0
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None
        self.opponent_grabbed_objects = []
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id2node = {}
        self.id_inside_room = {}
        self.satisfied = []
        self.reachable_objects = []
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.oppo = None
        self.change_action_able = True

        self.sub_task_memory = []
        self.communication_memory = []
        self.message = ""
        self.move_distance = 0
        self.move_history = []
        self.oppo_room_history = []
        self.task_id = 0

        self.last_communicate_room = ""

    @property
    def all_relative_name(self) -> list:
        return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']

    def goexplore(self):
        target_room_id = int(self.plan[0][1].split(' ')[-1][1:-1])
        if self.current_room['id'] == target_room_id:
            self.plan.pop(0)
            self.change_action_able = True
            return None
        return self.plan[0][1].replace('[goexplore]', '[walktowards]')

    def gocheck(self):
        assert len(self.grabbed_objects) < 2  # must have at least one free hands
        target_container_id = int(self.plan[0][1].split(' ')[-1][1:-1])
        target_container_name = self.plan[0][1].split(' ')[1]

        if target_container_id not in self.id_inside_room:
            target_container_room = self.oppo.id_inside_room[target_container_id]
        else:
            target_container_room = self.id_inside_room[target_container_id]
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]
        if 'OPEN' in target_container['states']:
            self.plan.pop(0)
            self.change_action_able = True
            return None
        if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan[0][1].replace('[gocheck]', '[open]')  # conflict will work right
        else:
            return self.plan[0][1].replace('[gocheck]', '[walktowards]')

    def gograb(self):
        target_object_id = int(self.plan[0][1].split(' ')[-1][1:-1])
        target_object_name = self.plan[0][1].split(' ')[1]
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan.pop(0)
            self.change_action_able = True
            return None
        assert len(self.grabbed_objects) < 2  # must have at least one free hands

        if target_object_id not in self.id_inside_room:
            target_object_room = self.oppo.id_inside_room[target_object_id]
        else:
            target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[
            target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.plan.pop(0)

            for idx in range(len(self.sub_task_memory)):
                if self.sub_task_memory[idx]['id'] == target_object_id:
                    self.sub_task_memory[idx]['efficiency'] = 0

            self.change_action_able = True
            return None
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan[0][1].replace('[gograb]', '[grab]')
        else:
            return self.plan[0][1].replace('[gograb]', '[walktowards]')

    def goput(self):
        # if len(self.progress['goal_location_room']) > 1: # should be ruled out
        if len(self.grabbed_objects) == 0:
            self.plan = []
            self.change_action_able = True
            return None
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                # print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        x = self.id2node[self.grabbed_objects[0]]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and 'CLOSED' in y['states']:
                return self.plan[0][1].replace('[goput]', '[open]')
            else:
                self.message = f"Finished one of our subgoals, <{x['class_name']}> ({x['id']})."
                self.message = self.LLM_subgoal(self.message)
                self.satisfied.append({'id': x['id'], 'class_name': x['class_name']})
                for key in list(self.unsatisfied.keys()):
                    if x['class_name'] in key:
                        if self.unsatisfied[key] > 1:
                            self.unsatisfied[key] -= 1
                        else:
                            self.unsatisfied.pop(key)
                        break
                action = '[putin]'
        else:
            self.message = f"Finished one of our subgoals, <{x['class_name']}> ({x['id']})."
            self.message = self.LLM_subgoal(self.message)
            self.satisfied.append({'id': x['id'], 'class_name': x['class_name']})
            for key in list(self.unsatisfied.keys()):
                if x['class_name'] in key:
                    if self.unsatisfied[key] > 1:
                        self.unsatisfied[key] -= 1
                    else:
                        self.unsatisfied.pop(key)
                    break
            action = '[putback]'
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"

    def LLM_init(self, message):
        return self.LLM.run_init(message)
    def LLM_parse_init(self, message):
        return self.oppo.LLM.run_parse_init(message)
    def LLM_request(self, category, my_plan, my_current_room):
        return self.LLM.run_request(category, my_plan, my_current_room)

    def LLM_parse_request(self, message):
        return self.oppo.LLM.run_parse_request(message, plan_list=self.action_history)

    def LLM_response(self, category, oppo_current_room):
        if len(self.action_history) == 0:
            current_pln = "I don't have any plan now."
        else:
            current_pln = self.action_history[-1]

        grabbed_object = []
        for idd in self.oppo.grabbed_objects:
            grabbed_object.append(f"<{self.oppo.id2node[idd]['class_name']}> ({idd})")
        return self.oppo.LLM.run_response(category, self.oppo.sub_task_memory, self.oppo.plan, oppo_current_room, current_plan=current_pln, grabbed_objects=grabbed_object)

    def LLM_parse_response(self, message):
        return self.oppo.LLM.run_parse_response(message)
    def LLM_subgoal(self, message):
        return self.oppo.LLM.run_subgoal(message)
    def LLM_parse_subgoal(self):
        return self.LLM.run_parse_subgoal(self.oppo.message)

    def LLM_efficiency_weight(self, object_name, object_from, object_interaction, main):
        if main == "me":
            for memory in self.sub_task_memory:
                if memory['name'] == object_name:
                    return memory['my_property']
        else:
            for memory in self.oppo.sub_task_memory:
                if memory['name'] == object_name:
                    return memory['oppo_property']

        interaction = []
        if 'CONTAINERS' in object_interaction:
            interaction.append('CONTAINERS')
        elif 'GRABBABLE' in object_interaction:
            interaction.append('GRABBABLE')

        return self.LLM.run_efficiency_weight(object_name, object_from, interaction)

    def run_belief_validation(self, is_other_visit, my_plan, room, my_relativity, other_relativity, distance, filtered_communication_memory):
        return self.LLM.belief_validation(is_other_visit, my_plan, room, my_relativity, other_relativity, distance, filtered_communication_memory)

    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and \
                        id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied

    def filter_graph(self, obs):
        relative_id = []
        for node in obs['nodes']:
            if node['class_name'] in self.all_relative_name:
                relative_id.append(node['id'])

        relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
        new_graph = {
            "edges": [edge for edge in obs['edges'] if
                      edge['from_id'] in relative_id and edge['to_id'] in relative_id],
            "nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
        }

        return new_graph

    def LLM_planning(self, sorted_sub_tasks):
        return self.LLM.planning(sorted_sub_tasks, self.action_history, len(self.grabbed_objects), self.oppo_room_history[-1], self.current_room['class_name'])

    def get_action(self, observation, goal):
        """
        :param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
        "nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
        "messages": [None, None]
        }
        :param goal:{predicate:[count, True, 2]}
        :return:
        """
        oppo_hisotry_check = False

        if self.steps == 0:
            self.oppo_room_history.append(self.oppo.current_room['class_name'])
            oppo_hisotry_check = True

        if self.oppo.message != "":
            self.communication_memory.append([self.steps, self.oppo.message])
            self.oppo_room_history.append(self.oppo.current_room['class_name'])
            oppo_hisotry_check = True

            output = self.LLM_parse_subgoal()
            try:
                object_name, object_id = output.split('\n')
                object_id = int(object_id)
                if not any(d == {'id': object_id, 'class_name': object_name} for d in self.satisfied):
                    self.satisfied.append({'id': object_id, 'class_name': object_name})
                    for key in list(self.unsatisfied.keys()):
                        if object_name in key:
                            if self.unsatisfied[key] > 1:
                                self.unsatisfied[key] -= 1
                            else:
                                self.unsatisfied.pop(key)
                            break

                self.oppo.message = ""
            except:
                print("error: ", output)
                self.oppo.message = ""

        for e in observation['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y]

        obs = self.filter_graph(observation)

        self.grabbed_objects = []
        opponent_grabbed_objects = []
        self.reachable_objects = []
        self.id2node = {x['id']: x for x in obs['nodes']}
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y]
                elif r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
                opponent_grabbed_objects.append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []
        for x in obs['nodes']:
            if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x['id'] == ungrab['id']:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x['id']] = self.current_room['class_name']
            if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
                unchecked_containers.append(x)
            if all(
                    [x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x[
                'id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
                ungrabbed_objects.append(x)

        if type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in \
                self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': obs,
                "obs": {
                    "grabbed_objects": self.grabbed_objects,
                    "opponent_grabbed_objects": opponent_grabbed_objects,
                    "reachable_objects": self.reachable_objects,
                    "progress": {
                        "unchecked_containers": self.unchecked_containers,
                        "ungrabbed_objects": self.ungrabbed_objects,
                    },
                    "satisfied": self.satisfied,
                    "current_room": self.current_room['class_name'],
                },
                }
        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects

        agent_node = None
        oppo_node = None
        for node in obs['nodes']:
            if node['id'] == self.agent_id:
                agent_node = node
            if node['id'] == self.oppo.agent_id:
                oppo_node = node

        if agent_node == None:
            return None, info

        if len(self.grabbed_objects) >= 2 and len(self.plan) > 0:
            act, name, id = self.plan[0][1].split(' ')
            if 'check' in act or 'grab' in act:
                self.plan.pop(0)

        for idx in range(len(self.sub_task_memory)):
            if self.sub_task_memory[idx]["id"] == self.current_room['id']:
                self.sub_task_memory[idx]["efficiency"] = 0
                break
        for idx in range(len(self.sub_task_memory)):
            if self.sub_task_memory[idx]["name"] == self.oppo_room_history[-1]:
                self.sub_task_memory[idx]["oppo_efficiency"] = 0
                break

        # update memory
        for node in obs['nodes']:
            is_already_in = False
            for idx in range(len(self.sub_task_memory)):
                if self.sub_task_memory[idx]["id"] == node['id']:
                    is_already_in = True

                    if self.sub_task_memory[idx]["efficiency"] != 0:
                        if node['class_name'] in self.containers_name and node['id'] != self.goal_location_id:
                            self.sub_task_memory[idx]["efficiency"] = 1 if "CLOSED" in node["states"] else 0
                            break

                        if 'GRABBABLE' in node['properties']:
                            self.sub_task_memory[idx]["efficiency"] = 1 if node['id'] not in self.grabbed_objects and \
                                                                           all([node['id'] != y['id'] for y in self.satisfied]) and \
                                                                           node['id'] not in [oppo_grab['id'] for oppo_grab in opponent_grabbed_objects] else 0
                            self.sub_task_memory[idx]['room'] = self.current_room['class_name']
                            break

            if not is_already_in:
                if node['class_name'] in self.containers_name and node['id'] != self.goal_location_id:
                    property = self.LLM_efficiency_weight(object_name=node['class_name'], object_from=self.current_room['class_name'], object_interaction=node['properties'], main="me")
                    self.sub_task_memory.append({"id": node['id'], "name": node['class_name'], "step": self.steps,
                                                 "efficiency": 1 if "CLOSED" in node["states"] else 0,
                                                 "position": [0, 0, 0], 'action': 'gocheck', 'room': self.current_room['class_name'], 'my_property': property, 'oppo_property': property})

                elif 'GRABBABLE' in node['properties']:
                    property = self.LLM_efficiency_weight(object_name=node['class_name'], object_from=self.current_room['class_name'], object_interaction=node['properties'], main="me")
                    self.sub_task_memory.append({"id": node['id'], "name": node['class_name'], "step": self.steps,
                                                 "efficiency": 1 if node['id'] not in self.grabbed_objects and all([node['id'] != y['id'] for y in self.satisfied]) and node['id'] not in [oppo_grab['id'] for oppo_grab in opponent_grabbed_objects] else 0,
                                                 "position": [0, 0, 0], 'action': 'gograb', 'room': self.current_room['class_name'], 'my_property': property, 'oppo_property': property})

                if node['id'] == self.goal_location_id:
                    property = self.LLM_efficiency_weight(object_name=node['class_name'], object_from=self.current_room['class_name'], object_interaction=node['properties'], main="me")
                    self.sub_task_memory.append({"id": node['id'], "name": node['class_name'], "step": self.steps,
                                                 "efficiency": 1,
                                                 "position": [0, 0, 0], 'action': 'goput', 'room': self.current_room['class_name'], 'my_property': property, 'oppo_property': property})

        room_position = {}
        for node in obs['nodes']:
            if node['class_name'] in ['livingroom', 'kitchen', 'bathroom', 'bedroom']:
                if node['class_name'] not in room_position:
                    room_position[node['class_name']] = node["obj_transform"]["position"]

        for node in obs['nodes']:
            for idx in range(len(self.sub_task_memory)):
                if self.sub_task_memory[idx]['id'] == node['id']:
                    self.sub_task_memory[idx]['position'] = node["obj_transform"]["position"]
                    if self.sub_task_memory[idx]['action'] == 'goexplore' and self.sub_task_memory[idx]['name'] != self.current_room['class_name']:
                        continue
                    self.sub_task_memory[idx]['step'] = self.steps

        for idx in range(len(self.sub_task_memory)):
            if self.sub_task_memory[idx]["action"] == "gograp":
                self.sub_task_memory[idx]["efficiency"] = 1 if self.sub_task_memory[idx]['id'] not in self.grabbed_objects and \
                                                               all([self.sub_task_memory[idx]['id'] != y['id'] for y in self.satisfied]) and \
                                                               any([self.sub_task_memory[idx]['name'] == g.split('_')[1] for g in self.unsatisfied]) and \
                                                               self.sub_task_memory[idx]['id'] not in [ oppo_grab['id'] for oppo_grab in opponent_grabbed_objects] else 0

        agent_position = agent_node["obj_transform"]["position"]
        oppo_position = room_position[self.oppo_room_history[-1]] if oppo_node == None else oppo_node["obj_transform"]["position"]
        for idx in range(len(self.sub_task_memory)):
            target_position = self.sub_task_memory[idx]["position"]
            distance = math.sqrt((agent_position[0] - target_position[0]) ** 2 +
                                 (agent_position[2] - target_position[2]) ** 2) + 1
            oppo_distance = math.sqrt((oppo_position[0] - target_position[0]) ** 2 +
                                 (oppo_position[2] - target_position[2]) ** 2) + 1
            if self.sub_task_memory[idx]['efficiency'] != 0:
                self.sub_task_memory[idx]['distance'] = distance - 1
                if self.sub_task_memory[idx]['my_property'] == 1:
                    self.sub_task_memory[idx]['efficiency'] = 1 / distance + 1000
                elif self.sub_task_memory[idx]['my_property'] == 0.5:
                    self.sub_task_memory[idx]['efficiency'] = 1 / distance + 100
                elif self.sub_task_memory[idx]['my_property'] == 0:
                    self.sub_task_memory[idx]['efficiency'] = 1 / distance + 10
                elif self.sub_task_memory[idx]['my_property'] == -1:
                    self.sub_task_memory[idx]['efficiency'] = 1 / distance + 1

                self.sub_task_memory[idx]['oppo_distance'] = oppo_distance - 1
                if self.sub_task_memory[idx]['oppo_property'] == 1:
                    self.sub_task_memory[idx]['oppo_efficiency'] = 1 / oppo_distance + 1000
                elif self.sub_task_memory[idx]['oppo_property'] == 0.5:
                    self.sub_task_memory[idx]['oppo_efficiency'] = 1 / oppo_distance + 100
                elif self.sub_task_memory[idx]['oppo_property'] == 0:
                    self.sub_task_memory[idx]['oppo_efficiency'] = 1 / oppo_distance + 10
                elif self.sub_task_memory[idx]['oppo_property'] == -1:
                    self.sub_task_memory[idx]['oppo_efficiency'] = 1 / oppo_distance + 1

        latest_position = self.move_history[-1]
        distance = math.sqrt((agent_position[0] - latest_position[0]) ** 2 +
                             (agent_position[2] - latest_position[2]) ** 2)
        self.move_distance += distance
        self.move_history.append(self.id2node[self.agent_id]["obj_transform"]["position"])

        if self.steps == 0:
            message_list = ""
            for memory in self.sub_task_memory:
                if memory['action'] != 'goexplore':
                    is_already_in = False
                    for oppo_memory in self.oppo.sub_task_memory:
                        if oppo_memory['id'] == memory['id']:
                            is_already_in = True
                            break
                    if not is_already_in:
                        self.oppo.sub_task_memory.append(memory)
                        if memory['name'] not in message_list:
                            message_list += f"<{memory['name']}> ({memory['id']}), "
            if message_list != "":
                message = f"I found {message_list}in the {self.current_room['class_name']}."
                message = self.LLM_init(message)
                output = self.LLM_parse_init(message)
                self.communication_memory.append([self.steps, message])
            action = None
            self.steps += 1
            return action, info

        action = None
        while action is None:
            if len(self.plan) == 0:
                max_sub_task = None
                k = 3

                filtered_memory = []
                for memory in self.sub_task_memory:
                    if len(self.grabbed_objects) == 2 and memory['action'] in ['gocheck', 'gograb']:
                        continue
                    if len(self.grabbed_objects) == 0 and memory['action'] == 'goput':
                        continue
                    if memory['efficiency'] != 0 and memory['efficiency'] > memory['oppo_efficiency']:
                        filtered_memory.append(memory)
                sorted_memory = sorted(filtered_memory, key=lambda x: x['efficiency'], reverse=True)
                filtered_memory = []
                for memory in self.sub_task_memory:
                    if len(self.grabbed_objects) == 2 and memory['action'] in ['gocheck', 'gograb']:
                        continue
                    if len(self.grabbed_objects) == 0 and memory['action'] == 'goput':
                        continue
                    if memory['efficiency'] != 0 and memory['efficiency'] <= memory['oppo_efficiency']:
                        filtered_memory.append(memory)
                sorted_memory2 = sorted(filtered_memory, key=lambda x: x['efficiency'], reverse=True)
                while len(sorted_memory) < k:
                    if len(sorted_memory2) > 0:
                        sorted_memory.append(sorted_memory2[0])
                        sorted_memory2.pop(0)
                    else:
                        break
                filtered_memory = []
                for memory in self.sub_task_memory:
                    if len(self.grabbed_objects) == 2 and memory['action'] in ['gocheck', 'gograb']:
                        continue
                    if len(self.grabbed_objects) == 0 and memory['action'] == 'goput':
                        continue
                    if memory['efficiency'] != 0 and memory['my_property'] == -1:
                        filtered_memory.append(memory)
                sorted_memory2 = sorted(filtered_memory, key=lambda x: x['efficiency'], reverse=True)
                while len(sorted_memory) < k:
                    if len(sorted_memory2) > 0:
                        sorted_memory.append(sorted_memory2[0])
                        sorted_memory2.pop(0)
                    else:
                        break

                if len(sorted_memory) > k:
                    sorted_memory = sorted_memory[:k]

                if len(sorted_memory) <= 1:
                    if len(sorted_memory) == 1:
                        max_sub_task = sorted_memory[0]
                    elif len(sorted_memory) == 0:
                        max_sub_task = None
                else:
                    max_sub_task = self.LLM_planning(sorted_memory)

                if max_sub_task != None:
                    false_belief = False
                    for i in range(max_sub_task['step'], self.steps):
                        if len(self.oppo_room_history) > i:
                            if self.oppo_room_history[i] == max_sub_task['room']:
                                false_belief = True
                    filtered_communication_memory = []
                    for i in range(0, len(self.communication_memory)):
                        if self.communication_memory[i][0] >= max_sub_task['step'] and self.communication_memory[i][0] <= self.steps:
                            filtered_communication_memory.append(self.communication_memory[i][1])

                    my_plan = f"[{max_sub_task['action']}] <{max_sub_task['name']}> ({max_sub_task['id']})"

                    if max_sub_task['room'] != self.current_room['class_name'] and self.last_communicate_room != max_sub_task['room'] and max_sub_task['action'] != 'goput':
                        is_need_comm = self.run_belief_validation(is_other_visit=false_belief, my_plan=my_plan,
                                                                  room=max_sub_task['room'], distance=max_sub_task['distance'] < max_sub_task['oppo_distance'],
                                                                  my_relativity=max_sub_task['my_property'],
                                                                  other_relativity=max_sub_task['oppo_property'],
                                                                  filtered_communication_memory=filtered_communication_memory)
                    else:
                        is_need_comm = False

                    if is_need_comm:
                        self.last_communicate_room = max_sub_task['room']

                        my_current_room = f"And currently I'm in the {self.current_room['class_name']}."
                        message = self.LLM_request(max_sub_task['room'], my_plan, my_current_room)
                        category = self.LLM_parse_request(message)

                        oppo_current_room = f"And currently I'm in the {self.oppo.current_room['class_name']}."
                        oppo_message = self.LLM_response(category, oppo_current_room)
                        self.oppo_room_history.append(self.oppo.current_room['class_name'])
                        oppo_hisotry_check = True

                        self.oppo.communication_memory.append([self.steps, message])
                        self.communication_memory.append([self.steps, oppo_message])

                        if "There's nothing more to do there." in oppo_message:
                            data = []

                        elif "I don't have any information about it yet." in oppo_message:
                            for idx in range(len(self.sub_task_memory)):
                                if self.sub_task_memory[idx]['room'] == category:
                                    self.sub_task_memory[idx]['step'] = self.steps
                            break

                        else:
                            parsed_data = self.LLM_parse_response(oppo_message)
                            try:
                                data = json.loads(parsed_data)
                            except:
                                break

                        for idx in range(len(self.sub_task_memory)):
                            is_already_in = False
                            for d in data:
                                if self.sub_task_memory[idx]['id'] == d['id']:
                                    is_already_in = True
                                    break

                            if not is_already_in and self.sub_task_memory[idx]['room'] == category:
                                self.sub_task_memory[idx]['efficiency'] = 0

                        for d in data:
                            is_already_in = False
                            for idx in range(len(self.sub_task_memory)):
                                if self.sub_task_memory[idx]['id'] == d['id']:
                                    self.sub_task_memory[idx]['step'] = self.steps
                                    is_already_in = True
                                    break

                            if not is_already_in:
                                if d['name'] in self.containers_name and d['id'] != self.goal_location_id:
                                    property = self.LLM_efficiency_weight(object_name=d['name'], object_from=category, object_interaction=['CONTAINERS'], main="me")
                                    self.sub_task_memory.append(
                                        {"id": d['id'], "name": d['name'], "step": self.steps,
                                         "efficiency": 1,
                                         "position": room_position[category], 'action': 'gocheck',
                                         'room': category, 'my_property': property, 'oppo_property': property})
                                elif d['id'] == self.goal_location_id:
                                    property = self.LLM_efficiency_weight(object_name=d['name'], object_from=category, object_interaction=['GRABBABLE'], main="me")
                                    self.sub_task_memory.append(
                                        {"id": d['id'], "name": d['name'], "step": self.steps,
                                         "efficiency": 1,
                                         "position": room_position[category], 'action': 'goput', 'room': category, 'my_property': property, 'oppo_property': property})
                                else:
                                    property = self.LLM_efficiency_weight(object_name=d['name'], object_from=category, object_interaction=[], main="me")
                                    self.sub_task_memory.append(
                                        {"id": d['id'], "name": d['name'], "step": self.steps,
                                         "efficiency": 1,
                                         "position": room_position[category], 'action': 'gograb',
                                         'room': category, 'my_property': property, 'oppo_property': property})
                        break

                    self.last_communicate_room = ""
                    choose_action = f"[{max_sub_task['action']}] <{max_sub_task['name']}> ({max_sub_task['id']})"
                    self.plan = [['', choose_action, '']]
                else:
                    print(self.LLM.agent_name, 'I dont have any plan')
                    break

            if self.plan[0][1].startswith('[goexplore]'):
                if self.action_history[-1] != self.plan[0][1]:
                    self.action_history.append(self.plan[0][1])
                action = self.goexplore()
            elif self.plan[0][1].startswith('[gocheck]'):
                if self.action_history[-1] != self.plan[0][1]:
                    self.action_history.append(self.plan[0][1])
                action = self.gocheck()
            elif self.plan[0][1].startswith('[gograb]'):
                if self.action_history[-1] != self.plan[0][1]:
                    self.action_history.append(self.plan[0][1])
                action = self.gograb()
            elif self.plan[0][1].startswith('[goput]'):
                if self.action_history[-1] != self.plan[0][1]:
                    self.action_history.append(self.plan[0][1])
                action = self.goput()
            else:
                raise ValueError(f"unavailable plan {self.plan[0][1]}")

        self.steps += 1
        info.update({"plan": self.plan,
                     })
        if action == self.last_action and self.current_room['class_name'] == self.last_room:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_action = action
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.plan = []
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walktowards] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0

        if not oppo_hisotry_check:
            self.oppo_room_history.append(self.oppo_room_history[-1])

        return action, info

    def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
        self.steps = 0
        self.containers_name = containers_name
        self.goal_objects_name = goal_objects_name
        self.rooms_name = rooms_name
        self.roomname2id = {x['class_name']: x['id'] for x in room_info}
        self.id2node = {x['id']: x for x in obs['nodes']}
        self.stuck = 0
        self.last_room = None
        self.unsatisfied = {k: v[0] for k, v in goal.items()}
        self.satisfied = []
        self.goal_location = list(goal.keys())[0].split('_')[-1]
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.id_inside_room = {self.goal_location_id: self.rooms_name[:], self.opponent_agent_id: None}
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.opponent_grabbed_objects = []
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id and r == 'INSIDE':
                self.current_room = self.id2node[y]
        self.plan = []
        self.change_action_able = True
        self.action_history = [f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []
        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)

        livingroom_property = self.LLM_efficiency_weight(object_name="livingroom",
                                                 object_from="livingroom",
                                                 object_interaction="goexplore", main="me")
        kitchen_property = self.LLM_efficiency_weight(object_name="kitchen",
                                                 object_from="kitchen",
                                                 object_interaction="goexplore", main="me")
        bedroom_property = self.LLM_efficiency_weight(object_name="bedroom",
                                                 object_from="bedroom",
                                                 object_interaction="goexplore", main="me")
        bathroom_property = self.LLM_efficiency_weight(object_name="bathroom",
                                                 object_from="bathroom",
                                                 object_interaction="goexplore", main="me")

        self.sub_task_memory = [{"id": self.roomname2id["livingroom"], "name": "livingroom", "step": 0,
                                "efficiency": 1, "position": [0, 0, 0], 'action': 'goexplore', 'room': 'livingroom',
                                 'my_property': livingroom_property, 'oppo_property': livingroom_property},
                               {"id": self.roomname2id["kitchen"], "name": "kitchen", "step": 0,
                                "efficiency": 1, "position": [0, 0, 0], 'action': 'goexplore', 'room': 'kitchen',
                                'my_property': kitchen_property, 'oppo_property': kitchen_property},
                               {"id": self.roomname2id["bedroom"], "name": "bedroom", "step": 0,
                                "efficiency": 1, "position": [0, 0, 0], 'action': 'goexplore', 'room': 'bedroom',
                                'my_property': bedroom_property, 'oppo_property': bedroom_property},
                               {"id": self.roomname2id["bathroom"], "name": "bathroom", "step": 0,
                                "efficiency": 1, "position": [0, 0, 0], 'action': 'goexplore', 'room': 'bathroom',
                                'my_property': bathroom_property, 'oppo_property': bathroom_property}]

        self.communication_memory = []
        self.message = ""
        self.move_distance = 0
        self.move_history = []
        self.oppo_room_history = []

        self.move_history.append(self.id2node[self.agent_id]["obj_transform"]["position"])

        self.task_id += 1
        self.last_communicate_room = ""