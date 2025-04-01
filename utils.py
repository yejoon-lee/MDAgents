import os
import json
import random
from typing import List
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
from openai import OpenAI
from pptree import *

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro':
            raise NotImplementedError("Gemini Pro is not supported.")
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                responses[temperature] = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses

class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info='gpt-4o-mini')
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info
    
def ask_fpfn(question, model):
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the relative cost between false negative and false positive', role='medical expert', model_info=model)
    cprint(f"Question: {question}", 'green', attrs=['bold'])
    
    response = medical_agent.chat(question)
    print(response)
    return response

def load_fpfn(task : str) -> str:
    TEMPLATE_EHRSHOT = "In predicting a prognosis of {task}, which is more costly, a false negative(missing the disease) or a false positive(wrongly predicting the disease)?"
    TEMPLATE_BMAD = "Assume a patient is going through {test}. Diseases and conditions such as {diseases} can be found. In such scenario, which is more costly, a false negative(missing the disease) or a false positive(wrongly predicting the disease)?"

    match task.split('-'):
        case ['ehrshot', _]:
            return TEMPLATE_EHRSHOT.format(task=task.split('-')[1].replace('_', ' '))
        case ['bmad', 'brats2021']:
            return TEMPLATE_BMAD.format(test='brain MRI', diseases='glioblastoma and lower-grade gliomas')
        case ['bmad', 'hist']:
            return TEMPLATE_BMAD.format(test='liver CT', diseases='hepatocellular carcinoma, liver metastases, liver cysts, hemangiomas, hepatic adenomas, focal nodular hyperplasia, fatty liver disease, liver cirrhosis, and hepatitis-related liver damage')
        case ['bmad', 'resc']:
            return TEMPLATE_BMAD.format(test='retinal OCT', diseases='diabetic retinopathy, hypertensive retinopathy, retinal vein occlusion, retinal artery occlusion, macular edema, glaucoma, and age-related macular degeneration')
        case ['bmad', 'oct2017']:
            return TEMPLATE_BMAD.format(test='retinal OCT', diseases='choroidal neovascularization, diabetic macular edema, and drusen')
        case ['bmad', 'rsna']:
            return TEMPLATE_BMAD.format(test='chest X-ray', diseases='pneumonia, lung opacity, atelectasis, pulmonary edema, pleural effusion, and pneumothorax')
        case ['bmad', 'camelyon16']:
            return TEMPLATE_BMAD.format(test='pathology test', diseases='breast cancer metastases')
        case ['baseline', 'death']:
            return "In predicting the death of a patient, which is more costly, a false negative(missing the death) or a false positive(wrongly predicting the death)?"
        case ['baseline', 'benign']:
            return "In predicting a prognosis of a mild cough, which is more costly, a false negative(missing the symptoms) or a false positive(wrongly predicting the symptoms)?"
        case _:
            raise ValueError(f"Unsupported task: {task}")
        

def load_relative_cost(task : str) -> str:
    TEMPLATE_EHRSHOT = "In predicting a prognosis of {task}, how much more costly is a false negative(missing the disease) to a false positive(wrongly predicting the disease)?"
    TEMPLATE_BMAD = "Assume a patient is going through {test}. Diseases and conditions such as {diseases} can be found. In such scenario, how much more costly is a false negative(missing the disease) to a false positive(wrongly predicting the disease)?"

    if task in ['ehrshot-hypertension', 'ehrshot-hyperlipidemia', 'ehrshot-pancreatic_cancer', 'ehrshot-celiac', 'ehrshot-lupus', 'ehrshot-acute_myocardial_infarction']:
        disease = task.split('-')[1]  # Extract disease name from task string
        disease = disease.replace('_', ' ')
        return TEMPLATE_EHRSHOT.format(task=disease)
    
    elif task == 'bmad-brats2021':
        # Brain MRI
        test = 'brain MRI'
        diseases = ['glioblastoma', 'lower-grade gliomas']
        return TEMPLATE_BMAD.format(test=test, diseases=', and '.join(diseases))
    
    elif task == 'bmad-hist':
        # Liver CT
        test = 'liver CT'
        diseases = ['hepatocellular carcinoma', 'liver metastases', 'liver cysts', 'hemangiomas', 'hepatic adenomas', 'focal nodular hyperplasia', 'fatty liver disease', 'liver cirrhosis', 'hepatitis-related liver damage']
        return TEMPLATE_BMAD.format(test=test, diseases=', '.join(diseases[:-1]) + ', and ' + diseases[-1])
    
    elif task == 'bmad-resc':
        # Retinal OCT
        test = 'retinal OCT'
        diseases = ['diabetic retinopathy', 'hypertensive retinopathy', 'retinal vein occlusion', 'retinal artery occlusion', 'macular edema', 'glaucoma', 'age-related macular degeneration']
        return TEMPLATE_BMAD.format(test=test, diseases=', '.join(diseases[:-1]) + ', and ' + diseases[-1])
    
    elif task == 'bmad-oct2017':
        # Retinal OCT
        test = 'retinal OCT'
        diseases = ['choroidal neovascularization', 'diabetic macular edema', 'drusen']
        return TEMPLATE_BMAD.format(test=test, diseases=', '.join(diseases[:-1]) + ', and ' + diseases[-1])
    
    elif task == 'bmad-rsna':
        # Chest X-Ray
        test = 'chest X-ray'
        diseases = ['pneumonia', 'lung opacity', 'atelectasis', 'pulmonary edema', 'pleural effusion', 'pneumothorax']
        return TEMPLATE_BMAD.format(test=test, diseases=', '.join(diseases[:-1]) + ', and ' + diseases[-1])
    
    elif task == 'bmad-camelyon16':
        # Pathology
        test = 'pathology test'
        diseases = ['breast cancer metastases']
        return TEMPLATE_BMAD.format(test=test, diseases=diseases[0])
    
    elif task == 'baseline-death':
        return "In predicting the death of a patient, how much more costly is a false negative(missing the death) to a false positive(wrongly predicting the death)?"
    
    elif task == 'baseline-benign':
        return "In predicting a prognosis of a mild cough, how much more costly is a false negative(missing the symptoms) to a false positive(wrongly predicting the symptoms)?"
    
    else:
        raise ValueError(f"Unsupported task: {task}")

def ask_relative_cost(question, model, examplar):
    examplar_ = examplar

    cprint(f"Question: {question}", 'green', attrs=['bold'])
    print()
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-3.5')
    tmp_agent.chat(recruit_prompt)
    
    num_agents = 5  # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")

    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
        description = agent[0].split('-')[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 5
    num_turns = 5
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(f'''Given the examplers, please return the answer to the medical query in a single figure.\n\n{examplar_}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model)
        agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking a median. Your answer should be like below format:\nAnswer: 15\n{final_answer}\n\nQuestion: {question}", img_path=None)
    final_decision = {'majority': _decision}

    print("\nFinal answers from each agent:")
    for agent_name, answer in final_answer.items():
        print(f"- {agent_name}: {answer}")

    moderator_emoji = '\U0001F468\u200D\u2696\uFE0F'
    print(f"{moderator_emoji} moderator's final decision (by taking median):", _decision)
    print()

    return final_decision
