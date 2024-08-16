import numpy as np
import torch
import gymnasium as gym
from akt import AKT
import os
import csv
import random
import json

class PracticeProblemEnv(gym.Env):
    
    def __init__(self, params, max_step=10, QperS = 1, rew_func="mock", n_PperQ = 1, units=None, device="cuda"):
        super(PracticeProblemEnv, self).__init__()
        self.max_step = max_step
        self.QperS = QperS
        self.rew_func = rew_func
        self.n_PperQ = n_PperQ if (self.rew_func == "mock") else None

        self.params = params
        self.device = device
        self.kt_model = self._load_model()
        self.p_q_dict, self.q_p_dict = self._load_pq_qp_dict(units)

        self.actions = [*self.p_q_dict.keys()]

        # self.action_space = gym.spaces.Discrete(len(self.actions)) 
        self.action_space = gym.spaces.MultiDiscrete([len(self.actions)] * self.QperS) #[0,n_question-1]
        # self.observation_space = gym.spaces.Box(np.array([1,0,1]), np.array([self.params.n_question, 1, self.params.n_pid])) #[1,n_question]/[0,1]/[1,n_pid]
        self.observation_space = gym.spaces.Box(low=np.array([[low]*self.QperS for low in [1,0,1]]), 
                                                high=np.array([[high]*self.QperS for high in [self.params.n_question, 1, self.params.n_pid]]),
                                                shape=(3,self.QperS),
                                                type=np.int32
                                                ) 
   
        self.reset()
    
    def _load_model(self, pretrained_path="_b24_nb1_gn-1_lr1e-05_s224_sl200_do0.05_dm256_ts1_kq1_l21e-05_178"):
        model = AKT(n_question=self.params.n_question, n_pid=self.params.n_pid, n_blocks=self.params.n_block, d_model=self.params.d_model,
                    dropout=self.params.dropout, kq_same=self.params.kq_same, model_type='akt', l2=self.params.l2).to(self.device)
        checkpoint = torch.load(os.path.join( 'model', self.params.model, self.params.save, pretrained_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def reset(self, seed=None):
        self.history = {'q':[],'target':[],'pid':[]} # Initialize past interactions
        self.curr_step = 0
        self.curr_q = [-1] * self.QperS
        self.curr_pred = [-1] * self.QperS
        self.curr_pid = [-1] * self.QperS

        return self.step(self.action_space.sample())[0], {}

    def _get_obs(self):
        return np.array([self.curr_q, self.curr_pred, self.curr_pid], dtype=int)
        
    def _rew(self, n_PperQ=1):
        sampled_concpets = []
        sampled_problems = []
        for question_type, question_ids in self.q_p_dict.items():
            num = min(n_PperQ, len(question_ids))
            sampled_problems += random.sample([*question_ids], num)
            sampled_concpets += ([question_type ] * num)

        mean_performance = self.predict(sampled_concpets,sampled_problems)
        return mean_performance
    
    def switch_rew(self, new_rew_func):
        self.rew_func = new_rew_func
    
    def step(self, action):#action is an np int e.g. nparray(3) of the index of the action specified in self.actions
        self.curr_step += 1

        # action is np.array of length QperS 
        self.curr_pid = [self.actions[a] for a in action]
        self.curr_q = [self.p_q_dict[pid] for pid in self.curr_pid]
        self.curr_pred = []

        for q,pid in zip(self.curr_q,self.curr_pid):
            
            # Predict the probability of getting the question correct
            correct_prob = self.predict([q],[pid])
            pred = int(np.random.rand() < correct_prob)
            self.curr_pred += [pred]

            # Update history with the action and the correctness
            self.history['q'] += [q]
            self.history['target'] += [pred]
            self.history['pid'] += [pid]

        if self.rew_func == "mock":
            reward = self._rew(n_PperQ=self.n_PperQ)
        elif self.rew_func == "correct":
            reward = np.mean(self.curr_pred)
        else:
            raise NotImplementedError
            
        # Recompute the state using the kt_model for each question
        obs = self._get_obs()

        done = self.curr_step >= self.max_step
        good = (self.rew_func == "mock") and (reward > 0.9)

        return obs, reward, done, good, {}
    
    def predict(self, curr_q, curr_pid):

        batch_size = len(curr_q)

        q = torch.cat((torch.tensor(self.history['q'][-(self.params.seqlen-1):]).tile((batch_size,1)),(torch.tensor(curr_q).unsqueeze(-1))),1)
        target = torch.tensor(self.history['target'][-(self.params.seqlen-1):]+[0]).tile((batch_size,1))
        pid = torch.cat((torch.tensor(self.history['pid'][-(self.params.seqlen-1):]).tile((batch_size,1)),(torch.tensor(curr_pid).unsqueeze(-1))),1)
        assert pid.shape == target.shape == pid.shape #(test_n_problem,3)
        qa = q+target*self.params.n_question
        
        padded_q = torch.zeros((batch_size, self.params.seqlen))
        padded_qa = torch.zeros((batch_size, self.params.seqlen))
        padded_target = torch.full((batch_size,self.params.seqlen),-1)
        padded_pid = torch.zeros((batch_size, self.params.seqlen))

        pred_index = q.shape[1]
        padded_q[:, :pred_index]= q
        padded_qa[:, :pred_index]= qa
        padded_target[:, :pred_index]= target
        padded_pid[:, :pred_index]= pid

        q = padded_q.long().to(self.device)
        qa = padded_qa.long().to(self.device)
        target = padded_target.long().to(self.device)
        pid = padded_pid.long().to(self.device)

        with torch.no_grad():
            loss, pred, ct = self.kt_model(q,qa,target,pid)

        nopadding_index = np.flatnonzero(padded_target.reshape((-1,)) >= -0.9).tolist()
        pred_nopadding = pred[nopadding_index]

        test_result = pred_nopadding[(pred_index-1)::pred_index]
        assert test_result.shape == (batch_size,)
        correct_prob = test_result.mean().item()

        return correct_prob

    def _load_pq_qp_dict(self, units=None):
        def iterate_over_data(file_path):
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                rows = list(reader)

            for i in range(0, len(rows), 4):
                # Extract the question ids and concept ids
                # question_ids = [int(q) for q in rows[i+1] if q]
                # concept_ids = [int(c) for c in rows[i+2] if c]
                q_c_ids = [(int(q),int(c)) for (q,c) in zip(rows[i+1],rows[i+2]) if (q and c and ((units is None) or (int(c) in units)))]

                # Build the dictionary mapping question ids to concept ids
                for question_id, concept_id in q_c_ids:
                    if question_id not in p_q_dict:
                        p_q_dict[question_id] = concept_id
                    if concept_id not in q_p_dict:
                        q_p_dict[concept_id] = {question_id}
                    else:
                        q_p_dict[concept_id].add(question_id)


        p_q_file,  q_p_file,  = 'p_q_dict.json', 'q_p_dict.json'
        p_q_dict, q_p_dict = {}, {}

        # Check if files exist
        if os.path.exists(p_q_file) and os.path.exists(q_p_file):
            # Load the dictionaries from the files
            with open(p_q_file, 'r') as p_q_f:
                p_q_dict = json.load(p_q_f)
            with open(q_p_file, 'r') as q_p_f:
                q_p_dict = json.load(q_p_f)
            p_q_dict = {int(k):v for k,v in p_q_dict.items()}
            q_p_dict = {int(k):v for k,v in q_p_dict.items()}
        else:
            all_files = os.listdir(self.params.data_dir)

            # Filter the list to include only CSV files
            csv_files = [file for file in all_files if file.endswith('.csv')]


            for f in csv_files:
                old_dict = p_q_dict.copy()
                iterate_over_data(self.params.data_dir+'/'+ f)
                if old_dict == p_q_dict:
                    break
            
            q_p_dict = {k:list(v) for k,v in q_p_dict.items()}

            with open(p_q_file, 'w') as p_q_f:
                json.dump(p_q_dict, p_q_f)
            with open(q_p_file, 'w') as q_p_f:
                json.dump(q_p_dict, q_p_f)
        
        return p_q_dict, q_p_dict
    
    def check_hist(self):
        return self.history