import numpy as np
from typing import Union, List
from transformers import AutoTokenizer,LlamaTokenizer

import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
import h5py
from torchvision import transforms
import numpy as np

# ----------------------------
# 2. 你的 ActionTokenizer 类
# ----------------------------
class ActionTokenizer:
    def __init__(
        self, tokenizer: AutoTokenizer, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        将连续的机器人动作离散化为每个维度 N 个 bin，并映射到分词器词汇表中最少使用的 token 上。
        
        默认假设使用的分词器为类似 BPE 的，例如 LlamaTokenizer，
        且词汇表末尾的 token 较少使用，可以用于存储动作信息。

        :param tokenizer: 要扩展的预训练分词器
        :param bins: 每个连续动作值离散化时的 bin 数量（采用均匀分箱策略）
        :param min_action: 动作的最小值（用于裁剪）
        :param max_action: 动作的最大值（用于裁剪）
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # 生成均匀分箱边界（例如 256 个点，代表 255 个区间）
        self.bins = np.linspace(min_action, max_action, self.n_bins)
       
        # 计算各个 bin 的中心值（共 n_bins-1 个 bin 中心）
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        

        # 根据约定，预留词汇表中最后 n_bins 个 token 来表示离散动作，
        # 这里计算动作 token 开始的索引（假设越靠后 token 越少使用）
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))
        
    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        将连续动作先裁剪到 [min_action, max_action] 范围，
        然后利用 np.digitize 进行分箱，最后将 bin 索引映射为词汇表中的 token id，
        并调用 tokenizer 的 decode 方法将 token id 转换为字符串。

        :param action: 连续动作数组（可以是 1 维，也可以是批量的 2 维数组）
        :return: 解码后的字符串或字符串列表
        """
        # 裁剪动作值到指定范围内
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        # 利用 np.digitize 将动作值映射到 bin 索引
        discretized_action = np.digitize(action, self.bins)
        
        # 注意：np.digitize 返回的索引范围是 [1, n_bins]

        # 将离散 bin 索引映射到 tokenizer 词汇表中“末尾” token 的 id
        # 即 token id = tokenizer.vocab_size - discretized_action
        if len(discretized_action.shape) == 1:
            # 单个动作向量
            token_ids = list(self.tokenizer.vocab_size - discretized_action)
            
            decoded = self.tokenizer.decode(token_ids)
            return decoded
        else:
            # 批量动作，每一行表示一个动作向量
            token_ids_batch = (self.tokenizer.vocab_size - discretized_action).tolist()
            return self.tokenizer.batch_decode(token_ids_batch)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        将离散动作 token id 数组转换回连续动作值（这里返回的是各 bin 的中心值）。
        由于 np.digitize 返回的索引是 [1, n_bins]，实际 bin 数量为 n_bins-1，
        因此需要对最高的索引做裁剪处理。

        :param action_token_ids: 离散动作的 token id 数组
        :return: 连续动作值数组（来自于 bin_centers）
        """
        # 反向映射：得到离散化的 bin 索引
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        # 调整：减 1 后将值限制在合法范围内（0 到 bin_centers 的最大索引）
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
    
class HDF5Dataset(Dataset):
    def __init__(self,file_path:str,transform,tokenizer,action_tokenizer,max_length:int):
        self.file_path=file_path
        self.transform=transform.Compose([transform.Resize((224, 224)),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])])
        self.tokenizer=tokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.action_tokenizer=action_tokenizer(tokenizer=self.tokenizer,bins=256,min_action=-1,max_action=1)
        
        with h5py.File(file_path, "r") as f:
            # self.problem_info = json.loads(f["data"].attrs["problem_info"])
          
            # self.language_instruction = "".join(self.problem_info["language_instruction"])
            # self.len = f["data/demo_0"].attrs["num_samples"]
            
            self.len=len(f["data/description"])
            self.max_length=f.attrs["max_length"]+2
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
         # 每次取样时单独打开文件，确保多进程时不会共享句柄
        with h5py.File(self.file_path, "r") as f:
            
            image=f["data/rgb"][index]
            joint_states=f["data/joints"][index]
            prompt=f["data/description"][index]
            
            
            prompt=prompt[0].decode("utf-8")
            
            
            # image = f["data/demo_0/obs/agentview_rgb"][index]
            # joint_states = f["data/demo_0/obs/ee_states"][index]
            # 将 numpy 数组转换为 tensor
            image=self.transform(Image.fromarray(image))
            
            token_actions=self.action_tokenizer(joint_states)
            
            target_actions=self.tokenizer(token_actions,return_tensors="pt",max_length=self.max_length,padding="max_length")
            
            
            input_token=self.tokenizer(str(prompt),return_tensors="pt",max_length=self.max_length,padding="max_length")
        
        return {'image':image,"inputs":input_token["input_ids"],"attention_mask":input_token["attention_mask"],
                "target":target_actions["input_ids"]}
    
            
            

            
        
        

# ----------------------------
# 3. 示例使用
# ----------------------------
if __name__ == "__main__":
    # 1. 初始化一个 DummyTokenizer，假定词汇表大小为 300
   
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    

    # 2. 初始化 ActionTokenizer
    #    这里我们采用默认的 bins=256, 动作范围为 [-1, 1]
    action_tokenizer = ActionTokenizer(tokenizer=tokenizer, bins=256, min_action=-1, max_action=1)

    # 3. 示例1：处理单个动作向量
    #    假设机器人有一个 3 维动作（例如三个连续控制信号）
    single_action = np.array([ 4.7470e-02,  2.5889e-01,  1.3883e-01, -1.9776e+00, -5.7641e-02,
          2.1842e+00,  1.0042e+00])  # 注意：1.2 会被裁剪到 1
    tokenized_action_str = action_tokenizer(single_action)
    print("单个动作向量：", single_action)
    print("离散化（token化）后的字符串：", tokenized_action_str)

    # # 4. 示例2：处理批量动作（2 个动作，每个动作有 3 个维度）
    batch_actions = np.array([
        [0.5, -0.8, 1.2],
        [-1.5, 0.0, 0.3]  # -1.5 会被裁剪到 -1
    ])
    tokenized_actions_str = action_tokenizer(batch_actions)
    print("\n批量动作向量：")
    print(batch_actions)
    print("离散化（token化）后的字符串列表：", tokenized_actions_str)

    # 5. 示例3：将 token id 转换回连续动作值（近似为 bin 的中心值）
    #    假设我们获得了一组动作 token id（这里直接模拟调用 __call__ 得到的结果，并反向映射）
    #    先获得单个动作 token 的 id（注意：__call__ 返回的是解码后的字符串，
    #    但为了演示，我们手动计算 token id）
    #    对 single_action 来说：
    clipped = np.clip(single_action, a_min=-1, a_max=1)
    discretized = np.digitize(clipped, action_tokenizer.bins)  # 返回 [bin_idx1, bin_idx2, bin_idx3]
    # 根据 __call__ 的映射，token id = tokenizer.vocab_size - discretized
    token_ids = np.array(tokenizer.vocab_size - discretized)
    # 反向解码回连续动作值（取 bin_center 值）
    recovered_actions = action_tokenizer.decode_token_ids_to_actions(token_ids)
    print("\n原始单个动作（裁剪后）：", clipped)
    print("对应的离散化 bin 索引：", discretized)
    print("映射到 token id：", token_ids)
    print("根据 token id 还原的连续动作（bin center）：", recovered_actions)
