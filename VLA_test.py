from VLA_finetuning import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
    
file_path="VLT/mutlimodal_VLT/mix_data.hfd5"
data=HDF5Dataset(file_path,transforms,AutoTokenizer,ActionTokenizer,20)

dataset=DataLoader(dataset=data,batch_size=1,shuffle=True,num_workers=2)

model=VLMFineTuning()
model.load_state_dict(torch.load("/docker-ros/local_ws/vlm_finetuned_model.pt"))
model=model.to(device)
model.eval()

for i,batch in enumerate(dataset):
            images=batch["image"].to(device)
            inputs=batch["inputs"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            target=batch["target"].to(device)
            
            outputs=model(images,inputs.squeeze(1),attention_mask.squeeze(1),None)
            break
    
action=outputs.argmax(dim=-1)

action = action.cpu().detach().numpy()[0]

text=data.tokenizer.decode(batch["inputs"].cpu().detach().numpy()[0][0],skip_special_tokens=True)
print(f"task is {text}")
action=data.action_tokenizer.decode_token_ids_to_actions(action[1:8])
print(f"predicted action:{action}")


target=target.cpu().detach().numpy()

target=data.action_tokenizer.decode_token_ids_to_actions(target[0][0][1:8])
print(f"ground truth action:{target}")
# data.action_tokenizer.decode_token_ids()
# print(action)

