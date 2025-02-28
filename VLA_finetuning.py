import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel,CLIPVisionModel,AutoTokenizer,AutoModelForCausalLM,AutoModel
from action_tokennizer import *
from tqdm import tqdm
import torch.optim as optim
class VLMFineTuning(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32", 
                 llm_model_name="meta-llama/Llama-3.2-1B",
                 fusion_dim=1024, output_dim=1024):
        super().__init__()
        
        self.vision_model=CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_dim=self.vision_model.config.hidden_size
        
        
        self.llm=AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm_dim=self.llm.config.hidden_size
        
        for param in self.vision_model.parameters():
            param.requires_grad=False
        
        # llm_layers=list(self.llm.parameters())
        
        # num_trainable_layers = len(llm_layers) // 4  # 只训练最后1/4的层
        # for param in list(self.llm.parameters())[:-num_trainable_layers]:
        #     param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad=False
                
        
        
        self.fusion_mlp=nn.Sequential(nn.Linear(self.vision_dim,fusion_dim),
                        nn.GELU(),
                        nn.Linear(fusion_dim,self.llm_dim))
        
        
        self.output_mlp=nn.Sequential(nn.Linear(self.llm_dim,output_dim),
                                      nn.GELU(),
                                      nn.Linear(output_dim,self.llm.config.vocab_size))
    
    
    def forward(self,pixel,prompt,attention_mask,target):
        #vision process
        vision_input=self.vision_model(pixel_values=pixel)
        vision_emds=vision_input.last_hidden_state[:,0,:]
        
        
        #language process
        prompt_embeds=self.llm.model.embed_tokens(prompt)
        
        vision_projected=self.fusion_mlp(vision_emds)
        vision_projected=vision_projected.unsqueeze(1)
        
        fused_embds=torch.cat([vision_projected,prompt_embeds],dim=1)
        
        visual_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        fused_mask=torch.cat([visual_mask,attention_mask],dim=1)
        
        output=self.llm(inputs_embeds=fused_embds,attention_mask=fused_mask,output_hidden_states=True)
        hidden_state=output.hidden_states[-1]
        
        action=self.output_mlp(hidden_state)
       
        if target is not None:
            loss_fct = nn.CrossEntropyLoss()
            # shift logits 和 labels: 假设 target 的形状是 [batch, seq_len]
            shift_logits = action[:, 1:, :].contiguous()
            shift_labels = target.contiguous()
            loss = loss_fct(shift_logits.view(-1, self.llm.config.vocab_size), shift_labels.view(-1))
            return {"loss": loss, "logits": action}
            
    
        return action[:, 1:, :].contiguous()
        
        
        
        
if __name__=="__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    file_path='/docker-ros/local_ws/VLT/mutlimodal_VLT/mix_data.hfd5'
    data=HDF5Dataset(file_path,transforms,AutoTokenizer,ActionTokenizer,20)

    dataset=DataLoader(dataset=data,batch_size=8,shuffle=True,num_workers=2)
    
    learning_rate=5e-5
    
    
    
    model=VLMFineTuning()
    
    model=model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss=nn.CrossEntropyLoss()
    
    
    model.train()
    num_epoch=80
    for epoch in range(num_epoch):
        total_loss=0
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epoch}")
        for batch in progress_bar:
            images=batch["image"].to(device)
            inputs=batch["inputs"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            target=batch["target"].to(device)
            
            outputs=model(images,inputs.squeeze(1),attention_mask.squeeze(1),target.squeeze(1))
            
            loss=outputs["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epoch}, Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "vlm_finetuned_model.pt")
            
    
    
    
    
    
    
    
    
   