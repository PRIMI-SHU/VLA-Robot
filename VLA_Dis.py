import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler,random_split
import torch.multiprocessing as mp
from VLA_finetuning import *
import time
def training_function(model,progress_bar,optimizer,device):
    model.train()
    count=0
    total_loss=0
    start_time=time.time()
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
            
            progress_bar.set_postfix({"loss": loss.item()})
            total_loss+=loss.item()
            count+=1
    end_time=time.time()
    return total_loss/count,end_time-start_time

def testing_function(model,dataset,device):
    model.eval()
    total_loss=0
    count=0
    with torch.no_grad():
        for batch in dataset:
            images=batch["image"].to(device)
            inputs=batch["inputs"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            target=batch["target"].to(device)
            
            outputs=model(images,inputs.squeeze(1),attention_mask.squeeze(1),target.squeeze(1))
            
            loss=outputs["loss"]
            total_loss+=loss.item()
            count+=1
        avg_loss=total_loss/count
    print(f"Testing Loss {avg_loss:.2f}")


def setup(rank,gpu_size):
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]='12355'
    dist.init_process_group(backend="nccl",rank=rank,world_size=gpu_size)

def clean_up():
    dist.destroy_process_group()

def ddp_train(rank,gpu_size,num_epochs=10,batch_size=32,lr_rate=5e-5):
    setup(rank,gpu_size)
    try:
        torch.manual_seed(42)
        device=torch.device(f"cuda:{rank}")

        file_path='./mix_data.hfd5'
        dataset=HDF5Dataset(file_path,transforms,AutoTokenizer,ActionTokenizer,20)
        train_size=int(0.8*len(dataset))
        test_size=len(dataset)-train_size
        train_dataset,test_dataset=random_split(dataset,[train_size,test_size])
        train_sampler=DistributedSampler(dataset=train_dataset,num_replicas=gpu_size,rank=rank)
        test_sampler=DistributedSampler(dataset=test_dataset,num_replicas=gpu_size,rank=rank)

        train_loader=DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler)
        test_loader=DataLoader(test_dataset,batch_size=batch_size,sampler=test_sampler)


        model=VLMFineTuning()
        model=model.to(device)
        model=DDP(model,device_ids=[rank])
        optimizer=optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_rate)
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            avg_loss,compute_time=training_function(model,progress_bar,optimizer,device)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f},Time spend:{compute_time:.2f}")
            testing_function(model,test_loader,device)
        
        torch.save(model.state_dict(), "vlm_finetuned_model.pt")
    except Exception as e:
        print(f"Error in process {e}")
    finally:
        clean_up()


def main():
    gpu_size=torch.cuda.device_count()
    print(f"{gpu_size} GPUS availiable")
    mp.spawn(ddp_train,args=(gpu_size,),nprocs=gpu_size,join=True)

if __name__=="__main__":
    # clean_up()
    main()
