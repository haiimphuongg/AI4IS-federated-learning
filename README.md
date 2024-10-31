# AI4IS-federated-learning
Repository for AI4IS subject - HCMUS 

- Run `preprocess.py` to export csv file of feature vectors. This helps clients read data faster during demo.
- Demo:

  1. For Server, run `python server.py`

  2. For each Client, run commands with the format (`client_id` is required):
  
      `python client.py [client-id] --method_extract ['hog', 'cnn'] --lr [learning-rate] --betas [betas] --weight_decay [wd] --num_epochs [num] --train_iterations [num]`
  
      Example: 
      
          python client.py 1 --method_extract cnn --lr 0.01 --betas 0.9 0.999 --weight_decay 0.0005 --num_epochs 1 --train_iterations 10
          python client.py 2 --method_extract cnn --lr 0.01 --betas 0.9 0.999 --weight_decay 0.0005 --num_epochs 1 --train_iterations 10
          python client.py 3 --method_extract cnn --lr 0.01 --betas 0.9 0.999 --weight_decay 0.0005 --num_epochs 1 --train_iterations 10
      
          python client.py 1 --method_extract hog --lr 0.001 --betas 0.9 0.999 --weight_decay 0.0001 --num_epochs 1 --train_iterations 10
          python client.py 2 --method_extract hog --lr 0.001 --betas 0.9 0.999 --weight_decay 0.0001 --num_epochs 1 --train_iterations 10
          python client.py 3 --method_extract hog --lr 0.001 --betas 0.9 0.999 --weight_decay 0.0001 --num_epochs 1 --train_iterations 10
