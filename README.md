# Sat2RDR

### release notes
```
- 2024.10.07 : Start Project
- 2024.10.18 : Add - probability head. (but optimization is not working) 
```


###  directory structure
```python
|
|-- data_utils
|-- dataset
|   |-- train
|   |   |-- trainA
|   |   `-- trainB
|   `-- val
|       |-- valA
|       `-- valB
|-- models
|   
`-- results_30

```
### How to train and eval the models
```python
# Train
python3 train.py
# Eval
python3 eval.py
```

### TODO LIST

- [ ] Add Various backbone models
- [x] Fix weight save code for hardware memory <---- **Doyi**
- [x] Add Probability head

### Sample Image

- Tanh + init (0, 1 channels)
![747_results](https://github.com/user-attachments/assets/7e5d1c09-266d-493c-8eb3-74378c96b561)
