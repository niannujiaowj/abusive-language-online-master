# Authors:
  - [niannujiaowj](https://github.com/niannujiaowj)

  
# Options:
```bash
$ python mtl_cnn.py -h

       USAGE: mtl_cnn.py [flags]
flags:

mtl_cnn.py:
  --batch_size: batch size
    (default: '64')
    (an integer)
  --corpus: input corpus
    (default: 'rt.txt')
  --embed_dim: word embedding dimentionality
    (default: '200')
    (an integer)
  --embed_fn: embeddings file
    (default: 'text8-vector.bin')
  --epoches: number of epoches
    (default: '16')
    (an integer)
  --keep_prob: dropout keep rate
    (default: '0.5')
    (a number)
  --lr: learning rate
    (default: '0.1')
    (a number)
  --max_len: max sentence length
    (default: '124')
    (an integer)
  --n_tasks: number of tasks
    (default: '3')
    (an integer)
  --num_filter: number of feature maps
    (default: '100')
    (an integer)
  --task_idx: index of the task, 1 for toxicity, 2 for aggression 3 for attack
    (default: '1')
    (an integer)

Try --helpfull to get a list of all flags.
```


# Conference
[ALW3: 3rd Workshop on Abusive Language Online](https://sites.google.com/view/alw3/home)

# Tutorials
* [Multi-task Learning](https://www.cs.ubc.ca/~schmidtm/MLRG/Multi-task%20Learning.pdf)
* [Multi-task Learning](http://multi-task-learning.info/)

# References
* [DEEP MULTI-TASK LEARNING FOR TREE GENERA CLASSIFICATION ](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2/153/2018/isprs-annals-IV-2-153-2018.pdf])
* [Representation Learning Using Multi-Task Deep Neural Networks for Semantic Classification and Information Retrieval](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/mltdnn.pdf)
* [Convolutional Neural Network with Multi-Task Learning Scheme for Acoustic Scene Classification
](http://www.apsipa.org/proceedings/2017/CONTENTS/papers2017/15DecFriday/FA-02/FA-02.5.pdf)
* [Learning Multiple Tasks with Multilinear Relationship Networks](http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-multilinear-relationship-networks.pdf)
* [The Linguistic Ideologies of Deep Abusive Language Classification](https://www.aclweb.org/anthology/W18-5120)
* [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)
* [Using tenses in scientific writing](https://services.unimelb.edu.au/__data/assets/pdf_file/0009/471294/Using_tenses_in_scientific_writing_Update_051112.pdf)
* [Multitask Learning](https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf)
* [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/pdf/1605.05101.pdf)
