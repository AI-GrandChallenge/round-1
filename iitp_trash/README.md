# IITP_TRASH
IITP trash recognition challenge

Dataset: `tr-3`
* num_test_images: 3000
* num_test_classes: 8

How to run:

```bash
nsml run -d tr-3 -g 1 --memory 12G -e main.py -a "--mode test"
```

How to list checkpoints saved:

```bash
nsml model ls [sessionName]
```

How to submit:

```bash
nsml submit [sessionName] [checkpoint]
```
