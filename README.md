Tina Abraham 
cvabraha@usc.edu

## Project Description:  
    To address the complexity of detecting sarcasm in news headlines, I developed a hybrid Binary NLP Classifier that combines BERT's contextual embeddings with a Bi-Directional LSTM for sequential analysis. This architecture captures both semantic and sentence nuances to accurately distinguish between sarcastic and non-sarcastic headlines for improved sentiment analysis.
  
## Dataset:
    This data comes from the "News Headlines Dataset for Sarcasm Detection" by Rishabh Misra on Kaggle. The dataset aggregates headlines from two primary sources: The Onion (sarcastic/satirical) and HuffPost (real news).

    The dataset contains approximately 26,000 rows with two main features: headline (the text input) and is_sarcastic (the binary target label, where 1 = sarcastic and 0 = not sarcastic). It was cleaned (lowercased with special characters removed) and tokenized using the BertTokenizer to match the pre-trained model's vocabulary.

## Model Choices and Hyperparameters:
    Epochs -- 3 to prevent overfitting; BERT learns fast
    Learning Rate -- 2e-5 (small to preserve BERT's pre-trained knowledge)
    Batch Size -- 16 (smaller to fit into GPU memory without crashing, but still work with the large data)
    Grad Clipping -- 1.0 to prevent overly large gradients in the LSTM layer
    (These parameters created a 25-minute runtime.)

## Model Evaluation:
```
Epoch 1/3
----------
Average Training Loss: 0.2988251569941584
Validation Accuracy: 0.9112691875701985
Validation F1 Score: 0.8963707914298207

Epoch 2/3
----------
Average Training Loss: 0.14928817978725856
Validation Accuracy: 0.9157618869337327
Validation F1 Score: 0.9069093918080264

Epoch 3/3
----------
Average Training Loss: 0.0716079566524919
Validation Accuracy: 0.9150131037064769
Validation F1 Score: 0.8991559306974678
```

## Model Analysis:

### Why did you use these metrics? 
    I selected Accuracy as the main measure to see how often the model is correct overall. However, I also used an F1 Score (the mean of precision and recall). This is because in sarcasm detection, false positives (flagging real news as sarcasm) and false negatives (missing the sarcasm) can have different implications. The F1 Score ensures the model isn't just guessing the majority class and provides a view of how well the classifier distinguishes between the subtle tonal shifts of the two categories.


### How well does your dataset, model architecture, training procedures, and chosen metrics fit the task at hand? 
    The fit is strong. Sarcasm relies heavily on context because even a single word changes the meaning of the entire sentence (e.g., "Man _finally_ finishes project"). BERT is ideal for this because it reads left-to-right and right-to-left, allowing it to catch these contextual cues better than standard models. Adding the Bi-LSTM then also helps capture the sequential flow typical of news headlines. The results (91.5% accuracy) show the model learned rapidly. However, the slight drop in Validation Accuracy between Epoch 2 (91.58%) and Epoch 3 (91.50%), contrasted with the plummeting Training Loss (from 0.14 to 0.07), indicates the model began to overfit in the final epoch.


### Can your efforts be extended to wider implications, or contribute to social good? Are there any limitations in your methods that should be considered before doing so?
    This technology can be used for content moderation because hate speech and cyberbullying can sometimes be seen in sarcasm. Standard toxicity filters miss this, but a sarcasm-aware model could detect it, creating safer online spaces. It can also aid accessibility tools for neurodivergent individuals who struggle to interpret tone in written texts.

    However, a limitation to its wider application is that this model is trained specifically on news headlines (The Onion/HuffPost) only. This model may struggle to detect sarcasm in informal social media text that relies on slang, emojis, images, or different grammatical structures (e.g. the "I guess bro" meme image).


### If you were to continue this project, what would be your next steps?
    The main improvements I would make would be to reduce overfitting potential and improve time efficiency. 91.5% accuracy doesn't inherently suggest overfitting, but I would introduce Dropout layers or Weight Decay to prevent the model from memorizing the training data, as seen in the last epoch. Then, I would try to implement a "Frozen BERT" structure in the model to improve runtime. By freezing the pre-trained weights and only training the LSTM, I could reduce the runtime from 25 minutes to under 5 minutes, making the model more practical for deployment on high-volume/rapid text streams (e.g. social media comments and posts).
