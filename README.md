# News Headline Sarcasm Classifier (BERT + Bi-LSTM)

A hybrid natural language processing (NLP) model designed to detect sarcasm in news headlines. This project combines the contextual embeddings of a pre-trained BERT model with a Bi-Directional LSTM to capture both semantic nuances and sequential flow, effectively distinguishing between satirical and literal news reporting. 

## Dataset
The model is trained on the "News Headlines Dataset for Sarcasm Detection" (Kaggle), which aggregates approximately 26,000 headlines from:
* **The Onion:** Sarcastic / Satirical (Target = 1)
* **HuffPost:** Real News (Target = 0)

Data was preprocessed by lowercasing, removing special characters, and tokenizing via `BertTokenizer` to align with the pre-trained vocabulary.

## Model Architecture
Sarcasm relies heavily on context; a single word can alter the entire meaning of a sentence. To capture this:
1. **BERT (`bert-base-uncased`):** Reads bidirectionally to capture deep contextual cues better than standard baseline models.
2. **Bi-Directional LSTM:** Processes BERT's output to capture the sequential, grammatical flow typical of journalistic headlines. 
3. **Hyperparameters:**
   * **Epochs:** 3 (to prevent overfitting)
   * **Learning Rate:** 2e-5 (preserves BERT's pre-trained weights)
   * **Batch Size:** 16 (optimized for GPU memory limits)
   * **Gradient Clipping:** 1.0 (stabilizes LSTM gradients)

## Performance & Evaluation
Accuracy was measured to track overall correctness, while the **F1 Score** was heavily monitored to account for the nuanced implications of false positives (flagging real news as sarcasm) and false negatives (missing the sarcasm).

* **Peak Validation Accuracy:** 91.58% (Epoch 2)
* **Peak Validation F1 Score:** 0.906 (Epoch 2)

*Note: Training loss plummeted from 0.14 to 0.07 between Epoch 2 and 3 alongside a slight drop in validation accuracy, indicating the onset of overfitting in the final epoch.*

## Use Cases & Social Impact
Beyond news analysis, sarcasm-aware NLP models have critical applications in:
* **Content Moderation:** Standard toxicity filters often miss hate speech or cyberbullying veiled as sarcasm. This architecture can improve the detection of nuanced malicious content.
* **Accessibility:** Can be integrated into assistive reading tools for neurodivergent individuals who may struggle to interpret tonal shifts in written text.

**Current Limitations:** The model is currently trained exclusively on formal journalistic structures. It may require domain adaptation before being applied to informal social media text containing slang, emojis, or fragmented grammar.

## Future Optimizations
* **Regularization:** Introduce Dropout layers and Weight Decay to further suppress the model's tendency to memorize training data by Epoch 3.
* **Inference Speed ("Frozen BERT"):** To make the model practical for deployment on high-volume real-time text streams, future iterations will freeze BERT's pre-trained weights and exclusively train the LSTM. This is projected to reduce runtime from 25 minutes to under 5 minutes.

---
*Note: This project was originally developed as an individual assessment for CAIS++ (USC Center for AI in Society's Student Branch) during the Winter term.*
