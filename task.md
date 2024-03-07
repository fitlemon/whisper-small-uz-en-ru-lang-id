Test task for data science position:
Audio Classification for Language Identification
Objective: Train a model that can classify audio clips into one of three languages: Uzbek, Russian, or English.
Input: Audio Clip Output: Probabilities for the three classes (uz, ru, en)
Technical Requirements:
Model Choice: You can choose any model architecture, but it would be good if model architecture implementation will be present in official Hugging Face Transformers repository. Our results show best accuracy on opensource Whisper model.
Preprocessing:
Download the Common Voice datasets for Uzbek, Russian, and English. Or find a way to stream those datasets during training
Standardize audio lengths if necessary or decide on a strategy for handling variable-length audio clips.
Ensure a balanced dataset, or devise strategies to handle class imbalance.
Training:
Use pytorch.
Split the merged dataset into training, validation, and test sets. Make sure test and validation samples are not present in training. Try to remove similar sentences from validation set.
Fine-tune the selected model on the training dataset, validating its performance to avoid overfitting.
Evaluation: Your primary metric should be accuracy.
Reporting
Describe the preprocessing steps and any challenges faced.
Discuss model performance, any tuning performed, and the final results on the test set.
Propose further improvements or other techniques/models that might be worth exploring.
Dataset Source: Common Voice Multilanguage dataset Languages to Download: Uzbek, Russian, English.
Classes uz: Uzbek ru: Russian en: English
Checkpoints: Start with any suitable pre-trained model available on Hugging Face's model hub. It's crucial to leverage transfer learning to achieve better results with potentially less data.
Deliverables: A Jupyter Notebook or Python script with all the steps: data preprocessing, model training, evaluation, and reporting. A short report or presentation summarizing your methodology, findings, and recommendations. Trained model weights and architecture. Upload you model weights to hugging face and give a link.
Bonus Points: Efficiency: Implement strategies to speed up training without compromising the model's performance. Visualization: Visualize audio data, model's attention, or any other interesting aspect. Interactivity: Implement a simple user interface or an API endpoint where a user can upload an audio clip and get its predicted language.