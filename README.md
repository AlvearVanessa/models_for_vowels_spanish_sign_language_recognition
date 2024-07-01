# models_for_vowels_spanish_sign_language_recognition

In this work, we presented a sign recognition system of the vowels of the Spanish Sign language (Lengua de Signos Española - LSE) in real-time, based on hand detection and image classification where each vowel is a class. The *vowelsLSE* dataset consists of 5 gestures of one person signing each vowel according to LSE and contains 3461 images. The *vowelsLSE_new_test* dataset has 197 images of five different people signing LSE vowels and was created to evaluate the models with different images rather than with training data from the same person. Both datasets consist of RGB images in JPG format with a size of 400 x 400 and have a white background to make them the same size. This dataset has been created as a proof of concept and is being worked on for improvement in future updates. This work was based on the Hand Sign Detection for the American Sign Language (ASL) course on the following website: https://www.computervision.zone/courses/hand-sign-detection-asl/

The repository includes:

    *inference_images* folder contains images for making inferences once the classification models are trained.
    
    *integration_codes* folder includes five Python scripts where the detection and classification models get merged to make the recognition of the vowels of the LSE:
    
      - DataCollection.py to collect and create the vowelsLSE dataset.          
      - HandTrackingModule_noSkeleton.py is the hand detection module and works for Keras and FastAI libraries.            
      - classificationModule_init_fastai.py is the classification module for the vowels of LSE works for the FastAI model.  
      - cuttingHand_new_test_data.py to cut, collect and create the vowelsLSE_new_test dataset.     
      - signRecognition_init_fastai.py is a recognition module for the vowels of LSE in real-time using the hand detection and image classification modules that work for the FastAI model.
      
      
    The *classification_models* folder has six image classification models using the FastAI library:    
    
      - Convnext_tiny.ipynb.        It is a model that applies ConvNeXt architecture.       
      - ResNet18.ipynb.             It is a model that applies ResNet18 architecture. 
      - ResNet50.ipynb.             It is a model that applies ResNet50 architecture. 
      - ViT_b_16.ipynb.             It is a model that applies ViT base 16 architecture. 
      - ViT_b_32.ipynb.             It is a model that applies ViT base 32 architecture. 
      - ensemble_best_models.ipynb. In this notebook, we create an Ensemble model with the best three models applied according to the results of the metrics.
      
    *notebook_images* folder has images used in the notebooks such as the transformations applied for data, and the samples of signs of the vowels of the LSE.

The corresponding datasets are in the following links: 
- *vowelsLSE*          :  https://unirioja-my.sharepoint.com/:f:/g/personal/maalvear_unirioja_es/EnatqT9WnMBMg22VVjbF354B3b468vw6zwivlqpSbABjZw?e=sOzRIP
- *vowelsLSE_new_test* :  https://unirioja-my.sharepoint.com/:f:/g/personal/maalvear_unirioja_es/Eq7UEiPeQvlOppvG_Fj1NgEBrlOwAIXGocOiVSC11JM0-w?e=HP3Qdl

For any further information: maalvear@unirioja.es
