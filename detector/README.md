#Detector - AI image detector

## Setup
1. Install dependencies
2. run "python -m uvicorn app.api:app --reload"
3. go to "http:127.0.0.1:8000"
This will allow for upload and detection of deep-fake images.

## Training
1. run "python train.py"
The system will begin training on the dataset and will save the best model to the system for future use.