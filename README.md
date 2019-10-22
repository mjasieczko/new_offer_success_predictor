# new_offer_success_predictor
provide list of probabilities for customers on who will accept new offer 

**task details**
<br> You can find it in `data/raw/description.txt` with one small change:
<br> instead list of customers who will accept new offer I will provide
<br> list of probabilities that customers will accept new offer :) 

**environment preparation**
<br> commands used to create and run project environment
<br> (also some of their outputs)
<br>`conda --version`
<br>`conda 4.7.12`


<br>`conda create --name new_offer_predictor python=3.6`
<br> `source activate new_offer_predictor`
<br> `pip install pandas `
<br> `pip install sklearn`
<br> `pip install jupyterlab`
<br> `pip install pyarrow` <- to read .parquet format files

<br> requirements were generated using pip freeze > requirements.txt